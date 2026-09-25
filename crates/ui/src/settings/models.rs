//! Per-agent model list preferences inside Settings → Providers: hide catalog
//! models from the picker, drag the visible ones into order, and add (or
//! later edit) custom models by backend id under a display name.
//!
//! The preferences are device-local (`ui-settings.json`) and every picker
//! re-applies them on change ([`crate::pickers::bump_model_preferences`]).
//! The catalog shown here comes from the page's target device, like the rest
//! of the page.
//!
//! Reordering follows the queue's drag treatment: an invisible ghost, the
//! dragged row lifted and travelling slot to slot, and every row in its path
//! sliding into the space it leaves.

use gpui::{
    AnyElement, Context, Entity, Focusable as _, IntoElement, Render, SharedString, Subscription,
    Window, div, prelude::*, px,
};
use zeron_proto::{HarnessId, Model};
use zeron_rpc::methods;

use super::HarnessesPage;
use crate::composer::{ComposerInput, ComposerInputEvent};
use crate::icons::{self, icon};
use crate::motion::{self, AnimationExt as _, TAB_SLIDE};
use crate::popover::{self, Loadable};
use crate::settings::{self, ModelPreferences, widgets};
use crate::terminal::panel::{drop_index, slide_offset};
use crate::theme::Theme;

/// Fixed row height: the drag maps the pointer to slots in these units.
const MODEL_ROW_HEIGHT: f32 = 52.0;

pub(super) struct CustomModelDialog {
    harness: HarnessId,
    /// The id of the custom model being edited; `None` adds a new one.
    editing: Option<String>,
    label: Entity<ComposerInput>,
    id: Entity<ComposerInput>,
    error: Option<SharedString>,
    focus_pending: bool,
    _events: [Subscription; 2],
}

/// A visible model row being dragged within one agent's list.
pub(super) struct ModelDragPayload {
    harness: HarnessId,
    from: usize,
}

/// Where the dragged row would land. `prev_over` restarts the slide from the
/// row's current visual slot; `epoch` keys each slide's animation.
pub(super) struct ModelDrag {
    harness: HarnessId,
    from: usize,
    over: usize,
    prev_over: usize,
    epoch: usize,
}

/// Invisible cursor ghost: the real row moves between slots instead of a
/// detached copy following the pointer.
struct ModelDragGhost;

impl Render for ModelDragGhost {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        gpui::Empty
    }
}

/// Start and target offsets (px) for row `ix` while `from` is dragged from
/// `prev_over` to `over`: the dragged row travels to the hovered slot, and
/// rows in its path slide into the space it leaves.
fn drag_offsets(ix: usize, from: usize, prev_over: usize, over: usize) -> (f32, f32) {
    if ix == from {
        (
            (prev_over as f32 - from as f32) * MODEL_ROW_HEIGHT,
            (over as f32 - from as f32) * MODEL_ROW_HEIGHT,
        )
    } else {
        (
            slide_offset(ix, from, prev_over) * MODEL_ROW_HEIGHT,
            slide_offset(ix, from, over) * MODEL_ROW_HEIGHT,
        )
    }
}

/// What a custom id must look like for `harness`, when it is constrained.
fn custom_id_hint(harness: HarnessId) -> Option<&'static str> {
    match harness {
        HarnessId::ClaudeCode | HarnessId::Codex => None,
        HarnessId::Opencode => {
            Some("OpenCode model ids are provider/model, e.g. anthropic/claude-sonnet-5.")
        }
        _ => Some("This agent only runs models it lists, so it may reject a custom id."),
    }
}

/// Why `id` can't be a custom model id for `harness`, if it can't.
fn custom_id_error(harness: HarnessId, id: &str) -> Option<&'static str> {
    if id.is_empty() {
        Some("Enter the model id the agent runs.")
    } else if id.chars().any(char::is_whitespace) {
        Some("Model ids can't contain spaces.")
    } else if harness == HarnessId::Opencode && !id.contains('/') {
        Some("OpenCode model ids are provider/model.")
    } else {
        None
    }
}

impl HarnessesPage {
    /// `ListModels` for `harness` on the target device, normalized the way
    /// the picker shows it (so ids and labels match what users see there).
    pub(super) fn load_models(&mut self, harness: HarnessId, force: bool, cx: &mut Context<Self>) {
        if !force && self.models.contains_key(&harness) {
            return;
        }
        let Some(engine) = self.state.read(cx).engine().cloned() else {
            return;
        };
        let params = self.with_target(serde_json::json!({ "harness": harness, "force": force }));
        if !matches!(self.models.get(&harness), Some(Loadable::Ready(_))) {
            self.models.insert(harness, Loadable::Loading);
        }
        let task = cx.spawn(async move |this, cx| {
            let result = engine.client().call(methods::LIST_MODELS, params).await;
            this.update(cx, |page, cx| {
                let loaded = match result {
                    Ok(value) => match serde_json::from_value::<Vec<Model>>(value) {
                        Ok(models) => {
                            Loadable::Ready(crate::pickers::normalize_model_rows(harness, models))
                        }
                        Err(err) => Loadable::Error(err.to_string()),
                    },
                    Err(err) => Loadable::Error(err.to_string()),
                };
                page.models.insert(harness, loaded);
                page.model_tasks.remove(&harness);
                cx.notify();
            })
            .ok();
        });
        self.model_tasks.insert(harness, task);
        cx.notify();
    }

    fn update_model_preferences(
        &mut self,
        harness: HarnessId,
        cx: &mut Context<Self>,
        change: impl FnOnce(&mut ModelPreferences),
    ) {
        settings::update(settings::SavePolicy::Immediate, cx, |settings| {
            let mut preferences = settings.model_preferences(harness);
            change(&mut preferences);
            if preferences == ModelPreferences::default() {
                settings.model_preferences_by_harness.remove(&harness);
            } else {
                settings
                    .model_preferences_by_harness
                    .insert(harness, preferences);
            }
        });
        crate::pickers::bump_model_preferences(cx);
        cx.notify();
    }

    /// Track the drop slot while a row is dragged over its list.
    fn update_model_drag(
        &mut self,
        harness: HarnessId,
        from: usize,
        over: usize,
        cx: &mut Context<Self>,
    ) {
        match &mut self.model_drag {
            Some(drag) if drag.harness == harness && drag.from == from => {
                if drag.over != over {
                    drag.prev_over = drag.over;
                    drag.over = over;
                    drag.epoch = drag.epoch.wrapping_add(1);
                    cx.notify();
                }
            }
            _ => {
                self.model_drag = Some(ModelDrag {
                    harness,
                    from,
                    over,
                    prev_over: from,
                    epoch: 0,
                });
                cx.notify();
            }
        }
    }

    fn cancel_model_drag(&mut self, cx: &mut Context<Self>) {
        if self.model_drag.take().is_some() {
            cx.notify();
        }
    }

    /// A drop outside the list ends gpui's drag without our `on_drop`: never
    /// leave rows parked in their slid positions.
    pub(super) fn settle_model_drag(&mut self, cx: &mut Context<Self>) {
        if self.model_drag.is_some() && !cx.has_active_drag() {
            self.model_drag = None;
        }
    }

    /// Open the custom model dialog: empty to add one, or filled with an
    /// existing custom model's name and id to edit it.
    fn open_custom_model(
        &mut self,
        harness: HarnessId,
        existing: Option<(String, String)>,
        cx: &mut Context<Self>,
    ) {
        let label = cx.new(|cx| ComposerInput::new("Display name, e.g. Mythos 5.1", cx));
        let id = cx.new(|cx| ComposerInput::new("Model id, e.g. claude-mythos-5-1", cx));
        if let Some((existing_id, existing_label)) = &existing {
            label.update(cx, |input, cx| input.set_text(existing_label.clone(), cx));
            id.update(cx, |input, cx| input.set_text(existing_id.clone(), cx));
        }
        let submit = |this: &mut Self,
                      _: Entity<ComposerInput>,
                      event: &ComposerInputEvent,
                      cx: &mut Context<Self>| {
            if matches!(event, ComposerInputEvent::Submitted) {
                this.submit_custom_model(cx);
            }
        };
        let events = [cx.subscribe(&label, submit), cx.subscribe(&id, submit)];
        self.custom_model = Some(CustomModelDialog {
            harness,
            editing: existing.map(|(existing_id, _)| existing_id),
            label,
            id,
            error: None,
            focus_pending: true,
            _events: events,
        });
        cx.notify();
    }

    fn submit_custom_model(&mut self, cx: &mut Context<Self>) {
        let Some(dialog) = self.custom_model.as_mut() else {
            return;
        };
        let harness = dialog.harness;
        let id = dialog.id.read(cx).text().trim().to_string();
        let label = dialog.label.read(cx).text().trim().to_string();
        if let Some(error) = custom_id_error(harness, &id) {
            dialog.error = Some(error.into());
            cx.notify();
            return;
        }
        let editing = dialog.editing.clone();
        self.custom_model = None;
        self.update_model_preferences(harness, cx, |preferences| match editing {
            Some(old_id) => preferences.edit_custom(&old_id, &id, &label),
            None => {
                preferences.upsert_custom(&id, &label);
                preferences.set_hidden(&id, false);
            }
        });
    }

    pub(super) fn render_models_for(
        &self,
        harness: HarnessId,
        theme: &Theme,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let preferences = settings::current(cx).model_preferences(harness);
        let muted = |text: SharedString| {
            div()
                .py(px(8.0))
                .text_size(crate::typography::ui_rems(12.0))
                .text_color(theme.text_muted)
                .child(text)
        };
        let mut section = div()
            .flex()
            .flex_col()
            .child(widgets::details_label(theme, "Models"));
        let catalog = match self.models.get(&harness) {
            Some(Loadable::Ready(models)) => Some(models.clone()),
            Some(Loadable::Error(message)) => {
                section = section.child(
                    muted(format!("Couldn't load this agent's models: {message}").into()).child(
                        widgets::text_action(theme, widgets::ActionTone::Quiet, "Retry")
                            .id(format!("models-retry-{harness:?}"))
                            .mt(px(6.0))
                            .on_click(cx.listener(move |page, _, _, cx| {
                                page.load_models(harness, true, cx)
                            })),
                    ),
                );
                // Custom models still list (and stay editable) without it.
                Some(Vec::new())
            }
            _ => {
                section = section.child(muted("Loading models…".into()));
                None
            }
        };
        if let Some(catalog) = catalog {
            let models = preferences.apply(catalog);
            let visible = preferences.visible_len(&models);
            let listed: Vec<String> = models[..visible].iter().map(|m| m.id.clone()).collect();
            let drag = self
                .model_drag
                .as_ref()
                .filter(|drag| drag.harness == harness)
                .map(|drag| (drag.from, drag.over, drag.prev_over, drag.epoch));
            // The visible rows are the drop zone; hidden rows sit below it,
            // out of the order until they're shown again.
            let mut draggable = div()
                .id(SharedString::from(format!("models-visible-{harness:?}")))
                .flex()
                .flex_col()
                .on_drag_move::<ModelDragPayload>(cx.listener(
                    move |page, event: &gpui::DragMoveEvent<ModelDragPayload>, _, cx| {
                        let payload = event.drag(cx);
                        if payload.harness != harness {
                            return;
                        }
                        let from = payload.from;
                        let rel_y =
                            f32::from(event.event.position.y) - f32::from(event.bounds.top());
                        let over = drop_index(rel_y, MODEL_ROW_HEIGHT, visible);
                        page.update_model_drag(harness, from, over, cx);
                    },
                ))
                .on_drop::<ModelDragPayload>(cx.listener(
                    move |page, payload: &ModelDragPayload, _, cx| {
                        let from = payload.from;
                        let to = page
                            .model_drag
                            .take()
                            .filter(|drag| drag.harness == harness && payload.harness == harness)
                            .map(|drag| drag.over);
                        match to {
                            Some(to) if to != from => {
                                let listed = listed.clone();
                                page.update_model_preferences(harness, cx, |preferences| {
                                    preferences.reorder(&listed, from, to)
                                });
                            }
                            _ => cx.notify(),
                        }
                    },
                ))
                .on_mouse_up_out(
                    gpui::MouseButton::Left,
                    cx.listener(|page, _, _, cx| page.cancel_model_drag(cx)),
                );
            for (ix, model) in models[..visible].iter().enumerate() {
                draggable = draggable.child(self.render_model_row(
                    harness,
                    model,
                    ix,
                    visible,
                    &preferences,
                    drag,
                    theme,
                    cx,
                ));
            }
            section = section.child(draggable);
            for (ix, model) in models.iter().enumerate().skip(visible) {
                section = section.child(self.render_model_row(
                    harness,
                    model,
                    ix,
                    visible,
                    &preferences,
                    None,
                    theme,
                    cx,
                ));
            }
        }
        section = section.child(
            div()
                .pt(px(10.0))
                .flex()
                .flex_col()
                .gap(px(6.0))
                .child(
                    widgets::ghost_action(theme)
                        .id(format!("models-add-{harness:?}"))
                        .tab_index(0)
                        .role(gpui::Role::Button)
                        .focus_visible(|s| s.border_2().border_color(theme.accent).opacity(1.0))
                        .on_click(cx.listener(move |page, _, _, cx| {
                            page.open_custom_model(harness, None, cx)
                        }))
                        .child(
                            icon(icons::PLUS)
                                .size(px(14.0))
                                .text_color(theme.text_muted),
                        )
                        .child("Add custom model"),
                )
                .when_some(custom_id_hint(harness), |el, hint| {
                    el.child(
                        div()
                            .text_size(crate::typography::ui_rems(11.0))
                            .text_color(theme.text_faint)
                            .child(hint),
                    )
                }),
        );
        section.into_any_element()
    }

    #[allow(clippy::too_many_arguments)]
    fn render_model_row(
        &self,
        harness: HarnessId,
        model: &Model,
        ix: usize,
        visible: usize,
        preferences: &ModelPreferences,
        drag: Option<(usize, usize, usize, usize)>,
        theme: &Theme,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let shown = ix < visible;
        let custom = preferences.is_custom(&model.id);
        // The picker needs something to list: the last visible model stays.
        let can_toggle = !shown || visible > 1;
        let draggable = shown && visible > 1;
        let lifted = drag.is_some_and(|(from, ..)| from == ix);
        let key = format!("{harness:?}-{}", model.id);
        let mut meta = vec![
            div()
                .child(SharedString::from(model.id.clone()))
                .into_any_element(),
        ];
        if custom {
            meta.push(div().child("Custom").into_any_element());
        }
        let icon_button = |id: String, glyph: &'static str, label: &'static str| {
            div()
                .id(SharedString::from(id))
                .size(px(26.0))
                .flex_none()
                .flex()
                .items_center()
                .justify_center()
                .rounded(px(6.0))
                .role(gpui::Role::Button)
                .aria_label(label)
                .cursor_pointer()
                .tab_index(0)
                .hover(|s| s.bg(theme.element_hover))
                .focus_visible(|s| s.border_2().border_color(theme.accent))
                .child(icon(glyph).size(px(14.0)).text_color(theme.text_muted))
        };
        let toggle = move |page: &mut Self, cx: &mut Context<Self>, id: String| {
            if can_toggle {
                page.update_model_preferences(harness, cx, |preferences| {
                    preferences.set_hidden(&id, shown);
                });
            }
        };
        let (toggle_click, toggle_key) = (model.id.clone(), model.id.clone());
        let mut row = div()
            .id(SharedString::from(format!("model-row-{key}")))
            .h(px(MODEL_ROW_HEIGHT))
            .flex_none()
            .px(px(6.0))
            .flex()
            .flex_row()
            .items_center()
            .gap(px(8.0))
            .rounded(px(8.0))
            .when(ix > 0 && !lifted, |row| {
                row.border_t_1().border_color(widgets::row_divider(theme))
            })
            .when(lifted, |row| row.bg(theme.element_hover).shadow_md())
            .when(draggable, |row| {
                row.cursor(if lifted {
                    gpui::CursorStyle::ClosedHand
                } else {
                    gpui::CursorStyle::OpenHand
                })
                .on_drag(ModelDragPayload { harness, from: ix }, |_, _, _, cx| {
                    cx.stop_propagation();
                    cx.new(|_| ModelDragGhost)
                })
            })
            .child(div().w(px(14.0)).flex_none().when(draggable, |el| {
                el.child(
                    icon(icons::DRAG_HANDLE)
                        .size(px(14.0))
                        .text_color(theme.text_muted.opacity(if lifted { 0.9 } else { 0.45 })),
                )
            }))
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .when(!shown, |el| el.opacity(0.55))
                    .child(widgets::row_title(theme, model.label.clone()))
                    .child(widgets::meta_line(theme, meta)),
            );
        if custom {
            let (edit_id, edit_label) = (model.id.clone(), model.label.clone());
            let remove_id = model.id.clone();
            row = row
                .child(
                    icon_button(format!("model-edit-{key}"), icons::PEN, "Edit custom model")
                        .on_click(cx.listener(move |page, _, _, cx| {
                            page.open_custom_model(
                                harness,
                                Some((edit_id.clone(), edit_label.clone())),
                                cx,
                            )
                        })),
                )
                .child(
                    icon_button(
                        format!("model-remove-{key}"),
                        icons::TRASH_BIN_MINIMALISTIC,
                        "Remove custom model",
                    )
                    .on_click(cx.listener(move |page, _, _, cx| {
                        let id = remove_id.clone();
                        page.update_model_preferences(harness, cx, |preferences| {
                            preferences.remove_custom(&id)
                        });
                    })),
                );
        }
        let row = row.child(
            div()
                .id(SharedString::from(format!("model-show-{key}")))
                .flex_none()
                .role(gpui::Role::Switch)
                .aria_label(format!("Show {} in the model picker", model.label))
                .aria_toggled(if shown {
                    gpui::Toggled::True
                } else {
                    gpui::Toggled::False
                })
                .when(can_toggle, |el| {
                    el.cursor_pointer()
                        .tab_index(0)
                        .focus_visible(|s| s.border_2().border_color(theme.accent))
                })
                .when(!can_toggle, |el| el.opacity(0.4))
                .on_click(cx.listener(move |page, _, _, cx| toggle(page, cx, toggle_click.clone())))
                .on_key_down(cx.listener(move |page, event: &gpui::KeyDownEvent, _, cx| {
                    if !event.is_held && matches!(event.keystroke.key.as_str(), "enter" | "space") {
                        toggle(page, cx, toggle_key.clone());
                        cx.stop_propagation();
                    }
                }))
                .child(widgets::toggle_switch(
                    theme,
                    shown,
                    format!("model-switch-{key}"),
                )),
        );

        let Some((from, over, prev_over, epoch)) = drag else {
            return row.into_any_element();
        };
        let (start, target) = drag_offsets(ix, from, prev_over, over);
        if motion::reduced_motion(cx) {
            return div()
                .relative()
                .top(px(target))
                .child(row)
                .into_any_element();
        }
        div()
            .child(row)
            .with_animation(
                ("model-row-slide", (ix as u64) | ((epoch as u64) << 32)),
                TAB_SLIDE.animation(),
                move |el, t| el.relative().top(px(motion::lerp(start, target, t))),
            )
            .into_any_element()
    }

    pub(super) fn render_custom_model_dialog(
        &mut self,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Option<AnyElement> {
        let theme = Theme::of(cx).for_popup();
        let dialog = self.custom_model.as_mut()?;
        if std::mem::take(&mut dialog.focus_pending) {
            window.focus(&dialog.label.focus_handle(cx), cx);
        }
        let (label, id, error) = (
            dialog.label.clone(),
            dialog.id.clone(),
            dialog.error.clone(),
        );
        let (title, action) = if dialog.editing.is_some() {
            ("Edit custom model", "Save")
        } else {
            ("Add custom model", "Add")
        };
        let field_label = |text: &'static str| {
            div()
                .mt(px(12.0))
                .mb(px(6.0))
                .text_size(crate::typography::ui_rems(12.0))
                .text_color(theme.text_muted)
                .child(text)
        };
        let card = popover::dialog_card(&theme)
            .id("custom-model-card")
            .role(gpui::Role::Dialog)
            .aria_label(title)
            .on_key_down(cx.listener(|this, event: &gpui::KeyDownEvent, _, cx| {
                if event.keystroke.key == "escape" {
                    this.custom_model = None;
                    cx.notify();
                    cx.stop_propagation();
                }
            }))
            .child(popover::dialog_title(&theme, title))
            .child(field_label("Display name"))
            .child(popover::dialog_field(label.into_any_element()))
            .child(field_label("Model id"))
            .child(popover::dialog_field(id.into_any_element()))
            .child(
                div()
                    .mt(px(8.0))
                    .text_size(crate::typography::ui_rems(11.0))
                    .text_color(theme.text_faint)
                    .child("The picker shows the display name; the agent receives the model id."),
            )
            .when_some(error, |el, error| {
                el.child(
                    div()
                        .mt(px(8.0))
                        .text_size(crate::typography::ui_rems(12.0))
                        .text_color(theme.danger)
                        .child(error),
                )
            })
            .child(
                div()
                    .mt(px(16.0))
                    .flex()
                    .flex_row()
                    .justify_end()
                    .gap(px(8.0))
                    .child(
                        widgets::text_action(&theme, widgets::ActionTone::Quiet, "Cancel")
                            .id("custom-model-cancel")
                            .tab_index(0)
                            .role(gpui::Role::Button)
                            .focus_visible(|s| s.border_2().border_color(theme.accent).opacity(1.0))
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.custom_model = None;
                                cx.notify();
                            })),
                    )
                    .child(
                        widgets::text_action(&theme, widgets::ActionTone::Solid, action)
                            .id("custom-model-save")
                            .tab_index(0)
                            .role(gpui::Role::Button)
                            .focus_visible(|s| s.border_2().border_color(theme.accent).opacity(1.0))
                            .on_click(cx.listener(|this, _, _, cx| this.submit_custom_model(cx))),
                    ),
            )
            .into_any_element();
        Some(popover::modal(
            "custom-model-dialog",
            window.viewport_size(),
            card,
        ))
    }
}

#[cfg(test)]
mod model_preferences_tests {
    use super::*;

    fn model(id: &str) -> Model {
        Model {
            id: id.into(),
            label: id.to_uppercase(),
            description: None,
            reasoning_levels: Vec::new(),
            options: Vec::new(),
        }
    }

    fn ids(models: &[Model]) -> Vec<&str> {
        models.iter().map(|m| m.id.as_str()).collect()
    }

    fn visible_ids(preferences: &ModelPreferences, catalog: &[Model]) -> Vec<String> {
        let models = preferences.apply(catalog.to_vec());
        let visible = preferences.visible_len(&models);
        models[..visible].iter().map(|m| m.id.clone()).collect()
    }

    #[test]
    fn preferences_hide_order_and_add_models() {
        let catalog = vec![model("fable-5-1"), model("fable-5"), model("opus-5-5")];
        let mut preferences = ModelPreferences::default();
        assert_eq!(
            ids(&preferences.apply(catalog.clone())),
            ["fable-5-1", "fable-5", "opus-5-5"]
        );

        preferences.set_hidden("fable-5", true);
        let models = preferences.apply(catalog.clone());
        assert_eq!(
            ids(&models),
            ["fable-5-1", "opus-5-5", "fable-5"],
            "hidden rows sort last"
        );
        assert_eq!(preferences.visible_len(&models), 2);

        preferences.upsert_custom("claude-mythos-5-1", "Mythos 5.1");
        let models = preferences.apply(catalog.clone());
        assert_eq!(
            ids(&models),
            ["fable-5-1", "opus-5-5", "claude-mythos-5-1", "fable-5"]
        );
        assert_eq!(models[2].label, "Mythos 5.1");
        assert_eq!(
            models[2].description.as_deref(),
            Some(settings::CUSTOM_MODEL_DESCRIPTION)
        );

        // Dragging the custom model from the last visible slot to the top.
        let listed = visible_ids(&preferences, &catalog);
        preferences.reorder(&listed, 2, 0);
        assert_eq!(
            ids(&preferences.apply(catalog.clone())),
            ["claude-mythos-5-1", "fable-5-1", "opus-5-5", "fable-5"],
            "the first row is the default the picker uses"
        );
        // Out-of-range and no-op drops leave the order alone.
        let before = preferences.order.clone();
        preferences.reorder(&listed, 0, 9);
        preferences.reorder(&listed, 1, 1);
        assert_eq!(preferences.order, before);

        // A custom entry naming a catalog id relabels it instead of duplicating.
        preferences.upsert_custom("opus-5-5", "My Opus");
        let models = preferences.apply(catalog.clone());
        assert_eq!(models.len(), 4);
        assert_eq!(models[2].label, "My Opus");

        preferences.remove_custom("claude-mythos-5-1");
        assert!(!ids(&preferences.apply(catalog)).contains(&"claude-mythos-5-1"));
        assert!(!preferences.order.contains(&"claude-mythos-5-1".to_string()));
    }

    #[test]
    fn editing_a_custom_model_keeps_its_place_and_state() {
        let catalog = vec![model("fable-5-1"), model("opus-5-5")];
        let mut preferences = ModelPreferences::default();
        preferences.upsert_custom("claude-mythos-5", "Mythos 5");
        preferences.upsert_custom("claude-other-1", "Other");
        let listed = visible_ids(&preferences, &catalog);
        preferences.reorder(&listed, 2, 0);
        preferences.set_hidden("claude-other-1", true);

        preferences.edit_custom("claude-mythos-5", "claude-mythos-5-1", "Mythos 5.1");
        let models = preferences.apply(catalog.clone());
        assert_eq!(models[0].id, "claude-mythos-5-1", "keeps its dragged place");
        assert_eq!(models[0].label, "Mythos 5.1");
        assert!(!ids(&models).contains(&"claude-mythos-5"));
        assert_eq!(preferences.custom.len(), 2);

        preferences.edit_custom("claude-other-1", "claude-other-2", "");
        assert!(
            preferences.is_hidden("claude-other-2"),
            "keeps its hidden state"
        );
        let other = preferences
            .custom
            .iter()
            .find(|c| c.id == "claude-other-2")
            .unwrap();
        assert_eq!(
            other.label, "claude-other-2",
            "an empty name falls back to the id"
        );

        // Renaming onto another custom model's id replaces that entry.
        preferences.edit_custom("claude-other-2", "claude-mythos-5-1", "Merged");
        assert_eq!(preferences.custom.len(), 1);
        assert_eq!(preferences.custom[0].label, "Merged");
        assert_eq!(
            preferences
                .order
                .iter()
                .filter(|id| *id == "claude-mythos-5-1")
                .count(),
            1
        );
    }

    #[test]
    fn drag_offsets_slide_rows_toward_the_gap() {
        // Dragging row 0 down to slot 2: it travels two rows, the two it
        // passes each slide up one.
        assert_eq!(drag_offsets(0, 0, 0, 2), (0.0, 2.0 * MODEL_ROW_HEIGHT));
        assert_eq!(drag_offsets(1, 0, 0, 2), (0.0, -MODEL_ROW_HEIGHT));
        assert_eq!(drag_offsets(2, 0, 0, 2), (0.0, -MODEL_ROW_HEIGHT));
        assert_eq!(drag_offsets(3, 0, 0, 2), (0.0, 0.0));
        // Backing off to slot 1 restarts each slide from where it was.
        assert_eq!(
            drag_offsets(0, 0, 2, 1),
            (2.0 * MODEL_ROW_HEIGHT, MODEL_ROW_HEIGHT)
        );
        assert_eq!(drag_offsets(2, 0, 2, 1), (-MODEL_ROW_HEIGHT, 0.0));
    }

    #[test]
    fn custom_ids_are_validated_per_agent() {
        assert!(custom_id_error(HarnessId::ClaudeCode, "claude-mythos-5-1").is_none());
        assert!(custom_id_error(HarnessId::ClaudeCode, "").is_some());
        assert!(custom_id_error(HarnessId::Codex, "gpt 6").is_some());
        assert!(custom_id_error(HarnessId::Opencode, "claude-sonnet-5").is_some());
        assert!(custom_id_error(HarnessId::Opencode, "anthropic/claude-sonnet-5").is_none());
    }
}
