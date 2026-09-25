//! Handing a conversation to a different agent.
//!
//! A chat can switch agents mid-conversation (Codex → Claude Code, …), and one
//! agent can't resume another's native session. The first run under the new
//! agent therefore starts a fresh session, and this module writes the
//! conversation so far ahead of the user's prompt, so the new agent continues
//! with the same context. Only the agent sees the preamble: the transcript
//! keeps the user's message as typed. Once the new agent's session exists, it
//! resumes normally (it already holds the handed-off context).

use zeron_doc::{MessagePart, MessageRole, SessionMessageEntry};
use zeron_proto::HarnessId;

/// Most recent conversation text carried over; older turns are dropped first.
const MAX_HANDOFF_CHARS: usize = 60_000;
/// Per-message cap, so one huge reply can't crowd out the rest.
const MAX_MESSAGE_CHARS: usize = 6_000;
/// Tool calls listed per assistant message.
const MAX_TOOLS_PER_MESSAGE: usize = 12;

pub(crate) fn agent_label(harness: HarnessId) -> &'static str {
    match harness {
        HarnessId::ClaudeCode => "Claude Code",
        HarnessId::Codex => "Codex",
        HarnessId::Cursor => "Cursor",
        HarnessId::Devin => "Devin",
        HarnessId::Grok => "Grok",
        HarnessId::Hermes => "Hermes",
        HarnessId::Pi => "Pi",
        HarnessId::Opencode => "OpenCode",
        HarnessId::Antigravity => "Antigravity",
        HarnessId::Mock => "Mock",
    }
}

/// `prompt` prefixed with the conversation in `entries` (minus the entry
/// `current_id`, the message being sent now), or `None` when there's nothing
/// to hand over or the prompt is a native command, which must stay first.
pub(crate) fn handoff_prompt(
    entries: &[SessionMessageEntry],
    current_id: &str,
    from: Option<HarnessId>,
    to: HarnessId,
    prompt: &str,
) -> Option<String> {
    let native = zeron_proto::invocation::harness_prompt(prompt, to);
    if zeron_proto::invocation::leading_command(&native).is_some() {
        return None;
    }
    let assistant = from.map_or("Assistant", agent_label);
    let mut messages: Vec<String> = entries
        .iter()
        .filter(|entry| entry.id != current_id)
        .filter_map(|entry| render_entry(entry, assistant))
        .collect();
    if messages.is_empty() {
        return None;
    }
    // Keep the most recent messages within the budget.
    let mut total = 0;
    let mut keep = messages.len();
    for (index, message) in messages.iter().enumerate().rev() {
        if total + message.len() > MAX_HANDOFF_CHARS && keep < messages.len() {
            break;
        }
        total += message.len();
        keep = index;
    }
    let omitted = keep;
    let messages = messages.split_off(keep);
    let source = from.map_or_else(
        || "another agent".to_string(),
        |from| agent_label(from).to_string(),
    );
    let mut out = format!(
        "<previous-conversation>\nUntil now this conversation was with {source}; you ({}) are \
         taking it over. Below is the conversation so far, oldest first, for context. Continue \
         from it; the user's new message follows after this block.\n",
        agent_label(to)
    );
    if omitted > 0 {
        out.push_str(&format!("\n[{omitted} earlier messages omitted]\n"));
    }
    for message in messages {
        out.push('\n');
        out.push_str(&message);
        out.push('\n');
    }
    out.push_str("</previous-conversation>\n\n");
    out.push_str(prompt);
    Some(out)
}

fn render_entry(entry: &SessionMessageEntry, assistant: &str) -> Option<String> {
    let mut text = Vec::new();
    let mut tools = Vec::new();
    for part in &entry.parts {
        match part {
            MessagePart::Text { text: body, .. } if !body.trim().is_empty() => {
                text.push(body.trim().to_string())
            }
            MessagePart::Tool { call, is_error, .. } => {
                let (label, detail) = zeron_proto::view::tool_chip_content(call);
                let failed = if *is_error { " (failed)" } else { "" };
                tools.push(if detail.is_empty() {
                    format!("- {label}{failed}")
                } else {
                    format!("- {label}: {}{failed}", clip(&detail, 200))
                });
            }
            _ => {}
        }
    }
    if text.is_empty() && tools.is_empty() {
        return None;
    }
    let speaker = match entry.role {
        MessageRole::User => "User",
        MessageRole::Assistant => assistant,
        MessageRole::System => "System",
    };
    let mut out = format!(
        "{speaker}:\n{}",
        clip(&text.join("\n\n"), MAX_MESSAGE_CHARS)
    );
    if !tools.is_empty() {
        let more = tools.len().saturating_sub(MAX_TOOLS_PER_MESSAGE);
        tools.truncate(MAX_TOOLS_PER_MESSAGE);
        if more > 0 {
            tools.push(format!("- … {more} more tool calls"));
        }
        out.push_str(&format!("\n[tools used]\n{}", tools.join("\n")));
    }
    Some(out)
}

fn clip(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((cut, _)) => format!("{}…", &text[..cut]),
        None => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zeron_doc::MessageStatus;
    use zeron_proto::ToolCall;

    fn entry(id: &str, role: MessageRole, parts: Vec<MessagePart>) -> SessionMessageEntry {
        SessionMessageEntry {
            id: id.into(),
            role,
            parts,
            created_at: 0,
            device_id: "d".into(),
            status: Some(MessageStatus::Complete),
            continuation_of: None,
            duration_ms: None,
        }
    }

    fn text(body: &str) -> MessagePart {
        MessagePart::Text {
            id: "t0".into(),
            text: body.into(),
        }
    }

    fn conversation() -> Vec<SessionMessageEntry> {
        vec![
            entry("u1", MessageRole::User, vec![text("Fix the login bug")]),
            entry(
                "a1",
                MessageRole::Assistant,
                vec![
                    MessagePart::Reasoning {
                        id: "r0".into(),
                        text: "private thoughts".into(),
                    },
                    MessagePart::Tool {
                        id: "c1".into(),
                        call: ToolCall::Exec {
                            command: "cargo test".into(),
                        },
                        is_error: true,
                        resolved: true,
                        output: None,
                        diff: None,
                        output_ref: None,
                        output_bytes: None,
                        diff_ref: None,
                        diff_stats: None,
                        subagent_ref: None,
                        subagent_status: None,
                        subagent_tail: None,
                    },
                    text("The token check was inverted; fixed."),
                ],
            ),
            entry("u2", MessageRole::User, vec![text("now add a test")]),
        ]
    }

    #[test]
    fn the_new_agent_gets_the_conversation_before_the_prompt() {
        let prompt = handoff_prompt(
            &conversation(),
            "u2",
            Some(HarnessId::Codex),
            HarnessId::ClaudeCode,
            "now add a test",
        )
        .expect("handoff");
        assert!(prompt.starts_with("<previous-conversation>"));
        assert!(prompt.contains("was with Codex; you (Claude Code) are taking it over"));
        assert!(prompt.contains("User:\nFix the login bug"));
        assert!(prompt.contains("Codex:\nThe token check was inverted; fixed."));
        assert!(prompt.contains("- Run: cargo test (failed)"));
        assert!(
            !prompt.contains("private thoughts"),
            "reasoning stays private"
        );
        assert!(
            !prompt.contains("User:\nnow add a test"),
            "the message being sent isn't repeated in the history"
        );
        assert!(prompt.ends_with("</previous-conversation>\n\nnow add a test"));
    }

    #[test]
    fn nothing_to_hand_over_or_a_native_command_sends_the_prompt_alone() {
        let only_current = vec![entry("u1", MessageRole::User, vec![text("hi")])];
        assert_eq!(
            handoff_prompt(&only_current, "u1", None, HarnessId::Codex, "hi"),
            None
        );
        assert_eq!(
            handoff_prompt(
                &conversation(),
                "u2",
                Some(HarnessId::ClaudeCode),
                HarnessId::Codex,
                "/compact"
            ),
            None,
            "commands must stay at the start of the prompt"
        );
    }

    #[test]
    fn long_histories_keep_the_latest_messages() {
        let mut entries = Vec::new();
        for i in 0..40 {
            entries.push(entry(
                &format!("u{i}"),
                MessageRole::User,
                vec![text(&format!("message {i} {}", "x".repeat(4_000)))],
            ));
        }
        let prompt = handoff_prompt(&entries, "none", None, HarnessId::ClaudeCode, "go").unwrap();
        assert!(prompt.len() < MAX_HANDOFF_CHARS + 2_000);
        assert!(prompt.contains("message 39 "), "the latest message is kept");
        assert!(
            !prompt.contains("message 0 "),
            "the oldest are dropped first"
        );
        assert!(prompt.contains("earlier messages omitted]"));
        assert!(prompt.contains("was with another agent"));
    }
}
