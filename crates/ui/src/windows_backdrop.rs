//! Windows: DWM's live Acrylic backdrop (`DWMSBT_TRANSIENTWINDOW`) behind the
//! window — real frosted glass over the apps behind it — for the "Live blur"
//! frost backdrop.
//!
//! gpui's `Blurred` background is the legacy accent-policy acrylic
//! (`SetWindowCompositionAttribute`), which recent Windows 11 builds render as
//! black behind composition-swapchain windows, and which also keeps the
//! documented system backdrop from showing. For live blur Zeron clears that
//! accent, extends the frame over the whole client area, and asks DWM for the
//! system Acrylic backdrop instead.
//!
//! Windows only draws it with Settings → Personalization → Colors →
//! Transparency effects on (and Energy saver off). Tools that rewrite other
//! apps' backdrops — e.g. Windhawk's "Translucent Windows" mod, whose global
//! Mica setting replaces this request with a wallpaper-only backdrop — need a
//! rule leaving `zeron.exe` on its default.

use gpui::Window;

/// Apply (`live`) or withdraw the Acrylic backdrop on `window`. Call after
/// every `Window::set_background_appearance`, which re-installs gpui's accent.
#[cfg(target_os = "windows")]
pub fn apply(window: &Window, live: bool, dark: bool) {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    // Nothing to apply or undo: skip the platform handle entirely (gpui's
    // test windows have none, and asking for it panics).
    if !live && !imp::applied() {
        return;
    }
    // Fully qualified: gpui's inherent `Window::window_handle` returns its own
    // handle type, not the platform one.
    let Ok(handle) = HasWindowHandle::window_handle(window) else {
        return;
    };
    let RawWindowHandle::Win32(win32) = handle.as_raw() else {
        return;
    };
    let hwnd = win32.hwnd.get() as windows_sys::Win32::Foundation::HWND;
    // SAFETY: `hwnd` is this live gpui window's handle, and every call below
    // only sets DWM/composition attributes on it.
    unsafe { imp::apply(hwnd, live, dark) }
}

#[cfg(not(target_os = "windows"))]
pub fn apply(_window: &Window, _live: bool, _dark: bool) {}

#[cfg(target_os = "windows")]
mod imp {
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicBool, Ordering};

    use windows_sys::Win32::Foundation::HWND;
    use windows_sys::Win32::Graphics::Dwm::{
        DWMSBT_NONE, DWMSBT_TRANSIENTWINDOW, DWMWA_SYSTEMBACKDROP_TYPE,
        DWMWA_USE_IMMERSIVE_DARK_MODE, DwmExtendFrameIntoClientArea, DwmSetWindowAttribute,
    };
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    use windows_sys::Win32::UI::Controls::MARGINS;

    /// Whether live blur was applied, so switching it off only undoes our own
    /// change and never touches a window that never had it.
    static APPLIED: AtomicBool = AtomicBool::new(false);

    #[repr(C)]
    struct AccentPolicy {
        state: u32,
        flags: u32,
        gradient_color: u32,
        animation_id: u32,
    }

    #[repr(C)]
    struct CompositionAttributeData {
        attribute: u32,
        data: *mut c_void,
        size: usize,
    }

    const WCA_ACCENT_POLICY: u32 = 0x13;
    const ACCENT_DISABLED: u32 = 0;

    pub(super) fn applied() -> bool {
        APPLIED.load(Ordering::Relaxed)
    }

    pub(super) unsafe fn apply(hwnd: HWND, live: bool, dark: bool) {
        if live {
            unsafe {
                set_accent_disabled(hwnd);
                // The whole client area is "frame", so DWM paints the backdrop
                // wherever the swapchain is transparent.
                let margins = MARGINS {
                    cxLeftWidth: -1,
                    cxRightWidth: -1,
                    cyTopHeight: -1,
                    cyBottomHeight: -1,
                };
                DwmExtendFrameIntoClientArea(hwnd, &margins);
                set_attribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, dark as i32);
                set_attribute(hwnd, DWMWA_SYSTEMBACKDROP_TYPE, DWMSBT_TRANSIENTWINDOW);
            }
            APPLIED.store(true, Ordering::Relaxed);
        } else if APPLIED.swap(false, Ordering::Relaxed) {
            unsafe {
                set_attribute(hwnd, DWMWA_SYSTEMBACKDROP_TYPE, DWMSBT_NONE);
                let margins = MARGINS {
                    cxLeftWidth: 0,
                    cxRightWidth: 0,
                    cyTopHeight: 0,
                    cyBottomHeight: 0,
                };
                DwmExtendFrameIntoClientArea(hwnd, &margins);
            }
        }
    }

    unsafe fn set_attribute(hwnd: HWND, attribute: i32, value: i32) {
        unsafe {
            DwmSetWindowAttribute(
                hwnd,
                attribute as u32,
                &value as *const i32 as *const c_void,
                std::mem::size_of::<i32>() as u32,
            );
        }
    }

    /// Clear gpui's legacy accent blur; `SetWindowCompositionAttribute` is
    /// undocumented, so it's looked up at runtime like gpui does.
    unsafe fn set_accent_disabled(hwnd: HWND) {
        type SetWindowCompositionAttribute =
            unsafe extern "system" fn(HWND, *mut CompositionAttributeData) -> i32;
        unsafe {
            let user32 = GetModuleHandleA(c"user32.dll".as_ptr().cast());
            if user32.is_null() {
                return;
            }
            let Some(proc) =
                GetProcAddress(user32, c"SetWindowCompositionAttribute".as_ptr().cast())
            else {
                return;
            };
            let set: SetWindowCompositionAttribute = std::mem::transmute(proc);
            let mut policy = AccentPolicy {
                state: ACCENT_DISABLED,
                flags: 0,
                gradient_color: 0,
                animation_id: 0,
            };
            let mut data = CompositionAttributeData {
                attribute: WCA_ACCENT_POLICY,
                data: &mut policy as *mut AccentPolicy as *mut c_void,
                size: std::mem::size_of::<AccentPolicy>(),
            };
            set(hwnd, &mut data);
        }
    }
}
