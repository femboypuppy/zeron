//! Zeron's own window frost on Windows: a heavily blurred copy of the desktop
//! wallpaper painted behind the whole window, aligned with the window's place
//! on its display — the Mica look, drawn by Zeron instead of DWM.
//!
//! DWM backdrops (legacy accent blur, Acrylic, Mica) quietly fall back to a
//! flat tint on some systems (virtual displays such as Parsec's, GPU or power
//! policies), leaving "frosted" windows looking solid. Painting the frost
//! ourselves makes the glass show the desktop's colours everywhere; the
//! theme's frost tint ([`crate::theme::Theme::glass`]) still sits on top, so
//! Settings → Appearance → Frost strength decides how much colour comes
//! through.
//!
//! The wallpaper is read from Windows' own transcoded copy of the current
//! wallpaper, downscaled and blurred once on a background thread, and cached
//! as a small PNG keyed by the source's path, size, and modification time.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use gpui::{
    AnyElement, App, Global, IntoElement, ObjectFit, ParentElement as _, Styled as _,
    StyledImage as _, Window, div, img,
};

/// Longest edge of the cached blur; gpui scales it up to the display, which
/// multiplies the blur radius along with it.
const BLUR_EDGE: u32 = 640;
const BLUR_SIGMA: f32 = 14.0;

/// The blurred-wallpaper cache, installed as a global at boot (Windows only).
pub struct WallpaperFrost {
    cache_dir: PathBuf,
    /// Identity of the wallpaper the current blur came from.
    source: Option<SourceKey>,
    blurred: Option<PathBuf>,
    rendering: bool,
}

impl Global for WallpaperFrost {}

#[derive(Clone, PartialEq, Eq)]
struct SourceKey {
    path: PathBuf,
    len: u64,
    modified: Option<SystemTime>,
}

impl SourceKey {
    fn read(path: PathBuf) -> Option<Self> {
        let meta = std::fs::metadata(&path).ok()?;
        meta.is_file().then(|| Self {
            len: meta.len(),
            modified: meta.modified().ok(),
            path,
        })
    }

    /// Cache file name: a hash of the identity, so a changed wallpaper gets a
    /// new file (and gpui a fresh texture) instead of a stale cached image.
    fn cache_name(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(self.path.to_string_lossy().as_bytes());
        hasher.update(self.len.to_le_bytes());
        if let Some(modified) = self
            .modified
            .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        {
            hasher.update(modified.as_nanos().to_le_bytes());
        }
        let digest = hasher.finalize();
        let hex: String = digest[..8].iter().map(|b| format!("{b:02x}")).collect();
        format!("wallpaper-{hex}.png")
    }
}

/// Install the cache and render the current wallpaper's blur. A no-op off
/// Windows, where the platform's own window blur is used.
pub fn init(data_dir: &Path, cx: &mut App) {
    if !cfg!(target_os = "windows") {
        return;
    }
    cx.set_global(WallpaperFrost {
        cache_dir: data_dir.join("frost"),
        source: None,
        blurred: None,
        rendering: false,
    });
    refresh(cx);
    // Follow wallpaper changes (and slideshows) with a cheap periodic stat.
    cx.spawn(async move |cx| {
        loop {
            cx.background_executor()
                .timer(std::time::Duration::from_secs(30))
                .await;
            cx.update(refresh);
        }
    })
    .detach();
}

/// Re-check the wallpaper (cheap: one `stat`) and re-blur it if it changed.
pub fn refresh(cx: &mut App) {
    let Some(state) = cx.try_global::<WallpaperFrost>() else {
        return;
    };
    if state.rendering {
        return;
    }
    let source = wallpaper_path().and_then(SourceKey::read);
    if source == state.source {
        return;
    }
    let cache_dir = state.cache_dir.clone();
    let Some(source) = source else {
        let state = cx.global_mut::<WallpaperFrost>();
        state.source = None;
        state.blurred = None;
        cx.refresh_windows();
        return;
    };
    cx.global_mut::<WallpaperFrost>().rendering = true;
    let out = cache_dir.join(source.cache_name());
    let work = {
        let (source, out) = (source.path.clone(), out.clone());
        cx.background_executor().spawn(async move {
            if out.is_file() {
                return Ok(());
            }
            render_blur(&source, &out, &cache_dir)
        })
    };
    cx.spawn(async move |cx| {
        let result = work.await;
        cx.update(|cx| {
            let state = cx.global_mut::<WallpaperFrost>();
            state.rendering = false;
            state.source = Some(source);
            match result {
                Ok(()) => state.blurred = Some(out),
                Err(err) => {
                    tracing::warn!(error = %err, "wallpaper frost: could not blur the wallpaper");
                    state.blurred = None;
                }
            }
            cx.refresh_windows();
        });
    })
    .detach();
}

/// The cached blur of the current wallpaper, once rendered.
pub fn blurred_wallpaper(cx: &App) -> Option<PathBuf> {
    cx.try_global::<WallpaperFrost>()?.blurred.clone()
}

/// The frost layer for `window`: the blurred wallpaper sized to the window's
/// display and offset so it lines up with the real desktop behind the window.
/// Paint it beneath the frost tint; `None` until a blur exists.
pub fn backdrop(window: &Window, cx: &App) -> Option<AnyElement> {
    let path = blurred_wallpaper(cx)?;
    let screen = window.display(cx)?.bounds();
    let bounds = window.bounds();
    Some(
        div()
            .absolute()
            .left(screen.origin.x - bounds.origin.x)
            .top(screen.origin.y - bounds.origin.y)
            .w(screen.size.width)
            .h(screen.size.height)
            .child(img(path).size_full().object_fit(ObjectFit::Cover))
            .into_any_element(),
    )
}

/// Windows keeps a transcoded copy of the current wallpaper (single image or
/// slideshow frame); the registry value is the fallback for setups without it.
fn wallpaper_path() -> Option<PathBuf> {
    let transcoded = std::env::var_os("APPDATA").map(|appdata| {
        PathBuf::from(appdata)
            .join("Microsoft")
            .join("Windows")
            .join("Themes")
            .join("TranscodedWallpaper")
    });
    if let Some(path) = transcoded.filter(|path| path.is_file()) {
        return Some(path);
    }
    registry_wallpaper()
}

#[cfg(target_os = "windows")]
fn registry_wallpaper() -> Option<PathBuf> {
    use std::os::windows::process::CommandExt as _;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let output = std::process::Command::new("reg")
        .args(["query", r"HKCU\Control Panel\Desktop", "/v", "WallPaper"])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .ok()?;
    parse_reg_wallpaper(&String::from_utf8_lossy(&output.stdout))
}

#[cfg(not(target_os = "windows"))]
fn registry_wallpaper() -> Option<PathBuf> {
    None
}

/// The path from `reg query … /v WallPaper` output
/// (`    WallPaper    REG_SZ    C:\…\image.jpg`).
fn parse_reg_wallpaper(output: &str) -> Option<PathBuf> {
    output.lines().find_map(|line| {
        let (_, value) = line.split_once("REG_SZ")?;
        let value = value.trim();
        (!value.is_empty())
            .then(|| PathBuf::from(value))
            .filter(|path| path.is_file())
    })
}

/// Decode `source` (format sniffed: the transcoded copy has no extension),
/// shrink it, blur it, and write the result to `out` as PNG.
fn render_blur(source: &Path, out: &Path, cache_dir: &Path) -> anyhow::Result<()> {
    let image = image::ImageReader::open(source)?
        .with_guessed_format()?
        .decode()?;
    let small = image.resize(BLUR_EDGE, BLUR_EDGE, image::imageops::FilterType::Triangle);
    let blurred = image::imageops::blur(&small.to_rgba8(), BLUR_SIGMA);
    std::fs::create_dir_all(cache_dir)?;
    // Old blurs are only a few KB each, but a changing slideshow would
    // accumulate them: keep just the one being written.
    if let Ok(entries) = std::fs::read_dir(cache_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("wallpaper-") && name.ends_with(".png") {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
    let partial = out.with_extension("png.partial");
    blurred.save_with_format(&partial, image::ImageFormat::Png)?;
    std::fs::rename(&partial, out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reg_output_parses_to_an_existing_path() {
        let dir = tempfile::tempdir().unwrap();
        let image = dir.path().join("wall paper.jpg");
        std::fs::write(&image, b"x").unwrap();
        let output = format!(
            "\r\nHKEY_CURRENT_USER\\Control Panel\\Desktop\r\n    WallPaper    REG_SZ    {}\r\n\r\n",
            image.display()
        );
        assert_eq!(parse_reg_wallpaper(&output), Some(image));
        assert_eq!(parse_reg_wallpaper("    WallPaper    REG_SZ    \r\n"), None);
        assert_eq!(
            parse_reg_wallpaper("    WallPaper    REG_SZ    C:\\missing\\x.jpg"),
            None
        );
    }

    #[test]
    fn a_wallpaper_blurs_into_a_small_cached_png() {
        let dir = tempfile::tempdir().unwrap();
        // Four colour bands, like a busy wallpaper; no extension, like
        // Windows' transcoded copy.
        let mut source = image::RgbImage::new(1600, 900);
        for (x, _, pixel) in source.enumerate_pixels_mut() {
            *pixel = match x / 400 {
                0 => image::Rgb([255, 40, 120]),
                1 => image::Rgb([255, 200, 0]),
                2 => image::Rgb([0, 200, 120]),
                _ => image::Rgb([40, 120, 255]),
            };
        }
        let source_path = dir.path().join("TranscodedWallpaper");
        source
            .save_with_format(&source_path, image::ImageFormat::Jpeg)
            .unwrap();
        let cache = dir.path().join("frost");
        let key = SourceKey::read(source_path.clone()).unwrap();
        let out = cache.join(key.cache_name());
        render_blur(&source_path, &out, &cache).unwrap();

        let blurred = image::open(&out).unwrap().to_rgba8();
        assert_eq!(blurred.width(), BLUR_EDGE, "long edge is shrunk");
        assert_eq!(blurred.height(), 360, "aspect ratio is kept");
        // Band colours survive, softened: the pink band stays pinkish and the
        // blue one bluish, while the hard edge between bands is gone.
        let pink = blurred.get_pixel(40, 180);
        let blue = blurred.get_pixel(600, 180);
        assert!(pink[0] > pink[2] && blue[2] > blue[0]);
        let edge_left = blurred.get_pixel(158, 180);
        let edge_right = blurred.get_pixel(162, 180);
        assert!(
            (edge_left[1] as i32 - edge_right[1] as i32).abs() < 40,
            "the band edge is blurred"
        );
        assert!(!out.with_extension("png.partial").exists());

        // A changed wallpaper gets a different cache name.
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(&source_path, b"changed").unwrap();
        let changed = SourceKey::read(source_path).unwrap();
        assert_ne!(changed.cache_name(), key.cache_name());
    }
}
