/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Wayland cursor shape resolution: converts a `CursorImageStatus` into a PNG image buffer
//! suitable for the Python cursor callback. Uses `xcursor` to load the system cursor theme
//! and render the appropriate frame at the current scale.

use image::{ImageBuffer, Rgba};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::HashMap;
use std::io::Cursor as IoCursor;
use std::io::Read;
use std::time::Duration;
use xcursor::{
    parser::{parse_xcursor, Image},
    CursorTheme,
};

/// One unit of cursor-callback work handed from the calloop thread to the `wl-cursor`
/// worker. The calloop resolves everything renderer/surface-affine (SHM copies, dmabuf
/// readbacks, hotspots); the worker does the PNG encode, the cache, and the GIL-bound Python
/// call so none of that latency lands on the input/render thread. One channel keeps cursor
/// updates ordered with callback (re)registration.
pub enum CursorJob {
    SetCallback(Py<PyAny>),
    /// Reload the worker's theme handle at a new pixel size; later `Named` jobs render at it.
    SetSize(i32),
    /// Cap the longest delivered cursor edge; larger images are downscaled like the X11
    /// XFixes monitor does (parity: `CaptureSettings.cursor_size_cap` was X11-only).
    SetSizeCap(i32),
    Named { name: &'static str },
    Hide,
    /// wl_shm cursor sprite: raw pool bytes plus the sub-image descriptor.
    Shm {
        hash: u64,
        width: i32,
        height: i32,
        stride: i32,
        offset: i32,
        opaque: bool,
        bytes: Vec<u8>,
        hot_x: i32,
        hot_y: i32,
    },
    /// dmabuf cursor sprite already read back to tightly-mapped RGBA on the calloop.
    Gles {
        hash: u64,
        width: i32,
        height: i32,
        bytes: Vec<u8>,
        hot_x: i32,
        hot_y: i32,
    },
}

/// Spawn the cursor delivery worker; returns its job channel. The worker owns its own
/// theme handle, the PNG cache, and the Python callback for the life of the process (like the
/// compositor thread itself); `PY_SHUTDOWN` gates every Python call.
pub fn spawn_cursor_worker(cursor_size: i32, size_cap: i32) -> std::sync::mpsc::Sender<CursorJob> {
    let (tx, rx) = std::sync::mpsc::channel::<CursorJob>();
    let _ = std::thread::Builder::new().name("wl-cursor".into()).spawn(move || {
        let mut helper = Cursor::load(cursor_size);
        let mut cap = size_cap;
        let mut cache: HashMap<u64, Vec<u8>> = HashMap::new();
        let mut callback: Option<Py<PyAny>> = None;
        while let Ok(job) = rx.recv() {
            let (msg_type, data, hot_x, hot_y): (&str, Vec<u8>, i32, i32) = match job {
                CursorJob::SetCallback(cb) => {
                    callback = Some(cb);
                    continue;
                }
                CursorJob::SetSize(size) => {
                    helper = Cursor::load(size);
                    continue;
                }
                CursorJob::SetSizeCap(c) => {
                    cap = c;
                    continue;
                }
                CursorJob::Named { name } => match helper.get_png_data(name) {
                    Some((png, x, y)) => {
                        let (png, x, y) = cap_cursor_png(png, x as i32, y as i32, cap);
                        ("png", png, x, y)
                    }
                    None => ("error", Vec::new(), 0, 0),
                },
                CursorJob::Hide => ("hide", Vec::new(), 0, 0),
                CursorJob::Shm {
                    hash,
                    width,
                    height,
                    stride,
                    offset,
                    opaque,
                    bytes,
                    hot_x,
                    hot_y,
                } => capped_job(
                    &mut cache,
                    hash,
                    cap,
                    hot_x,
                    hot_y,
                    || encode_shm_cursor(width, height, stride, offset, opaque, &bytes),
                ),
                CursorJob::Gles { hash, width, height, bytes, hot_x, hot_y } => capped_job(
                    &mut cache,
                    hash,
                    cap,
                    hot_x,
                    hot_y,
                    || encode_gles_cursor(width, height, &bytes),
                ),
            };
            // A sprite whose pixels could not be read yields empty data; suppressing it
            // preserves the consumer's last cursor instead of blanking it (only an
            // intentional hide passes with no payload).
            if data.is_empty() && msg_type != "hide" {
                continue;
            }
            if crate::PY_SHUTDOWN.load(std::sync::atomic::Ordering::Relaxed) {
                continue;
            }
            if let Some(ref cb) = callback {
                Python::attach(|py| {
                    let py_bytes = PyBytes::new(py, &data);
                    let _ = cb.call1(py, (msg_type, py_bytes, hot_x, hot_y));
                });
            }
        }
    });
    tx
}

/// Content-hash PNG cache lookup with bounded arbitrary eviction; an evicted sprite simply
/// re-encodes on its next appearance.
fn cached_png(
    cache: &mut HashMap<u64, Vec<u8>>,
    hash: u64,
    encode: impl FnOnce() -> Vec<u8>,
) -> Vec<u8> {
    if let Some(png) = cache.get(&hash) {
        return png.clone();
    }
    let png = encode();
    if !png.is_empty() {
        cache.insert(hash, png.clone());
        if cache.len() > 100 {
            if let Some(&evict) = cache.keys().next() {
                cache.remove(&evict);
            }
        }
    }
    png
}

/// Downscale a cursor PNG so its longest edge is at most `cap` (<= 0 = uncapped), scaling
/// the hotspot with it. Mirrors the X11 XFixes monitor's `cursor_size_cap` handling so a
/// Wayland session ships cursor payloads no larger than an X11 one.
fn cap_cursor_png(png: Vec<u8>, hot_x: i32, hot_y: i32, cap: i32) -> (Vec<u8>, i32, i32) {
    if cap <= 0 || png.is_empty() {
        return (png, hot_x, hot_y);
    }
    let img = match image::load_from_memory(&png) {
        Ok(i) => i.to_rgba8(),
        Err(_) => return (png, hot_x, hot_y),
    };
    let (w, h) = (img.width(), img.height());
    let longest = w.max(h);
    if longest <= cap as u32 || longest == 0 {
        return (png, hot_x, hot_y);
    }
    let scale = cap as f64 / longest as f64;
    let nw = ((w as f64 * scale).round() as u32).max(1);
    let nh = ((h as f64 * scale).round() as u32).max(1);
    let resized = image::imageops::resize(
        &img,
        nw,
        nh,
        image::imageops::FilterType::Lanczos3,
    );
    let mut out = Vec::new();
    if resized
        .write_to(&mut IoCursor::new(&mut out), image::ImageFormat::Png)
        .is_err()
    {
        return (png, hot_x, hot_y);
    }
    (
        out,
        (hot_x as f64 * scale).round() as i32,
        (hot_y as f64 * scale).round() as i32,
    )
}

/// Cache wrapper for sprite jobs: encode (or reuse) the full-size PNG, then apply the
/// size cap. `cap_cursor_png` is idempotent for already-capped images, so cache hits
/// only pay a cheap dimension check.
fn capped_job(
    cache: &mut HashMap<u64, Vec<u8>>,
    hash: u64,
    cap: i32,
    hot_x: i32,
    hot_y: i32,
    encode: impl FnOnce() -> Vec<u8>,
) -> (&'static str, Vec<u8>, i32, i32) {
    let png = cached_png(cache, hash, encode);
    let (png, sx, sy) = cap_cursor_png(png, hot_x, hot_y, cap);
    ("png", png, sx, sy)
}

/// Convert a wl_shm BGRA/XRGB sprite sub-image to a straight-alpha PNG. Stride/offset are
/// clamped non-negative with checked arithmetic so a garbage descriptor skips pixels instead of
/// panicking; sprites larger than 128x128 are ignored (never a hardware cursor).
fn encode_shm_cursor(
    width: i32,
    height: i32,
    stride: i32,
    offset: i32,
    opaque: bool,
    raw_bytes: &[u8],
) -> Vec<u8> {
    if width <= 0 || height <= 0 || width > 128 || height > 128 || raw_bytes.is_empty() {
        return Vec::new();
    }
    let mut img_buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(width as u32, height as u32);
    let stride_usize = stride.max(0) as usize;
    let base_offset = offset.max(0) as usize;
    for y in 0..(height as u32) {
        for x in 0..(width as u32) {
            let offset = (y as usize)
                .checked_mul(stride_usize)
                .and_then(|row| base_offset.checked_add(row))
                .and_then(|o| o.checked_add((x as usize) * 4));
            let offset = match offset {
                Some(o) => o,
                None => continue,
            };
            if offset.checked_add(4).is_some_and(|end| end <= raw_bytes.len()) {
                let alpha = if opaque { 255 } else { raw_bytes[offset + 3] };
                img_buf.put_pixel(
                    x,
                    y,
                    Rgba([raw_bytes[offset + 2], raw_bytes[offset + 1], raw_bytes[offset], alpha]),
                );
            }
        }
    }
    crate::unpremultiply_rgba(&mut img_buf);
    let mut bytes = Vec::new();
    if img_buf.write_to(&mut IoCursor::new(&mut bytes), image::ImageFormat::Png).is_ok() {
        bytes
    } else {
        Vec::new()
    }
}

/// Convert a dmabuf sprite's RGBA readback (stride recovered from the mapping length) to a
/// straight-alpha PNG.
fn encode_gles_cursor(width: i32, height: i32, raw_bytes: &[u8]) -> Vec<u8> {
    if width <= 0 || height <= 0 || width > 128 || height > 128 || raw_bytes.is_empty() {
        return Vec::new();
    }
    let stride = super::frontend::rgba_readback_stride(
        raw_bytes.len(),
        height as usize,
        width as usize,
    );
    let mut img_buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(width as u32, height as u32);
    for y in 0..(height as u32) {
        for x in 0..(width as u32) {
            let offset = (y as usize * stride) + (x as usize * 4);
            if offset + 4 <= raw_bytes.len() {
                img_buf.put_pixel(
                    x,
                    y,
                    Rgba([
                        raw_bytes[offset],
                        raw_bytes[offset + 1],
                        raw_bytes[offset + 2],
                        raw_bytes[offset + 3],
                    ]),
                );
            }
        }
    }
    crate::unpremultiply_rgba(&mut img_buf);
    let mut bytes = Vec::new();
    if img_buf.write_to(&mut IoCursor::new(&mut bytes), image::ImageFormat::Png).is_ok() {
        bytes
    } else {
        Vec::new()
    }
}

/// The loaded XCursor theme, held for the whole capture so cursor lookups stay cheap.
///
/// The default cursor's frames are parsed once and kept here because they are consulted on nearly
/// every frame; the theme handle is retained alongside them so the rarer named cursors (`hand1`,
/// `text`, …) can still be resolved on demand. Everything is sized to the one resolved pixel size.
pub struct Cursor {
    icons: Vec<Image>,
    theme: CursorTheme,
    size: u32,
}

impl Cursor {
    /// Load the theme named by `XCURSOR_THEME` (default `"default"`) at the requested size.
    ///
    /// `size_override` comes from `CaptureSettings.cursor_size` (selkies `--cursor-size` /
    /// `XCURSOR_SIZE`); a value `<= 0` falls back to 24. When the theme's default icon cannot be
    /// loaded, a 16×16 solid-red placeholder stands in so the caller always has a valid image.
    pub fn load(size_override: i32) -> Cursor {
        let name = std::env::var("XCURSOR_THEME").unwrap_or_else(|_| "default".into());
        let size: u32 = if size_override > 0 { size_override as u32 } else { 24 };

        let theme = CursorTheme::load(&name);
        let icons = load_icon(&theme, "default").unwrap_or_else(|_| {
            let size = 16;
            let mut pixels = Vec::with_capacity((size * size * 4) as usize);
            for _ in 0..(size * size) {
                pixels.extend_from_slice(&[255, 0, 0, 255]);
            }

            vec![Image {
                size,
                width: size,
                height: size,
                xhot: 0,
                yhot: 0,
                delay: 1,
                pixels_rgba: pixels,
                pixels_argb: vec![],
            }]
        });

        Cursor { icons, theme, size }
    }

    /// The default cursor's animation frame for `time`, at the theme size times `scale`.
    pub fn get_image(&self, scale: u32, time: Duration) -> Image {
        let size = self.size * scale;
        frame(time.as_millis() as u32, size, &self.icons)
    }

    /// A named cursor's animation frame for `time`, or `None` when the theme lacks it.
    pub fn get_image_by_name(&self, name: &str, scale: u32, time: Duration) -> Option<Image> {
        let icons = load_icon(&self.theme, name).ok()?;
        let size = self.size * scale;
        Some(frame(time.as_millis() as u32, size, &icons))
    }

    /// A named cursor icon as PNG bytes plus its hotspot (x, y), for web clients.
    /// Xcursor stores premultiplied color; the PNG carries straight alpha. The
    /// compositing paths (`get_image*`) keep the premultiplied pixels blending needs.
    pub fn get_png_data(&self, name: &str) -> Option<(Vec<u8>, u32, u32)> {
        let icons = load_icon(&self.theme, name).ok()?;
        let image_data = nearest_images(self.size, &icons).next()?;

        let mut img_buf: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_raw(
            image_data.width,
            image_data.height,
            image_data.pixels_rgba.clone(),
        )?;
        crate::unpremultiply_rgba(&mut img_buf);

        let mut bytes: Vec<u8> = Vec::new();
        let mut cursor = IoCursor::new(&mut bytes);
        img_buf
            .write_to(&mut cursor, image::ImageFormat::Png)
            .ok()?;

        Some((bytes, image_data.xhot, image_data.yhot))
    }
}

/// All frames of the theme variant whose pixel size is closest to `size`. XCursor files
/// bundle the same cursor at several sizes, and choosing the nearest one avoids scaling a
/// mismatched bitmap into a blurry or aliased cursor.
fn nearest_images(size: u32, images: &[Image]) -> impl Iterator<Item = &Image> {
    let nearest_image = images
        .iter()
        .min_by_key(|image| (size as i32 - image.size as i32).abs())
        .unwrap();
    images
        .iter()
        .filter(move |image| image.width == nearest_image.width && image.height == nearest_image.height)
}

/// Pick which animation frame to show for the elapsed time, so animated cursors (a spinner,
/// a progress ring) actually advance instead of freezing on frame zero; it maps the time onto the
/// frames by cycling their cumulative delays.
fn frame(mut millis: u32, size: u32, images: &[Image]) -> Image {
    let total = nearest_images(size, images).fold(0, |acc, image| acc + image.delay);
    if total == 0 {
        return nearest_images(size, images).next().unwrap().clone();
    }
    millis %= total;
    for img in nearest_images(size, images) {
        if millis < img.delay {
            return img.clone();
        }
        millis -= img.delay;
    }
    unreachable!()
}

/// Parse the named icon from the theme's cursor file into its frames.
///
/// Empty parses are rejected so `nearest_images`'s `min_by_key().unwrap()` cannot panic on a
/// cursor file that has a valid header but zero images.
fn load_icon(theme: &CursorTheme, name: &str) -> Result<Vec<Image>, String> {
    let icon_path = theme.load_icon(name).ok_or("Icon not found")?;
    let mut cursor_file = std::fs::File::open(icon_path).map_err(|e| e.to_string())?;
    let mut cursor_data = Vec::new();
    cursor_file
        .read_to_end(&mut cursor_data)
        .map_err(|e| e.to_string())?;
    let imgs = parse_xcursor(&cursor_data).ok_or("Failed to parse".to_string())?;
    if imgs.is_empty() {
        return Err("Cursor file has no images".to_string());
    }
    Ok(imgs)
}
