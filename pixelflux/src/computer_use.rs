/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! HTTP server implementing the [Anthropic Computer Use](https://github.com/anthropics/claude-quickstarts/tree/main/computer-use-demo) specification.
//!
//! Enabled by setting the `PIXELFLUX_CU` environment variable to the listen port. The server
//! handles `POST /computer-use` requests for screenshots, mouse/keyboard injection, scrolling,
//! and cursor position queries. Actions run against a [`CuBackend`], resolved per request: the
//! Wayland compositor owned by this process when one is registered, otherwise the X server named
//! by `DISPLAY` (XTEST injection on a private connection, no active capture required).
//!
//! The same server also exposes the built-in MP4 recorder at `/record_start`, `/record_stop`
//! and `/record_status`, so a headless script can drive a session and record it over plain HTTP.

use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::Duration;
use std::io::Cursor;
use std::io::Read;

use smithay::input::keyboard::xkb;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use image::{ImageBuffer, Rgba, ImageFormat};
use serde::Deserialize;
use tiny_http;

use crate::wayland::keymap::keysym_for_char;
use crate::ThreadCommand;

fn clamp<T: PartialOrd>(v: T, lo: T, hi: T) -> T {
    if v < lo { lo } else if v > hi { hi } else { v }
}

/// Turn a raw framebuffer into a PNG the Computer Use agent can actually look at.
///
/// The `screenshot` and `zoom` crop paths have to hand the agent an image, and the API carries it
/// as base64 PNG, so this encodes a flat RGBA buffer (with its dimensions) through the `image`
/// crate and returns the PNG bytes for that response payload.
pub fn encode_png_rgba(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    let img = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, data.to_vec())
        .ok_or("Failed to create image buffer")?;
    let mut png = Vec::new();
    img.write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
        .map_err(|e| format!("PNG encode error: {}", e))?;
    Ok(png)
}

fn scancode_for_keyname(name: &str) -> Option<u32> {
    Some(match name.to_lowercase().as_str() {
        "return" | "enter" => 36,
        "tab" => 23,
        "escape" | "esc" => 9,
        "backspace" => 22,
        "delete" => 119,
        "insert" => 118,
        "home" => 110,
        "end" => 115,
        "pageup" => 112,
        "pagedown" => 117,
        "up" => 111,
        "down" => 116,
        "left" => 113,
        "right" => 114,
        "space" => 65,
        "capslock" => 66,
        "numlock" => 77,
        "scrolllock" => 78,
        "printscreen" | "sysrq" | "print" => 107,
        "pause" | "break" => 127,
        "f1" => 67, "f2" => 68, "f3" => 69, "f4" => 70,
        "f5" => 71, "f6" => 72, "f7" => 73, "f8" => 74,
        "f9" => 75, "f10" => 76, "f11" => 95, "f12" => 96,
        "f13" => 191, "f14" => 192, "f15" => 193, "f16" => 194,
        "f17" => 195, "f18" => 196, "f19" => 197, "f20" => 198,
        "f21" => 199, "f22" => 200, "f23" => 201, "f24" => 202,
        "ctrl" | "lctrl" | "leftctrl" => 37,
        "rctrl" | "rightctrl" => 105,
        "shift" | "lshift" | "leftshift" => 50,
        "rshift" | "rightshift" => 62,
        "alt" | "lalt" | "leftalt" => 64,
        "ralt" | "rightalt" => 108,
        "super" | "meta" | "lsuper" | "leftmeta" | "leftsuper" | "windows" | "leftwindows" => 133,
        "rsuper" | "rightmeta" | "rightsuper" | "rightwindows" => 134,
        "menu" | "compose" => 135,
        "kp_0" | "kp0" => 90, "kp_1" | "kp1" => 87,
        "kp_2" | "kp2" => 88, "kp_3" | "kp3" => 89,
        "kp_4" | "kp4" => 83, "kp_5" | "kp5" => 84,
        "kp_6" | "kp6" => 85, "kp_7" | "kp7" => 79,
        "kp_8" | "kp8" => 80, "kp_9" | "kp9" => 81,
        "kp_decimal" | "kp_dot" => 91,
        "kp_divide" | "kp_slash" => 106,
        "kp_multiply" | "kp_asterisk" => 63,
        "kp_subtract" | "kp_minus" => 82,
        "kp_add" | "kp_plus" => 86,
        "kp_enter" => 104,
        _ => return None,
    })
}

/// Keysym for one component of an xdotool-style key spec that is not a named key: a single
/// literal character (`a`, `7`, `-`, `#`) via the Unicode mapping, or a keysym NAME
/// (`minus`, `bracketleft`, `udiaeresis`, …) via xkb's name lookup (case-tolerant fallback
/// so `Minus` still resolves). Returns 0 when the spec names no keysym.
fn keysym_for_key_spec(name: &str) -> u32 {
    let mut chars = name.chars();
    if let (Some(c), None) = (chars.next(), chars.next()) {
        return keysym_for_char(c);
    }
    let sym = xkb::keysym_from_name(name, xkb::KEYSYM_NO_FLAGS).raw();
    if sym != 0 {
        return sym;
    }
    xkb::keysym_from_name(name, xkb::KEYSYM_CASE_INSENSITIVE).raw()
}

fn is_modifier(scancode: u32) -> bool {
    matches!(scancode, 37 | 105 | 50 | 62 | 64 | 108 | 133 | 134)
}

/// Semantic mouse-button vocabulary of the action layer; each backend maps it to its native
/// codes (evdev `BTN_` on Wayland, X core buttons 1/2/3 on X11).
#[derive(Clone, Copy)]
pub enum CuButton {
    Left,
    Right,
    Middle,
}

/// The desktop-side primitives one Computer Use action decomposes into. Keycodes are X/xkb
/// keycodes (evdev + 8) — the numbering both the smithay seat and XTEST consume — and
/// coordinates are framebuffer/root pixels, already clamped by the action layer.
pub trait CuBackend {
    fn name(&self) -> &'static str;
    fn fb_size(&self) -> Result<(i32, i32), String>;
    /// One display's framebuffer size; 0 = the primary. Unknown ids are an error. The
    /// X11 backend serves only the root (display 0); Wayland resolves any live output id.
    fn display_fb_size(&self, display: u32) -> Result<(i32, i32), String> {
        if display == 0 {
            self.fb_size()
        } else {
            Err(format!("Unknown display: {display}"))
        }
    }
    fn key(&self, scancode: u32, pressed: bool);
    fn mouse_move(&self, x: f64, y: f64);
    fn button(&self, btn: CuButton, pressed: bool);
    fn scroll(&self, dx: f64, dy: f64);
    /// PNG of one display's framebuffer; 0 = the primary. Unknown ids are an error.
    fn screenshot_png(&self, display: u32) -> Result<Vec<u8>, String>;
    fn cursor_pos(&self) -> Result<(f64, f64), String>;
    /// Run `seq` with every keysym in `keysyms` (which `resolve_keysyms` could not place)
    /// made temporarily typeable, when the backend can arrange that; `seq` receives
    /// keysym -> keycode for the transient bindings (empty when none could be arranged,
    /// in which case those keysyms simply stay untypeable). Wayland's `resolve_keysyms`
    /// already overlay-binds anything the base keymap lacks, so the default runs `seq`
    /// with no bindings; the X11 backend overrides this with a grab-atomic spare-keycode
    /// remap.
    fn with_transient_keysyms(&self, keysyms: &[u32], seq: &mut dyn FnMut(&HashMap<u32, u32>)) {
        let _ = keysyms;
        seq(&HashMap::new());
    }
    /// Resolve keysyms against the backend's ACTIVE keymap: one `(keycode, shift level)`
    /// per input keysym, `(0, 0)` where the keymap cannot produce it. Level bit 0 = Shift,
    /// bit 1 = AltGr — the order xkb two/four-level key types use. Wayland resolves through
    /// the compositor's keymap policy (overlay-binding what the base lacks); X11 through
    /// the server's own `GetKeyboardMapping`.
    fn resolve_keysyms(&self, keysyms: &[u32]) -> Vec<(u32, u32)>;
    /// Keycode synthesized around AltGr-level hits (level bit 1). The default is the
    /// pc105 right-Alt position the compositor's seat keymap binds to `ISO_Level3_Shift`;
    /// the X11 backend overrides it with the level-3 modifier key found in the server's
    /// keymap (0 = none, in which case `resolve_keysyms` reports no AltGr levels).
    fn altgr_keycode(&self) -> u32 {
        KC_ALTGR
    }
}

const BTN_LEFT: u32 = 0x110;
const BTN_RIGHT: u32 = 0x111;
const BTN_MIDDLE: u32 = 0x112;

/// Reply deadline for compositor round-trips (screenshot readback, cursor position, fb size).
/// Bounded so a wedged render path turns into a JSON error instead of hanging the sequential
/// HTTP loop — and every request behind it — forever.
const REPLY_TIMEOUT: Duration = Duration::from_secs(5);

/// Wayland implementation: every primitive is a `ThreadCommand` on the compositor's calloop
/// channel, so injection and readback serialize naturally with rendering and encoding.
pub struct CuWaylandBackend {
    tx: smithay::reexports::calloop::channel::Sender<ThreadCommand>,
}

impl CuBackend for CuWaylandBackend {
    fn name(&self) -> &'static str {
        "wayland"
    }

    fn fb_size(&self) -> Result<(i32, i32), String> {
        self.display_fb_size(0)
    }

    fn display_fb_size(&self, display: u32) -> Result<(i32, i32), String> {
        let (resp_tx, resp_rx) = mpsc::channel();
        self.tx.send(ThreadCommand::CuGetInfo { display_id: display, resp: resp_tx })
            .map_err(|_| "Failed to request compositor info".to_string())?;
        let (w, h, _) = resp_rx.recv_timeout(REPLY_TIMEOUT)
            .map_err(|_| "Compositor info request failed".to_string())?;
        // CuGetInfo reports zeros for output ids that don't exist (and for an output
        // with no mode, which live outputs always have).
        if w <= 0 || h <= 0 {
            return Err(format!("Unknown display: {display}"));
        }
        Ok((w, h))
    }

    fn key(&self, scancode: u32, pressed: bool) {
        let _ = self.tx.send(ThreadCommand::KeyboardKey {
            scancode,
            state: if pressed { 1 } else { 0 },
        });
    }

    fn mouse_move(&self, x: f64, y: f64) {
        let _ = self.tx.send(ThreadCommand::PointerMotion { x, y });
    }

    fn button(&self, btn: CuButton, pressed: bool) {
        let btn = match btn {
            CuButton::Left => BTN_LEFT,
            CuButton::Right => BTN_RIGHT,
            CuButton::Middle => BTN_MIDDLE,
        };
        let _ = self.tx.send(ThreadCommand::PointerButton {
            btn,
            state: if pressed { 1 } else { 0 },
        });
    }

    fn scroll(&self, dx: f64, dy: f64) {
        let _ = self.tx.send(ThreadCommand::PointerAxis { x: dx, y: dy });
    }

    fn screenshot_png(&self, display: u32) -> Result<Vec<u8>, String> {
        let (resp_tx, resp_rx) = mpsc::channel();
        self.tx.send(ThreadCommand::CuScreenshot { display_id: display, resp: resp_tx })
            .map_err(|_| "Failed to request screenshot".to_string())?;
        resp_rx.recv_timeout(REPLY_TIMEOUT).map_err(|_| "Screenshot failed".to_string())?
    }

    fn cursor_pos(&self) -> Result<(f64, f64), String> {
        let (resp_tx, resp_rx) = mpsc::channel();
        self.tx.send(ThreadCommand::CuCursorPosition { resp: resp_tx })
            .map_err(|_| "Failed to request cursor position".to_string())?;
        resp_rx.recv_timeout(REPLY_TIMEOUT).map_err(|_| "Cursor position failed".to_string())
    }

    fn resolve_keysyms(&self, keysyms: &[u32]) -> Vec<(u32, u32)> {
        let (resp_tx, resp_rx) = mpsc::channel();
        if self
            .tx
            .send(ThreadCommand::BindKeysyms { keysyms: keysyms.to_vec(), reply: resp_tx })
            .is_err()
        {
            return vec![(0, 0); keysyms.len()];
        }
        resp_rx
            .recv_timeout(REPLY_TIMEOUT)
            .unwrap_or_else(|_| vec![(0, 0); keysyms.len()])
    }
}

/// Per-request literal-key resolver: batches keysym lookups against the backend's active
/// keymap and caches them for the request's burst of key events (a fresh backend — and thus
/// a fresh cache — is resolved per HTTP request, so a runtime layout switch is picked up by
/// the next request).
struct KeyResolver<'a> {
    backend: &'a dyn CuBackend,
    cache: HashMap<u32, (u32, u32)>,
}

impl<'a> KeyResolver<'a> {
    fn new(backend: &'a dyn CuBackend) -> Self {
        Self { backend, cache: HashMap::new() }
    }

    /// Resolve a batch up front so a `type` action costs one backend round trip.
    fn prefetch(&mut self, keysyms: &[u32]) {
        let missing: Vec<u32> = {
            let mut seen = std::collections::HashSet::new();
            keysyms
                .iter()
                .copied()
                .filter(|&s| s != 0 && !self.cache.contains_key(&s) && seen.insert(s))
                .collect()
        };
        if missing.is_empty() {
            return;
        }
        let out = self.backend.resolve_keysyms(&missing);
        for (sym, hit) in missing.into_iter().zip(out) {
            self.cache.insert(sym, hit);
        }
    }

    /// `(keycode, shift level)` for `keysym`, or `None` when the active keymap cannot
    /// produce it.
    fn resolve(&mut self, keysym: u32) -> Option<(u32, u32)> {
        if keysym == 0 {
            return None;
        }
        self.prefetch(&[keysym]);
        let hit = self.cache.get(&keysym).copied().unwrap_or((0, 0));
        (hit.0 != 0).then_some(hit)
    }
}

/// Shift and default AltGr (pc105 right Alt / ISO_Level3_Shift) keycodes, held to reach a
/// resolved key's shift level: bit 0 = Shift, bit 1 = AltGr.
const KC_SHIFT: u32 = 50;
const KC_ALTGR: u32 = 108;

/// The level modifiers `level` requires beyond whatever is already held (`held_mask` uses
/// the same bit layout), in press order. `altgr` is the backend's AltGr keycode
/// ([`CuBackend::altgr_keycode`]).
fn level_modifiers(level: u32, held_mask: u32, altgr: u32) -> Vec<u32> {
    let mut out = Vec::new();
    if level & 1 != 0 && held_mask & 1 == 0 {
        out.push(KC_SHIFT);
    }
    if level & 2 != 0 && held_mask & 2 == 0 && altgr != 0 {
        out.push(altgr);
    }
    out
}

#[derive(Deserialize)]
struct CuActionRequest {
    action: String,
    coordinate: Option<[f64; 2]>,
    start_coordinate: Option<[f64; 2]>,
    text: Option<String>,
    key: Option<String>,
    scroll_direction: Option<String>,
    scroll_amount: Option<i32>,
    duration: Option<f64>,
    region: Option<[f64; 4]>,
    /// Display id for `screenshot` / `zoom` (default 0, the primary; `record_start` uses
    /// its own `display` field with the same meaning). Unknown ids get an error reply.
    display: Option<u32>,
}

fn ok_json() -> String {
    "{\"result\":\"ok\"}".to_string()
}

fn handle_action(req: CuActionRequest, backend: &dyn CuBackend) -> String {
    let result = handle_action_inner(req, backend);
    match result {
        Ok(response) => response,
        Err(e) => format!("{{\"error\":\"{}\"}}", e.replace('"', "\\\"")),
    }
}

/// Turn one parsed Computer Use request into a real action on the captured desktop — the
/// bridge that lets an external AI agent see and drive the session.
///
/// Every action ultimately becomes a backend input primitive or a framebuffer read, so this is
/// where the API's vocabulary meets the running desktop. Coordinates are clamped to the current
/// framebuffer so a mistaken agent click cannot address off-screen pixels; keyboard and pointer
/// actions translate into input events; and actions that must look at the screen (`screenshot`,
/// `zoom`) request a fresh capture first, so the agent reasons about the current frame rather than
/// a stale one. Returns the JSON response, or an error string explaining why the action could not
/// run.
fn handle_action_inner(req: CuActionRequest, b: &dyn CuBackend) -> Result<String, String> {
    let sleep_ms = |ms: u64| thread::sleep(Duration::from_millis(ms));

    let (fb_w, fb_h) = b.fb_size()?;

    let handle_coord = |coord: [f64; 2]| -> (f64, f64) {
        (
            clamp(coord[0], 0.0, (fb_w - 1) as f64),
            clamp(coord[1], 0.0, (fb_h - 1) as f64),
        )
    };

    let handle_modifier = |mod_name: &str| -> Option<u32> {
        let sc = scancode_for_keyname(mod_name)?;
        if is_modifier(sc) { Some(sc) } else { None }
    };

    match req.action.as_str() {
        "screenshot" => {
            let png = b.screenshot_png(req.display.unwrap_or(0))?;
            let b64 = BASE64.encode(&png);
            Ok(format!("{{\"data\":\"{}\"}}", b64))
        }

        "mouse_move" => {
            let coord = req.coordinate.ok_or("Missing coordinate")?;
            let (fx, fy) = handle_coord(coord);
            b.mouse_move(fx, fy);
            Ok(ok_json())
        }

        "left_click" | "right_click" | "middle_click" => {
            let btn = match req.action.as_str() {
                "left_click" => CuButton::Left,
                "right_click" => CuButton::Right,
                _ => CuButton::Middle,
            };
            if let Some(coord) = req.coordinate {
                let (fx, fy) = handle_coord(coord);
                b.mouse_move(fx, fy);
                sleep_ms(30);
            }
            if let Some(ref mod_name) = req.text {
                if let Some(sc) = handle_modifier(mod_name) {
                    b.key(sc, true);
                    sleep_ms(20);
                }
            }
            b.button(btn, true);
            sleep_ms(20);
            b.button(btn, false);
            if let Some(ref mod_name) = req.text {
                if let Some(sc) = handle_modifier(mod_name) {
                    sleep_ms(10);
                    b.key(sc, false);
                }
            }
            Ok(ok_json())
        }

        "double_click" | "triple_click" => {
            let n = if req.action == "double_click" { 2 } else { 3 };
            if let Some(coord) = req.coordinate {
                let (fx, fy) = handle_coord(coord);
                b.mouse_move(fx, fy);
                sleep_ms(30);
            }
            if let Some(ref mod_name) = req.text {
                if let Some(sc) = handle_modifier(mod_name) {
                    b.key(sc, true);
                    sleep_ms(20);
                }
            }
            for _ in 0..n {
                b.button(CuButton::Left, true);
                sleep_ms(10);
                b.button(CuButton::Left, false);
                sleep_ms(10);
            }
            if let Some(ref mod_name) = req.text {
                if let Some(sc) = handle_modifier(mod_name) {
                    sleep_ms(10);
                    b.key(sc, false);
                }
            }
            Ok(ok_json())
        }

        "left_click_drag" => {
            let start = req.start_coordinate.ok_or("Missing start_coordinate")?;
            let end = req.coordinate.ok_or("Missing coordinate")?;
            let (sx, sy) = handle_coord(start);
            let (ex, ey) = handle_coord(end);
            b.mouse_move(sx, sy);
            sleep_ms(30);
            b.button(CuButton::Left, true);
            sleep_ms(30);
            b.mouse_move(ex, ey);
            sleep_ms(30);
            b.button(CuButton::Left, false);
            Ok(ok_json())
        }

        "left_mouse_down" => {
            b.button(CuButton::Left, true);
            Ok(ok_json())
        }

        "left_mouse_up" => {
            b.button(CuButton::Left, false);
            Ok(ok_json())
        }

        "type" => {
            let text = req.text.as_deref().ok_or("Missing text")?;
            // A nested app compositor owns the apps on its own socket; type there as
            // a virtual-keyboard client, since keys on pixelflux's own seat carry an
            // overlay keymap the inner compositor never sees. Falls through to the
            // local seat if the app socket is unreachable.
            if b.name() == "wayland" {
                if let Some(sock) = app_wayland_socket_path() {
                    // Failures log once per socket value; every request still
                    // retries, so a compositor that comes back is used again
                    // immediately (and re-arms the logging).
                    static FAILED_SOCK: Mutex<Option<String>> = Mutex::new(None);
                    match crate::wayland::vkclient::type_text_to(&sock, text) {
                        Ok(()) => {
                            *FAILED_SOCK.lock().unwrap() = None;
                            return Ok(ok_json());
                        }
                        Err(e) => {
                            let mut last = FAILED_SOCK.lock().unwrap();
                            if last.as_deref() != Some(sock.as_str()) {
                                eprintln!(
                                    "[ComputerUse] app-compositor type via {sock} failed ({e}); using local seat until it is reachable"
                                );
                                *last = Some(sock);
                            }
                        }
                    }
                }
            }
            let mut resolver = KeyResolver::new(b);
            let syms: Vec<u32> = text.chars().map(keysym_for_char).collect();
            resolver.prefetch(&syms);
            // Base+AltGr resolution stays the preferred path; only what the active keymap
            // cannot reach at all goes through the backend's transient-bind fallback.
            let mut unresolved: Vec<u32> = Vec::new();
            for &sym in &syms {
                if sym != 0 && resolver.resolve(sym).is_none() && !unresolved.contains(&sym) {
                    unresolved.push(sym);
                }
            }
            b.with_transient_keysyms(&unresolved, &mut |bound| {
                for (i, &sym) in syms.iter().enumerate() {
                    if i > 0 && i % 50 == 0 {
                        sleep_ms(20);
                    }
                    if let Some((sc, level)) = resolver.resolve(sym) {
                        let level_mods = level_modifiers(level, 0, b.altgr_keycode());
                        for &m in &level_mods {
                            b.key(m, true);
                            sleep_ms(5);
                        }
                        b.key(sc, true);
                        sleep_ms(10);
                        b.key(sc, false);
                        for &m in level_mods.iter().rev() {
                            b.key(m, false);
                        }
                        sleep_ms(8);
                    } else if let Some(&kc) = bound.get(&sym) {
                        // Transient binds sit at the plain level: no modifiers needed.
                        b.key(kc, true);
                        sleep_ms(10);
                        b.key(kc, false);
                        sleep_ms(8);
                    }
                }
            });
            Ok(ok_json())
        }

        "key" => {
            let text = req.text.as_deref().ok_or("Missing text")?;
            let mut resolver = KeyResolver::new(b);
            let mut mods: Vec<u32> = Vec::new();
            let mut main_key: Option<(u32, u32)> = None;
            // Keysym of the last main-key spec the active keymap could not place (cleared
            // when a later part resolves): typed via the transient-bind fallback.
            let mut unresolved_sym: Option<u32> = None;
            for part in text.split('+') {
                let trimmed = part.trim();
                if let Some(sc) = scancode_for_keyname(trimmed) {
                    if is_modifier(sc) {
                        mods.push(sc);
                    } else {
                        main_key = Some((sc, 0));
                        unresolved_sym = None;
                    }
                } else {
                    let sym = keysym_for_key_spec(trimmed);
                    if let Some(hit) = resolver.resolve(sym) {
                        main_key = Some(hit);
                        unresolved_sym = None;
                    } else if sym != 0 {
                        main_key = None;
                        unresolved_sym = Some(sym);
                    }
                }
            }
            let unresolved: Vec<u32> = unresolved_sym.into_iter().collect();
            b.with_transient_keysyms(&unresolved, &mut |bound| {
                let main_key = main_key.or_else(|| {
                    // Transient binds sit at the plain level: no modifiers needed.
                    unresolved_sym.and_then(|s| bound.get(&s)).map(|&kc| (kc, 0))
                });
                for &sc in &mods {
                    b.key(sc, true);
                    sleep_ms(10);
                }
                if let Some((sc, level)) = main_key {
                    // Modifiers reaching the key's shift level are added unless the spec
                    // already asked for them (Shift as 50/62, AltGr as 108 or the backend's
                    // resolved level-3 keycode).
                    let altgr = b.altgr_keycode();
                    let held = mods.iter().fold(0u32, |m, &sc| match sc {
                        50 | 62 => m | 1,
                        sc if sc == KC_ALTGR || sc == altgr => m | 2,
                        _ => m,
                    });
                    let level_mods = level_modifiers(level, held, altgr);
                    for &m in &level_mods {
                        b.key(m, true);
                        sleep_ms(10);
                    }
                    b.key(sc, true);
                    sleep_ms(30);
                    b.key(sc, false);
                    for &m in level_mods.iter().rev() {
                        sleep_ms(10);
                        b.key(m, false);
                    }
                } else if let Some(&last_mod) = mods.last() {
                    b.key(last_mod, true);
                    sleep_ms(30);
                    b.key(last_mod, false);
                }
                for &sc in mods.iter().rev() {
                    sleep_ms(10);
                    b.key(sc, false);
                }
            });
            Ok(ok_json())
        }

        "hold_key" => {
            let text = req.text.as_deref().ok_or("Missing text")?;
            let duration = req.duration.ok_or("Missing duration")?;
            let duration = duration.min(100.0);
            let trimmed = text.trim();
            let mut resolver = KeyResolver::new(b);
            let hit = match scancode_for_keyname(trimmed) {
                Some(sc) => Some((sc, 0)),
                None => resolver.resolve(keysym_for_key_spec(trimmed)),
            };
            if let Some((sc, level)) = hit {
                let level_mods = level_modifiers(level, 0, b.altgr_keycode());
                for &m in &level_mods {
                    b.key(m, true);
                    sleep_ms(10);
                }
                b.key(sc, true);
                sleep_ms((duration * 1000.0) as u64);
                b.key(sc, false);
                for &m in level_mods.iter().rev() {
                    sleep_ms(10);
                    b.key(m, false);
                }
            }
            Ok(ok_json())
        }

        "scroll" => {
            let dir = req.scroll_direction.as_deref().ok_or("Missing scroll_direction")?;
            let amount = req.scroll_amount.unwrap_or(1).max(0) as f64;
            if let Some(coord) = req.coordinate {
                let (fx, fy) = handle_coord(coord);
                b.mouse_move(fx, fy);
                sleep_ms(30);
            }
            if let Some(ref mod_name) = req.text {
                if let Some(sc) = handle_modifier(mod_name) {
                    b.key(sc, true);
                    sleep_ms(20);
                }
            }
            let (dx, dy) = match dir {
                "up" => (0.0, -amount),
                "down" => (0.0, amount),
                "left" => (-amount, 0.0),
                "right" => (amount, 0.0),
                _ => return Err(format!("Invalid scroll_direction: {}", dir)),
            };
            b.scroll(dx, dy);
            sleep_ms(30);
            if let Some(ref mod_name) = req.text {
                if let Some(sc) = handle_modifier(mod_name) {
                    sleep_ms(10);
                    b.key(sc, false);
                }
            }
            Ok(ok_json())
        }

        "cursor_position" => {
            let (x, y) = b.cursor_pos()?;
            Ok(format!("{{\"text\":\"X={},Y={}\"}}", x.round() as i32, y.round() as i32))
        }

        "wait" => {
            let duration = req.duration.unwrap_or(1.0).min(100.0);
            sleep_ms((duration * 1000.0) as u64);
            Ok(ok_json())
        }

        "zoom" => {
            let region = req.region.ok_or("Missing region")?;
            let display = req.display.unwrap_or(0);
            // The region clamps against the TARGET display's own framebuffer, which need
            // not match the primary's.
            let (dw, dh) = b.display_fb_size(display)?;
            let (x0, y0, x1, y1) = (region[0], region[1], region[2], region[3]);
            let left = clamp(x0.round() as u32, 0, dw as u32 - 1);
            let top = clamp(y0.round() as u32, 0, dh as u32 - 1);
            let right = clamp(x1.round() as u32, left + 1, dw as u32);
            let bottom = clamp(y1.round() as u32, top + 1, dh as u32);
            let crop_w = right - left;
            let crop_h = bottom - top;
            let png = b.screenshot_png(display)?;
            if crop_w > 0 && crop_h > 0 {
                if let Ok(img) = image::load_from_memory(&png) {
                    let cropped = img.crop_imm(left, top, crop_w, crop_h);
                    let mut out = Vec::new();
                    cropped.write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
                        .map_err(|e| format!("Crop/encode error: {}", e))?;
                    let b64 = BASE64.encode(&out);
                    return Ok(format!("{{\"data\":\"{}\"}}", b64));
                }
            }
            let b64 = BASE64.encode(&png);
            Ok(format!("{{\"data\":\"{}\"}}", b64))
        }

        _ => Err(format!("Unknown action: {}", req.action)),
    }
}

static WAYLAND_TX: Mutex<Option<smithay::reexports::calloop::channel::Sender<ThreadCommand>>> =
    Mutex::new(None);

/// Body of `POST /record_start`. All fields are optional; unset ones fall back to the
/// `PIXELFLUX_RECORD_*` environment variables and built-in defaults.
#[derive(Deserialize, Default)]
struct RecordStartRequest {
    /// Output MP4 path (default: `$PIXELFLUX_RECORD`, else `/tmp/pixelflux-record-<unix_ts>.mp4`).
    path: Option<String>,
    /// Wayland output id to record (default 0; ignored on X11).
    display: Option<u32>,
    /// Capture fps cap for a recorder-owned capture.
    fps: Option<f64>,
    /// Bitrate override in kbps for a recorder-owned capture.
    bitrate_kbps: Option<i32>,
}

/// Handle the recorder REST endpoints sharing the CU server: `record_start`,
/// `record_stop` and `record_status`, all one JSON round-trip into the same recorder
/// implementation the Python API and env vars use. Returns `None` for any other URL.
fn handle_record_endpoint(url: &str, body: &str) -> Option<String> {
    let reply = match url {
        "/record_start" => {
            let req: RecordStartRequest = if body.trim().is_empty() {
                RecordStartRequest::default()
            } else {
                match serde_json::from_str(body) {
                    Ok(r) => r,
                    Err(e) => return Some(format!("{{\"error\":\"Invalid JSON: {}\"}}", e)),
                }
            };
            let path = req
                .path
                .or_else(|| std::env::var("PIXELFLUX_RECORD").ok().filter(|p| !p.is_empty()))
                .unwrap_or_else(|| {
                    let ts = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    format!("/tmp/pixelflux-record-{ts}.mp4")
                });
            let mut opts = crate::recorder::RecordOptions::from_env(path);
            if let Some(d) = req.display {
                opts.display_id = d;
            }
            if let Some(f) = req.fps {
                opts.fps = f;
            }
            if let Some(b) = req.bitrate_kbps {
                opts.bitrate_kbps = b;
            }
            match crate::recorder::start(opts) {
                Ok(s) => crate::recorder::status_to_json(&s).to_string(),
                Err(e) => serde_json::json!({ "error": e }).to_string(),
            }
        }
        "/record_stop" => match crate::recorder::stop() {
            Ok(s) => {
                let mut v = crate::recorder::status_to_json(&s);
                v["stopped"] = serde_json::Value::Bool(true);
                v.to_string()
            }
            Err(e) => serde_json::json!({ "error": e }).to_string(),
        },
        "/record_status" => match crate::recorder::status() {
            Some(s) => crate::recorder::status_to_json(&s).to_string(),
            None => serde_json::json!({ "active": false }).to_string(),
        },
        _ => return None,
    };
    Some(reply)
}

/// Socket of the compositor apps run under when it is nested under pixelflux
/// (labwc/kwin): keys injected into pixelflux's own seat carry an overlay keymap
/// the inner compositor never sees, so CU text is typed as a client of this
/// socket instead. Set by selkies over the ScreenCapture ABI.
static CU_APP_WAYLAND_DISPLAY: Mutex<Option<String>> = Mutex::new(None);

/// Set (or clear, with None/empty) the app compositor socket for CU text injection.
pub fn set_app_wayland_display(display: Option<String>) {
    *CU_APP_WAYLAND_DISPLAY.lock().unwrap() = display.filter(|s| !s.is_empty());
}

/// Resolve the app compositor socket PATH for CU typing, or None to type on the
/// local seat. The ABI value selkies set wins; a standalone CU (no selkies) falls
/// back to PIXELFLUX_APP_WAYLAND_DISPLAY. A value naming pixelflux's own
/// compositor means nothing is nested.
fn app_wayland_socket_path() -> Option<String> {
    let name = CU_APP_WAYLAND_DISPLAY
        .lock()
        .unwrap()
        .clone()
        .or_else(|| std::env::var("PIXELFLUX_APP_WAYLAND_DISPLAY").ok())
        .filter(|s| !s.is_empty())?;
    if crate::wait_socket_name(Duration::from_millis(0)).as_deref() == Some(name.as_str()) {
        return None;
    }
    crate::wayland::wlclient::socket_path(&name)
}

/// Make the Wayland compositor the preferred CU backend: once a live calloop sender is
/// registered, every subsequent request routes to it instead of an X11 connection.
pub fn register_wayland_backend(tx: smithay::reexports::calloop::channel::Sender<ThreadCommand>) {
    *WAYLAND_TX.lock().unwrap() = Some(tx);
}

/// The registered compositor's command channel, if a Wayland compositor is running in this
/// process. The recorder uses it to attach to (or start) a capture without any Python client.
pub(crate) fn wayland_command_sender(
) -> Option<smithay::reexports::calloop::channel::Sender<ThreadCommand>> {
    WAYLAND_TX.lock().unwrap().clone()
}

/// Start the CU server if `PIXELFLUX_CU` names a bind (the standalone fallback;
/// a selkies-managed session passes the setting through [`start_cu_server`]).
pub fn spawn_cu_from_env() {
    if let Ok(bind) = std::env::var("PIXELFLUX_CU") {
        start_cu_server(&bind);
    }
}

/// Start the CU server on `bind`: a bare port listens on all interfaces (the
/// container-deployment default), `host:port` scopes it. Guarded so that exactly
/// one server binds per process no matter how many call sites (module import,
/// Wayland compositor init, the selkies setting) race to spawn it; backend
/// selection stays per-request, so a server bound at import serves a compositor
/// that only starts later.
pub fn start_cu_server(bind: &str) {
    let addr = if bind.contains(':') {
        bind.to_string()
    } else {
        match bind.parse::<u16>() {
            Ok(port) => format!("0.0.0.0:{port}"),
            Err(_) => {
                println!("[ComputerUse] Invalid bind '{bind}' - expected a port or host:port");
                return;
            }
        }
    };
    static SPAWNED: OnceLock<()> = OnceLock::new();
    let mut first = false;
    SPAWNED.get_or_init(|| {
        first = true;
    });
    if first {
        thread::spawn(move || run_cu_server(addr));
    }
}

/// Pick the backend for one request: the registered Wayland compositor when present,
/// otherwise a fresh private connection to `DISPLAY`. The X11 connection is per-request so a
/// restarted X server never leaves the CU thread holding a dead connection.
fn resolve_backend() -> Result<Box<dyn CuBackend>, String> {
    if let Some(tx) = WAYLAND_TX.lock().unwrap().clone() {
        return Ok(Box::new(CuWaylandBackend { tx }));
    }
    crate::x11::computer_use::CuX11Backend::connect()
        .map(|be| Box::new(be) as Box<dyn CuBackend>)
}

/// Expose the captured desktop to an AI agent over HTTP, so a Computer Use client can drive
/// the session much as a human viewer would.
///
/// It runs on its own thread listening on `addr` for POST `/computer-use` JSON actions
/// (screenshot, mouse_move, click, key, scroll, …). The backend and the framebuffer dimensions
/// are re-resolved on every request rather than cached, because the compositor can start, the
/// stream can resize, or the X server can restart underneath the agent — a stale size would
/// misplace every coordinate. On Wayland a screenshot forces a one-frame GPU readback when the
/// pipeline is in zero-copy mode; on X11 it is a one-shot `GetImage` of the root window.
pub fn run_cu_server(addr: String) {
    println!("[ComputerUse] Server listening on {}", addr);

    let server = match tiny_http::Server::http(addr.as_str()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[ComputerUse] Failed to start server: {}", e);
            return;
        }
    };

    let mut last_backend = "";
    for mut request in server.incoming_requests() {
        // A CU body is a single input command: cap its size so a hostile client
        // of the (unauthenticated) endpoint cannot exhaust memory with a giant POST.
        const MAX_CU_BODY: u64 = 4 * 1024 * 1024;
        let mut body = String::new();
        if let Err(e) = request
            .as_reader()
            .take(MAX_CU_BODY + 1)
            .read_to_string(&mut body)
        {
            let _ = request.respond(tiny_http::Response::from_string(format!(
                "{{\"error\":\"{}\"}}", e
            ))
            .with_status_code(400)
            .with_header(
                "Content-Type: application/json".parse::<tiny_http::Header>().unwrap()
            ));
            continue;
        }

        if body.len() as u64 > MAX_CU_BODY {
            let resp = tiny_http::Response::from_string(
                "{\"error\":\"request body too large\"}".to_string(),
            )
            .with_status_code(413)
            .with_header(
                "Content-Type: application/json".parse::<tiny_http::Header>().unwrap()
            );
            let _ = request.respond(resp);
            continue;
        }

        if let Some(json) = handle_record_endpoint(request.url(), &body) {
            let _ = request.respond(
                tiny_http::Response::from_string(json)
                    .with_status_code(200)
                    .with_header(
                        "Content-Type: application/json".parse::<tiny_http::Header>().unwrap()
                    )
            );
            continue;
        }

        let parsed: CuActionRequest = match serde_json::from_str(&body) {
            Ok(r) => r,
            Err(e) => {
                let _ = request.respond(tiny_http::Response::from_string(format!(
                    "{{\"error\":\"Invalid JSON: {}\"}}", e
                ))
                .with_status_code(400)
                .with_header(
                    "Content-Type: application/json".parse::<tiny_http::Header>().unwrap()
                ));
                continue;
            }
        };

        let json_response = match resolve_backend() {
            Ok(backend) => {
                if backend.name() != last_backend {
                    last_backend = backend.name();
                    println!("[ComputerUse] Using {} backend", last_backend);
                }
                handle_action(parsed, backend.as_ref())
            }
            Err(e) => format!("{{\"error\":\"{}\"}}", e.replace('"', "\\\"")),
        };
        let _ = request.respond(
            tiny_http::Response::from_string(json_response)
                .with_status_code(200)
                .with_header(
                    "Content-Type: application/json".parse::<tiny_http::Header>().unwrap()
                )
        );
    }
}
