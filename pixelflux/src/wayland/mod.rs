/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Wayland backend: a headless Smithay compositor that stands in for a real display server.
//!
//! `frontend` owns the compositor state machine, protocol handlers, and input routing.
//! `cursor` resolves Wayland cursor shapes to PNG images for the Python callback.

/// Headless Smithay compositor, protocol handlers, and input routing.
pub mod frontend;
/// Wayland cursor shape to PNG resolution and the cursor delivery worker.
pub mod cursor;
/// Seat keymap ownership: base layout plus batched overlay keysym binding.
pub mod keymap;
/// Virtual-keyboard client for typing into a nested app compositor's socket.
pub mod vkclient;
/// Shared plumbing for outbound Wayland client connections.
pub mod wlclient;
/// Data-control clipboard client bridging a nested app compositor's selection.
pub mod dcclient;
/// Host-capture mode: capture/inject as a client of an external compositor.
pub mod host;
