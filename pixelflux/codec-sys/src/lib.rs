/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Bindings for the software codec libraries pixelflux links, one module per library,
//! generated at build time from the headers of the library pkg-config located (`build.rs`).
//! A library is present only under its feature; the wheel enables all of them.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(clippy::all)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unnecessary_transmutes)]
#![allow(improper_ctypes)]

/// libvpx: the VP8 and VP9 encoders and decoders.
#[cfg(feature = "vpx")]
pub mod vpx {
    include!(concat!(env!("OUT_DIR"), "/vpx.rs"));
}

/// x265: the GPL HEVC encoder, reached through its versioned API table.
#[cfg(feature = "x265")]
pub mod x265 {
    include!(concat!(env!("OUT_DIR"), "/x265.rs"));

    /// `api.encoder_encode`, whose output takes an array of pictures, or on builds 210 to
    /// 212 an array of pointers to them; `out` points at two pictures either way.
    ///
    /// # Safety
    /// The pointers are the library's own, and `out` addresses two initialized pictures.
    pub unsafe fn encoder_encode(
        api: &x265_api,
        encoder: *mut x265_encoder,
        nals: *mut *mut x265_nal,
        count: *mut u32,
        pic_in: *mut x265_picture,
        out: *mut x265_picture,
    ) -> ::std::os::raw::c_int {
        let encode = api.encoder_encode.expect("x265's API table names encoder_encode");
        #[cfg(x265_layer_pointers)]
        {
            let mut layers = [out, unsafe { out.add(1) }];
            unsafe { encode(encoder, nals, count, pic_in, layers.as_mut_ptr()) }
        }
        #[cfg(not(x265_layer_pointers))]
        {
            unsafe { encode(encoder, nals, count, pic_in, out) }
        }
    }
}

/// kvazaar: the BSD HEVC encoder, reached through its API table.
#[cfg(feature = "kvazaar")]
pub mod kvazaar {
    include!(concat!(env!("OUT_DIR"), "/kvazaar.rs"));
}

/// SVT-AV1: the AV1 encoder. `svtav1_handle_priv` is set for a release before 3.0, whose
/// handle takes an application pointer, and `svtav1_rtc` for one from 3.1, which has the
/// real-time mode.
#[cfg(feature = "svtav1")]
pub mod svtav1 {
    include!(concat!(env!("OUT_DIR"), "/svtav1.rs"));
    pub const HAS_RTC: bool = cfg!(svtav1_rtc);

    /// `svt_av1_enc_init_handle`, with the application pointer a release before 3.0 took.
    ///
    /// # Safety
    /// `handle` and `config` are valid for writes.
    pub unsafe fn init_handle(handle: *mut *mut EbComponentType, config: *mut EbSvtAv1EncConfiguration) -> EbErrorType {
        #[cfg(svtav1_handle_priv)]
        {
            unsafe { svt_av1_enc_init_handle(handle, ::std::ptr::null_mut(), config) }
        }
        #[cfg(not(svtav1_handle_priv))]
        {
            unsafe { svt_av1_enc_init_handle(handle, config) }
        }
    }
}

/// dav1d: the AV1 decoder.
#[cfg(feature = "dav1d")]
pub mod dav1d {
    include!(concat!(env!("OUT_DIR"), "/dav1d.rs"));
}

/// libde265: the HEVC decoder.
#[cfg(feature = "de265")]
pub mod de265 {
    include!(concat!(env!("OUT_DIR"), "/de265.rs"));
}
