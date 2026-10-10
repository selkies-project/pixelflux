/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

//! Validated packed TrueColor layouts for independent X11 screenshots.

use x11rb::protocol::xproto::{ImageOrder, VisualClass};

#[derive(Clone, Copy, Debug)]
pub(super) struct ScreenshotFormat {
    pub rgb_bits: u8,
    masks: [u32; 3],
    shifts: [u32; 3],
    little_endian: bool,
    scanline_pad: usize,
}

impl ScreenshotFormat {
    pub fn new(
        depth: u8,
        bits_per_pixel: u8,
        scanline_pad: u8,
        byte_order: ImageOrder,
        visual_class: VisualClass,
        masks: [u32; 3],
    ) -> Result<Self, String> {
        let rgb_bits = match depth {
            24 | 32 => 8,
            30 => 10,
            _ => return Err(format!("Unsupported screenshot root depth {depth}")),
        };
        if bits_per_pixel != 32
            || !matches!(scanline_pad, 8 | 16 | 32)
            || visual_class != VisualClass::TRUE_COLOR
        {
            return Err("Screenshot requires a packed 32-bpp TrueColor visual".to_string());
        }
        let little_endian = match byte_order {
            ImageOrder::LSB_FIRST => true,
            ImageOrder::MSB_FIRST => false,
            _ => return Err("Unknown screenshot image byte order".to_string()),
        };
        let shifts = masks.map(u32::trailing_zeros);
        let max = (1u32 << rgb_bits) - 1;
        if masks.iter().zip(shifts).any(|(&mask, shift)| {
            mask == 0 || shift >= 32 || (mask >> shift) != max
        }) || masks[0] & masks[1] != 0
            || masks[0] & masks[2] != 0
            || masks[1] & masks[2] != 0
        {
            return Err("Unsupported screenshot TrueColor channel masks".to_string());
        }
        Ok(Self {
            rgb_bits,
            masks,
            shifts,
            little_endian,
            scanline_pad: scanline_pad as usize,
        })
    }

    pub fn stride(&self, width: u32) -> Result<usize, String> {
        (width as usize)
            .checked_mul(32)
            .and_then(|bits| bits.checked_add(self.scanline_pad - 1))
            .map(|bits| (bits / self.scanline_pad) * (self.scanline_pad / 8))
            .ok_or_else(|| "Screenshot row size overflow".to_string())
    }

    fn pixels<'a>(
        &'a self,
        data: &'a [u8],
        width: u32,
        height: u32,
    ) -> Result<impl Iterator<Item = [u16; 3]> + 'a, String> {
        let stride = self.stride(width)?;
        if width == 0
            || height == 0
            || stride.checked_mul(height as usize) != Some(data.len())
        {
            return Err("Screenshot buffer does not match its visual and geometry".to_string());
        }
        Ok(data.chunks_exact(stride).flat_map(move |row| {
            row[..width as usize * 4].as_chunks::<4>().0.iter().map(move |bytes| {
                let word = if self.little_endian {
                    u32::from_le_bytes(*bytes)
                } else {
                    u32::from_be_bytes(*bytes)
                };
                std::array::from_fn(|channel| {
                    ((word & self.masks[channel]) >> self.shifts[channel]) as u16
                })
            })
        }))
    }

    /// Normalize RGB8 visuals to the BGRA layout used by cursor compositing.
    pub fn normalize_bgra8(&self, data: &mut [u8], width: u32, height: u32) -> Result<(), String> {
        if self.rgb_bits != 8 {
            return Err("Refusing to reduce a higher-precision screenshot to RGB8".to_string());
        }
        let _ = self.pixels(data, width, height)?;
        if self.little_endian && self.shifts == [16, 8, 0] {
            return Ok(());
        }
        for bytes in data.as_chunks_mut::<4>().0 {
            let word = if self.little_endian {
                u32::from_le_bytes(*bytes)
            } else {
                u32::from_be_bytes(*bytes)
            };
            let [r, g, b] = std::array::from_fn(|channel| {
                ((word & self.masks[channel]) >> self.shifts[channel]) as u8
            });
            *bytes = [b, g, r, 255];
        }
        Ok(())
    }

    /// Expand RGB10 codes reversibly into network-order RGB16 PNG samples.
    pub fn rgb16(&self, data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
        if self.rgb_bits != 10 {
            return Err("Screenshot source is not native RGB10".to_string());
        }
        let pixels = self.pixels(data, width, height)?;
        let size = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(6))
            .ok_or("Screenshot RGB16 size overflow")?;
        let mut result = Vec::with_capacity(size);
        for rgb in pixels {
            for code in rgb {
                let sample = (code << 6) | (code >> 4);
                result.extend_from_slice(&sample.to_be_bytes());
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::ScreenshotFormat;
    use x11rb::protocol::xproto::{ImageOrder, VisualClass};

    fn format(depth: u8, order: ImageOrder, masks: [u32; 3]) -> ScreenshotFormat {
        ScreenshotFormat::new(depth, 32, 32, order, VisualClass::TRUE_COLOR, masks).unwrap()
    }

    #[test]
    fn packed_rgb10_preserves_every_code_in_each_byte_order_and_channel_order() {
        for order in [ImageOrder::LSB_FIRST, ImageOrder::MSB_FIRST] {
            for shifts in [[20, 10, 0], [0, 10, 20]] {
                let layout = format(30, order, shifts.map(|shift| 1023 << shift));
                let mut source = Vec::new();
                let mut expected = Vec::new();
                for code in 0..1024u32 {
                    let rgb = [code, 1023 - code, (73 * code) % 1024];
                    let word = 3 << 30
                        | rgb[0] << shifts[0]
                        | rgb[1] << shifts[1]
                        | rgb[2] << shifts[2];
                    source.extend_from_slice(&if order == ImageOrder::LSB_FIRST {
                        word.to_le_bytes()
                    } else {
                        word.to_be_bytes()
                    });
                    expected.extend(rgb);
                }
                let decoded = layout.rgb16(&source, 1024, 1).unwrap();
                let actual: Vec<u32> = decoded
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|bytes| (u16::from_be_bytes(*bytes) >> 6) as u32)
                    .collect();
                assert_eq!(actual, expected);
                assert!(layout.normalize_bgra8(&mut source, 1024, 1).is_err());
            }
        }
    }

    #[test]
    fn eight_bit_layouts_preserve_rgb_and_ignore_padding_bits() {
        for order in [ImageOrder::LSB_FIRST, ImageOrder::MSB_FIRST] {
            for shifts in [[16, 8, 0], [0, 8, 16]] {
                let layout = format(24, order, shifts.map(|shift| 255 << shift));
                let word = 0xa5000000 | 17u32 << shifts[0] | 91u32 << shifts[1] | 203u32 << shifts[2];
                let mut bytes = if order == ImageOrder::LSB_FIRST {
                    word.to_le_bytes()
                } else {
                    word.to_be_bytes()
                };
                layout.normalize_bgra8(&mut bytes, 1, 1).unwrap();
                assert_eq!(&bytes[..3], &[203, 91, 17]);
                assert!(layout.rgb16(&bytes, 1, 1).is_err());
            }
        }
    }

    #[test]
    fn depth32_with_validated_rgb8_masks_preserves_channels_without_claiming_rgb10() {
        for order in [ImageOrder::LSB_FIRST, ImageOrder::MSB_FIRST] {
            for shifts in [[16, 8, 0], [0, 8, 16]] {
                let layout = format(32, order, shifts.map(|shift| 255 << shift));
                assert_eq!(layout.rgb_bits, 8);
                for padding in [0u32, 0x55, 0xff] {
                    let word = padding << 24
                        | 17u32 << shifts[0]
                        | 91u32 << shifts[1]
                        | 203u32 << shifts[2];
                    let mut bytes = if order == ImageOrder::LSB_FIRST {
                        word.to_le_bytes()
                    } else {
                        word.to_be_bytes()
                    };
                    layout.normalize_bgra8(&mut bytes, 1, 1).unwrap();
                    assert_eq!(&bytes[..3], &[203, 91, 17]);
                    assert!(layout.rgb16(&bytes, 1, 1).is_err());
                }
            }
        }
        assert!(ScreenshotFormat::new(
            32, 32, 32, ImageOrder::LSB_FIRST, VisualClass::TRUE_COLOR,
            [0x3ff00000, 0xffc00, 0x3ff],
        ).is_err());
    }

    #[test]
    fn unsupported_or_malformed_layouts_fail_before_conversion() {
        for (depth, bpp, pad, class, masks) in [
            (16, 16, 16, VisualClass::TRUE_COLOR, [0xf800, 0x07e0, 0x001f]),
            (30, 32, 32, VisualClass::TRUE_COLOR, [0xff0000, 0xff00, 0xff]),
            (30, 32, 32, VisualClass::TRUE_COLOR, [0x3ff, 0x3ff, 0x3ff]),
            (30, 32, 32, VisualClass::TRUE_COLOR, [0, 0xffc00, 0x3ff]),
            (24, 24, 32, VisualClass::TRUE_COLOR, [0xff0000, 0xff00, 0xff]),
            (24, 32, 7, VisualClass::TRUE_COLOR, [0xff0000, 0xff00, 0xff]),
            (24, 32, 32, VisualClass::DIRECT_COLOR, [0xff0000, 0xff00, 0xff]),
        ] {
            assert!(ScreenshotFormat::new(depth, bpp, pad, ImageOrder::LSB_FIRST, class, masks).is_err());
        }
        let layout = format(30, ImageOrder::LSB_FIRST, [0x3ff00000, 0xffc00, 0x3ff]);
        for (width, height, len) in [(0, 1, 0), (1, 0, 0), (1, 1, 3), (1, 1, 5), (2, 1, 4)] {
            assert!(layout.rgb16(&vec![0; len], width, height).is_err());
        }
    }
}
