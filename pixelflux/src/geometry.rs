//! The geometry the frame pipeline speaks: rectangles and points in one space
//! or another, where the space is a marker type on the generic.
//!
//! `Physical` marks pixels on the capture's output; `Logical` marks the
//! desktop-layout coordinates captures are placed in. The types carry the same
//! shape every backend produces and consumes, so the pipeline, the striped
//! encoders, and the per-OS capture modules pass frames between each other
//! without conversion shims at the boundaries.

use std::marker::PhantomData;

/// Pixels on an output, as the capture and encoder measure them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Physical;

/// Desktop-layout coordinates a capture is placed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Logical;

/// An x/y position in the space `Space` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Point<N, Space> {
    pub x: N,
    pub y: N,
    _space: PhantomData<Space>,
}

/// A width/height in the space `Space` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Size<N, Space> {
    pub w: N,
    pub h: N,
    _space: PhantomData<Space>,
}

/// A rectangle: where it starts and how big it is, both in `Space`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rectangle<N, Space> {
    pub loc: Point<N, Space>,
    pub size: Size<N, Space>,
}

impl<N, Space> Rectangle<N, Space> {
    /// A rectangle from its top-left corner and its size.
    pub fn new(loc: Point<N, Space>, size: Size<N, Space>) -> Self {
        Self { loc, size }
    }
}

impl<N, Space> From<(N, N)> for Point<N, Space> {
    fn from(value: (N, N)) -> Self {
        Self {
            x: value.0,
            y: value.1,
            _space: PhantomData,
        }
    }
}

impl<N, Space> From<(N, N)> for Size<N, Space> {
    fn from(value: (N, N)) -> Self {
        Self {
            w: value.0,
            h: value.1,
            _space: PhantomData,
        }
    }
}
