//! Neural network layers
//!
//! This module contains implementations of various neural network layers.
//! All layers implement the `Layer` trait for a consistent interface.

pub mod layer_norm;

mod linear;
mod traits;

pub use linear::Linear;
pub use traits::Layer;
