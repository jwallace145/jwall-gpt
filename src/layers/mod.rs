//! Neural network layers
//!
//! This module contains implementations of various neural network layers.
//! All layers implement the `Layer` trait for a consistent interface.

pub mod layer_norm;

mod attention;
mod linear;
mod multi_head;
mod traits;

pub use attention::Attention;
pub use linear::Linear;
pub use multi_head::MultiHeadAttention;
pub use traits::Layer;
