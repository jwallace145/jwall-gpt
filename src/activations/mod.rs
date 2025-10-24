//! Activation Functions for Neural Networks
//!
//! This module implements common activation functions used in deep learning,
//! particularly those relevant to transformer-based architectures such as GPT
//! (Generative Pre-trained Transformer).
//!
//! # Overview
//!
//! Activation functions introduce **non-linearity** into neural networks. Without them,
//! even a deep model would behave like a single linear transformation, unable to
//! capture complex relationships in data.
//!
//! GPT and other transformer models rely heavily on the **GELU** activation function
//! for its smooth and differentiable behavior.
//!
//! # Available Activation Functions
//!
//! - [`relu`] — Rectified Linear Unit, simple and widely used.
//! - [`gelu`] — Gaussian Error Linear Unit, used in GPT models.
//! - [`sigmoid`] — Squashes values into (0, 1), useful for gates or probabilities.
//!
//! # Examples
//!
//! ```rust
//! use jwall_gpt::activations::{relu_scalar, gelu, sigmoid_scalar};
//!
//! let x = 0.5;
//! assert_eq!(relu_scalar(x), 0.5);
//!
//! let g = gelu(x);
//! let s = sigmoid_scalar(x);
//!
//! println!("GELU: {g}, Sigmoid: {s}");
//! ```
//!
//! # Notes
//!
//! - These implementations prioritize **readability and educational value** over raw performance.
//! - In production frameworks (e.g. PyTorch, TensorFlow), optimized GPU kernels are typically used instead.
//! - GPT models (and most modern transformers) use the **GELU** activation by default.
//!
//! # References
//!
//! - [Hendrycks & Gimpel, 2016: *Gaussian Error Linear Units (GELUs)*](https://arxiv.org/abs/1606.08415)
//! - [Vaswani et al., 2017: *Attention is All You Need*](https://arxiv.org/abs/1706.03762)
//!
//! # Module Structure
//!
//! Each activation function is implemented in its own file for better organization
//! and maintainability:
//!
//! - `relu.rs` - ReLU activation function
//! - `gelu.rs` - GELU activation function
//! - `sigmoid.rs` - Sigmoid activation function
//!
//! This modular structure makes it easy to add new activation functions in the future.

pub mod gelu;
pub mod relu;
pub mod sigmoid;

// Re-export the functions for convenient access
pub use gelu::gelu;
pub use relu::{relu as relu_scalar, relu_derivative as relu_derivative_scalar};
pub use sigmoid::{sigmoid as sigmoid_scalar, sigmoid_derivative as sigmoid_derivative_scalar};

use crate::matrix::Matrix;

/// Apply ReLU activation element-wise to a matrix
#[cfg(feature = "blas")]
pub fn relu(m: &Matrix) -> Matrix {
    m.mapv(relu_scalar)
}

/// Apply ReLU derivative element-wise to a matrix
#[cfg(feature = "blas")]
pub fn relu_derivative(m: &Matrix) -> Matrix {
    m.mapv(relu_derivative_scalar)
}

/// Apply sigmoid activation element-wise to a matrix
#[cfg(feature = "blas")]
pub fn sigmoid(m: &Matrix) -> Matrix {
    m.mapv(sigmoid_scalar)
}

/// Apply sigmoid derivative element-wise to a matrix
#[cfg(feature = "blas")]
pub fn sigmoid_derivative(m: &Matrix) -> Matrix {
    m.mapv(sigmoid_derivative_scalar)
}

/// Apply ReLU activation element-wise to a matrix (naive backend)
#[cfg(not(feature = "blas"))]
pub fn relu(m: &Matrix) -> Matrix {
    m.iter()
        .map(|row| row.iter().map(|&x| relu_scalar(x)).collect())
        .collect()
}

/// Apply ReLU derivative element-wise to a matrix (naive backend)
#[cfg(not(feature = "blas"))]
pub fn relu_derivative(m: &Matrix) -> Matrix {
    m.iter()
        .map(|row| row.iter().map(|&x| relu_derivative_scalar(x)).collect())
        .collect()
}

/// Apply sigmoid activation element-wise to a matrix (naive backend)
#[cfg(not(feature = "blas"))]
pub fn sigmoid(m: &Matrix) -> Matrix {
    m.iter()
        .map(|row| row.iter().map(|&x| sigmoid_scalar(x)).collect())
        .collect()
}

/// Apply sigmoid derivative element-wise to a matrix (naive backend)
#[cfg(not(feature = "blas"))]
pub fn sigmoid_derivative(m: &Matrix) -> Matrix {
    m.iter()
        .map(|row| row.iter().map(|&x| sigmoid_derivative_scalar(x)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to compare floating point numbers with tolerance
    fn assert_float_eq(a: f32, b: f32, tolerance: f32) {
        assert!(
            (a - b).abs() < tolerance,
            "Expected {}, got {} (difference: {})",
            b,
            a,
            (a - b).abs()
        );
    }

    // ============================================
    // COMPARISON TESTS (Test Multiple Functions Together)
    // ============================================

    #[test]
    fn test_activation_comparison_at_zero() {
        // All activations at zero
        assert_float_eq(relu_scalar(0.0), 0.0, 0.001);
        assert_float_eq(gelu(0.0), 0.0, 0.001);
        assert_float_eq(sigmoid_scalar(0.0), 0.5, 0.001);
    }

    #[test]
    fn test_relu_vs_gelu() {
        // For large positive x, GELU ≈ ReLU
        assert_float_eq(relu_scalar(5.0), gelu(5.0), 0.01);

        // For negative x, ReLU = 0, but GELU allows small negative values
        assert_eq!(relu_scalar(-1.0), 0.0);
        assert!(gelu(-1.0) < 0.0); // GELU is negative here
    }
}
