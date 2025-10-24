//! GELU (Gaussian Error Linear Unit) Activation Function
//!
//! GELU is the activation function used in modern transformer models including
//! GPT (Generative Pre-trained Transformer). It provides smooth, non-monotonic
//! behavior that helps with training deep networks.
//!
//! # Formula
//!
//! ```text
//! GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
//! ```
//!
//! This is the approximation formula from Hendrycks & Gimpel (2016).
//!
//! # Properties
//!
//! - **Smooth and differentiable**: Unlike ReLU, there are no sharp corners
//! - **Non-monotonic**: Small negative values can pass through (unlike ReLU)
//! - **Used in GPT**: The default activation in GPT-2, GPT-3, and BERT
//! - **Probabilistic interpretation**: Weights inputs by their value in a Gaussian CDF
//!
//! # Examples
//!
//! ```rust
//! use jwall_gpt::activations::gelu;
//!
//! // GELU allows small negative values through
//! assert!(gelu(-1.0) < 0.0);
//!
//! // For large positive values, acts like identity
//! assert!((gelu(5.0) - 5.0).abs() < 0.01);
//!
//! // Zero maps to zero
//! assert!((gelu(0.0)).abs() < 0.001);
//! ```
//!
//! # References
//!
//! - [Hendrycks & Gimpel, 2016: *Gaussian Error Linear Units (GELUs)*](https://arxiv.org/abs/1606.08415)

use std::f32::consts::PI;

/// Approximation of GELU from Hendrycks & Gimpel (2016)
pub fn gelu(x: f32) -> f32 {
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715x^3)))
    0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
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

    #[test]
    fn test_gelu_zero() {
        // GELU(0) should be exactly 0
        assert_float_eq(gelu(0.0), 0.0, 0.001);
    }

    #[test]
    fn test_gelu_positive() {
        // GELU(1.0) ≈ 0.841
        assert_float_eq(gelu(1.0), 0.841, 0.01);
    }

    #[test]
    fn test_gelu_negative() {
        // GELU allows small negative values through (unlike ReLU)
        // gelu(-1.0) ≈ -0.159
        assert_float_eq(gelu(-1.0), -0.159, 0.01);

        // Should be negative
        assert!(gelu(-1.0) < 0.0);

        // Should be small
        assert!(gelu(-1.0) > -0.2);
    }

    #[test]
    fn test_gelu_large_positive() {
        // For large positive x, GELU(x) ≈ x (acts like identity)
        assert_float_eq(gelu(5.0), 5.0, 0.01);
        assert_float_eq(gelu(10.0), 10.0, 0.01);
    }

    #[test]
    fn test_gelu_large_negative() {
        // For large negative x, GELU(x) ≈ 0
        assert!(gelu(-5.0).abs() < 0.01);
    }
}
