//! Sigmoid Activation Function
//!
//! The sigmoid function squashes any input value into the range (0, 1), making it
//! useful for probability outputs, gates in LSTM/GRU networks, and binary classification.
//!
//! # Formula
//!
//! ```text
//! σ(x) = 1 / (1 + e^(-x))
//! ```
//!
//! # Properties
//!
//! - **Output range**: Always between 0 and 1
//! - **S-shaped curve**: Smooth, differentiable everywhere
//! - **Symmetry**: σ(x) + σ(-x) = 1
//! - **Vanishing gradients**: Can cause slow training in deep networks
//! - **Use cases**: Output layers for binary classification, gate mechanisms
//!
//! # Examples
//!
//! ```rust
//! use jwall_gpt::activations::sigmoid_scalar;
//!
//! // Sigmoid of 0 is 0.5 (middle of the range)
//! assert!((sigmoid_scalar(0.0) - 0.5).abs() < 0.001);
//!
//! // Large positive values approach 1.0
//! assert!(sigmoid_scalar(10.0) > 0.999);
//!
//! // Large negative values approach 0.0
//! assert!(sigmoid_scalar(-10.0) < 0.001);
//!
//! // Symmetry property
//! let x = 2.5;
//! assert!((sigmoid_scalar(x) + sigmoid_scalar(-x) - 1.0).abs() < 0.001);
//! ```

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Derivative of sigmoid activation function
///
/// The derivative is σ(x) * (1 - σ(x))
pub fn sigmoid_derivative(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 - s)
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
    fn test_sigmoid_zero() {
        // sigmoid(0) = 0.5 (right in the middle)
        assert_float_eq(sigmoid(0.0), 0.5, 0.001);
    }

    #[test]
    fn test_sigmoid_positive() {
        // sigmoid(2) ≈ 0.88
        assert_float_eq(sigmoid(2.0), 0.88, 0.01);
    }

    #[test]
    fn test_sigmoid_negative() {
        // sigmoid(-2) ≈ 0.12
        assert_float_eq(sigmoid(-2.0), 0.12, 0.01);
    }

    #[test]
    fn test_sigmoid_large_positive() {
        // Should approach 1.0 for large positive values
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(100.0) > 0.999);
    }

    #[test]
    fn test_sigmoid_large_negative() {
        // Should approach 0.0 for large negative values
        assert!(sigmoid(-10.0) < 0.001);
        assert!(sigmoid(-100.0) < 0.001);
    }

    #[test]
    fn test_sigmoid_symmetry() {
        // Property: sigmoid(x) + sigmoid(-x) = 1
        let x = 2.5;
        assert_float_eq(sigmoid(x) + sigmoid(-x), 1.0, 0.001);

        let x = -1.5;
        assert_float_eq(sigmoid(x) + sigmoid(-x), 1.0, 0.001);
    }

    #[test]
    fn test_sigmoid_range() {
        // Sigmoid should ALWAYS be between 0 and 1
        for i in -100..=100 {
            let x = i as f32;
            let y = sigmoid(x);
            assert!(
                y >= 0.0 && y <= 1.0,
                "sigmoid({}) = {} is out of range!",
                x,
                y
            );
        }
    }
}
