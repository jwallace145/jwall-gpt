//! ReLU (Rectified Linear Unit) Activation Function
//!
//! ReLU is one of the most widely used activation functions in deep learning.
//! It introduces non-linearity by outputting the input directly if positive,
//! otherwise outputting zero.
//!
//! # Formula
//!
//! ```text
//! ReLU(x) = max(0, x)
//! ```
//!
//! # Properties
//!
//! - **Simple and efficient**: Just a max operation
//! - **Solves vanishing gradient problem**: For positive values, gradient is 1
//! - **Sparse activation**: About 50% of neurons are zero in typical scenarios
//! - **Dying ReLU problem**: Neurons can become permanently inactive
//!
//! # Examples
//!
//! ```rust
//! use jwall_gpt::activations::relu_scalar;
//!
//! assert_eq!(relu_scalar(5.0), 5.0);   // Positive passes through
//! assert_eq!(relu_scalar(-3.0), 0.0);  // Negative becomes zero
//! assert_eq!(relu_scalar(0.0), 0.0);   // Zero stays zero
//! ```

pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Derivative of ReLU activation function
///
/// The derivative is 1 for positive values, 0 otherwise.
pub fn relu_derivative(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_positive() {
        // Positive values should pass through unchanged
        assert_eq!(relu(5.0), 5.0);
        assert_eq!(relu(100.0), 100.0);
        assert_eq!(relu(0.001), 0.001);
    }

    #[test]
    fn test_relu_negative() {
        // Negative values should become zero
        assert_eq!(relu(-3.0), 0.0);
        assert_eq!(relu(-100.0), 0.0);
        assert_eq!(relu(-0.001), 0.0);
    }

    #[test]
    fn test_relu_zero() {
        // Zero should stay zero
        assert_eq!(relu(0.0), 0.0);
    }
}
