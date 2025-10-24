//! Linear (fully connected) layer implementation
//!
//! A linear layer performs the transformation Y = XW + b where:
//! - X is the input
//! - W is the weight matrix
//! - b is the bias vector
//!
//! # Example
//! ```
//! use jwall_gpt::layers::{Linear, Layer};
//! use jwall_gpt::matrix::{Matrix, MatrixExt};
//!
//! let mut layer = Linear::new(128, 256);
//! let input = Matrix::random(32, 128, 0.1); // batch_size=32, features=128
//! let output = layer.forward(&input);  // [32, 256]
//! ```

use crate::layers::Layer;
use crate::matrix::{Matrix, MatrixExt, zeros_matrix};

/// Linear transformation layer: Y = XW + b
#[derive(Debug, Clone)]
pub struct Linear {
    /// Weight matrix [input_dim, output_dim]
    pub weights: Matrix,

    /// Bias vector [output_dim]
    pub bias: Matrix,

    /// Cached input for backward pass
    last_input: Option<Matrix>,

    /// Accumulated weight gradients
    pub weight_grad: Matrix,

    /// Accumulated bias gradients
    pub bias_grad: Matrix,
}

impl Linear {
    /// Create a new linear layer with Xavier initialization
    ///
    /// # Arguments
    /// * `input_dim` - Size of input features
    /// * `output_dim` - Size of output features
    ///
    /// # Example
    /// ```
    /// use jwall_gpt::layers::Linear;
    ///
    /// let layer = Linear::new(512, 2048);
    /// ```
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();

        Self {
            weights: MatrixExt::random(input_dim, output_dim, scale),
            bias: zeros_matrix(1, output_dim),
            last_input: None,
            weight_grad: zeros_matrix(input_dim, output_dim),
            bias_grad: zeros_matrix(1, output_dim),
        }
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        MatrixExt::rows(&self.weights)
    }

    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        MatrixExt::cols(&self.weights)
    }
}

impl Layer for Linear {
    fn forward(&mut self, input: &Matrix) -> Matrix {
        assert_eq!(
            MatrixExt::cols(input),
            self.input_dim(),
            "Input dimension mismatch: expected {}, got {}",
            self.input_dim(),
            MatrixExt::cols(input)
        );

        // Cache for backward pass
        self.last_input = Some(input.clone());

        // Y = XW + b
        let output = MatrixExt::matmul(input, &self.weights);
        MatrixExt::add_broadcast(&output, &self.bias)
    }

    fn backward(&mut self, grad_output: &Matrix) -> Matrix {
        let input = self
            .last_input
            .as_ref()
            .expect("forward must be called before backward");

        // Gradient w.r.t. input
        let grad_input = MatrixExt::matmul(grad_output, &MatrixExt::transpose(&self.weights));

        // Gradient w.r.t. weights
        self.weight_grad = MatrixExt::matmul(&MatrixExt::transpose(input), grad_output);

        // Gradient w.r.t. bias (sum over batch dimension)
        self.bias_grad = MatrixExt::sum_axis(grad_output, 0);

        grad_input
    }

    fn parameters(&mut self) -> Vec<&mut Matrix> {
        vec![&mut self.weights, &mut self.bias]
    }

    fn zero_grad(&mut self) {
        self.weight_grad.fill_value(0.0);
        self.bias_grad.fill_value(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::MatrixExt;

    #[test]
    fn test_forward_shape() {
        let mut layer = Linear::new(10, 5);
        let input: Matrix = MatrixExt::from_vec(vec![vec![1.0; 10]; 3]); // batch_size=3
        let output = layer.forward(&input);

        assert_eq!(MatrixExt::shape(&output), (3, 5));
    }

    #[test]
    fn test_backward_gradients() {
        let mut layer = Linear::new(2, 3);
        let input: Matrix = MatrixExt::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let _output = layer.forward(&input);
        let grad_output: Matrix = MatrixExt::from_vec(vec![vec![1.0; 3]; 2]);
        let grad_input = layer.backward(&grad_output);

        assert_eq!(MatrixExt::shape(&grad_input), (2, 2));
        assert_eq!(MatrixExt::shape(&layer.weight_grad), (2, 3));
        assert_eq!(MatrixExt::shape(&layer.bias_grad), (1, 3));
    }

    #[test]
    #[should_panic(expected = "Input dimension mismatch")]
    fn test_dimension_mismatch() {
        let mut layer = Linear::new(10, 5);
        let input: Matrix = MatrixExt::from_vec(vec![vec![1.0; 8]; 3]); // Wrong input dimension!
        layer.forward(&input);
    }
}
