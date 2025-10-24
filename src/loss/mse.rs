//! Mean Squared Error loss
//!
//! Simple loss for regression tasks and testing. Not used for GPT but useful
//! for validating your training pipeline.

use crate::loss::Loss;
use crate::matrix::{Matrix, MatrixExt, zeros_matrix};

/// Mean Squared Error Loss
///
/// Measures the average squared difference between predictions and targets.
///
/// # Formula
///
/// ```text
/// MSE = (1/N) * Σ(prediction - target)²
/// ```
///
/// # Example
///
/// ```
/// use jwall_gpt::loss::{MSELoss, Loss};
/// use jwall_gpt::matrix::{Matrix, MatrixExt};
///
/// let loss_fn = MSELoss::new();
///
/// let predictions = Matrix::from_vec(vec![vec![1.0, 2.0, 3.0]]);
/// let targets = Matrix::from_vec(vec![vec![1.5, 2.5, 2.5]]);
///
/// let loss = loss_fn.forward(&predictions, &targets);
/// ```
pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        Self
    }
}

impl Loss for MSELoss {
    /// Compute mean squared error
    fn forward(&self, predictions: &Matrix, targets: &Matrix) -> f32 {
        assert_eq!(
            MatrixExt::shape(predictions),
            MatrixExt::shape(targets),
            "Predictions and targets must have same shape"
        );

        let (rows, cols) = MatrixExt::shape(predictions);
        let n = (rows * cols) as f32;

        let mut sum_squared_error = 0.0;
        for i in 0..rows {
            for j in 0..cols {
                let diff = predictions[[i, j]] - targets[[i, j]];
                sum_squared_error += diff * diff;
            }
        }

        sum_squared_error / n
    }

    /// Compute gradient: 2(prediction - target) / N
    fn backward(&self, predictions: &Matrix, targets: &Matrix) -> Matrix {
        assert_eq!(MatrixExt::shape(predictions), MatrixExt::shape(targets));

        let (rows, cols) = MatrixExt::shape(predictions);
        let n = (rows * cols) as f32;
        let scale = 2.0 / n;

        let mut grad = zeros_matrix(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                grad[[i, j]] = scale * (predictions[[i, j]] - targets[[i, j]]);
            }
        }

        grad
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_zero_loss() {
        let loss_fn = MSELoss::new();

        let predictions: Matrix = MatrixExt::from_vec(vec![vec![1.0, 2.0, 3.0]]);
        let targets = predictions.clone();

        let loss = loss_fn.forward(&predictions, &targets);

        assert!(loss.abs() < 1e-6, "MSE should be 0 for identical inputs");
    }

    #[test]
    fn test_mse_known_value() {
        let loss_fn = MSELoss::new();

        let predictions = MatrixExt::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let targets = MatrixExt::from_vec(vec![vec![0.0, 0.0], vec![0.0, 0.0]]);

        let loss = loss_fn.forward(&predictions, &targets);

        // MSE = (1² + 2² + 3² + 4²) / 4 = 30/4 = 7.5
        assert!((loss - 7.5).abs() < 1e-5, "MSE: {}", loss);
    }

    #[test]
    fn test_mse_gradient() {
        let loss_fn = MSELoss::new();

        let predictions = MatrixExt::from_vec(vec![vec![2.0, 4.0]]);
        let targets = MatrixExt::from_vec(vec![vec![1.0, 2.0]]);

        let grad = loss_fn.backward(&predictions, &targets);

        // Gradient = 2(pred - target) / N
        // = 2([2-1, 4-2]) / 2 = [1.0, 2.0]
        assert!((grad[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((grad[[0, 1]] - 2.0).abs() < 1e-5);
    }
}
