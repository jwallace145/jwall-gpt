//! Common trait for all loss functions

use crate::matrix::Matrix;

/// Trait that all loss functions must implement
pub trait Loss {
    /// Compute the loss value
    ///
    /// # Arguments
    /// * `predictions` - Model outputs
    /// * `targets` - Ground truth
    ///
    /// # Returns
    /// Scalar loss value
    fn forward(&self, predictions: &Matrix, targets: &Matrix) -> f32;

    /// Compute gradient of loss w.r.t. predictions
    ///
    /// # Arguments
    /// * `predictions` - Model outputs
    /// * `targets` - Ground truth
    ///
    /// # Returns
    /// Gradient matrix same shape as predictions
    fn backward(&self, predictions: &Matrix, targets: &Matrix) -> Matrix;
}
