//! Cross-Entropy Loss for classification and language modeling
//!
//! This is the primary loss function for GPT models. It measures the difference
//! between predicted probability distributions and actual token distributions.
//!
//! # Numerical Stability
//!
//! We use the log-sum-exp trick to avoid numerical overflow/underflow when
//! computing softmax and log probabilities.

use crate::loss::Loss;
use crate::matrix::{Matrix, MatrixExt, zeros_matrix};

/// Cross-Entropy Loss with automatic softmax
///
/// Expects raw logits (unnormalized scores) as input.
/// Applies softmax internally for numerical stability.
///
/// # Formula
///
/// ```text
/// CE = -Σ y_true * log(y_pred)
///
/// For language modeling (one-hot targets):
/// CE = -log(p[correct_class])
/// ```
///
/// # Example
///
/// ```
/// use jwall_gpt::loss::{CrossEntropyLoss, Loss};
/// use jwall_gpt::matrix::{Matrix, MatrixExt};
///
/// let loss_fn = CrossEntropyLoss::new();
///
/// // Logits: [batch_size=2, num_classes=3]
/// let logits = Matrix::from_vec(vec![
///     vec![2.0, 1.0, 0.1],
///     vec![0.5, 2.5, 0.3],
/// ]);
///
/// // Targets: one-hot encoded
/// let targets = Matrix::from_vec(vec![
///     vec![1.0, 0.0, 0.0],  // Class 0
///     vec![0.0, 1.0, 0.0],  // Class 1
/// ]);
///
/// let loss = loss_fn.forward(&logits, &targets);
/// ```
pub struct CrossEntropyLoss {}

impl CrossEntropyLoss {
    /// Create a new cross-entropy loss function
    pub fn new() -> Self {
        Self {}
    }

    /// Compute softmax with numerical stability (log-sum-exp trick)
    ///
    /// # Arguments
    /// * `logits` - Raw scores [batch_size, num_classes]
    ///
    /// # Returns
    /// Probabilities [batch_size, num_classes] that sum to 1.0 per row
    fn softmax(&self, logits: &Matrix) -> Matrix {
        let (batch_size, num_classes) = MatrixExt::shape(logits);
        let mut probs = zeros_matrix(batch_size, num_classes);

        // Process each example in the batch
        for i in 0..batch_size {
            // Find max for numerical stability (prevents overflow in exp)
            let mut max_logit = logits[[i, 0]];
            for j in 1..num_classes {
                max_logit = max_logit.max(logits[[i, j]]);
            }

            // Compute exp(logit - max) and sum
            let mut sum_exp = 0.0;
            for j in 0..num_classes {
                let exp_val = (logits[[i, j]] - max_logit).exp();
                probs[[i, j]] = exp_val;
                sum_exp += exp_val;
            }

            // Normalize to get probabilities
            for j in 0..num_classes {
                probs[[i, j]] /= sum_exp;
            }
        }

        probs
    }

    /// Compute log-softmax with numerical stability
    ///
    /// More efficient than softmax + log for the forward pass
    fn log_softmax(&self, logits: &Matrix) -> Matrix {
        let (batch_size, num_classes) = MatrixExt::shape(logits);
        let mut log_probs = zeros_matrix(batch_size, num_classes);

        for i in 0..batch_size {
            // Find max
            let mut max_logit = logits[[i, 0]];
            for j in 1..num_classes {
                max_logit = max_logit.max(logits[[i, j]]);
            }

            // Compute log-sum-exp
            let mut sum_exp: f32 = 0.0;
            for j in 0..num_classes {
                sum_exp += (logits[[i, j]] - max_logit).exp();
            }
            let log_sum_exp = max_logit + sum_exp.ln();

            // log_softmax = logit - log_sum_exp
            for j in 0..num_classes {
                log_probs[[i, j]] = logits[[i, j]] - log_sum_exp;
            }
        }

        log_probs
    }
}

impl Loss for CrossEntropyLoss {
    /// Compute cross-entropy loss
    ///
    /// # Arguments
    /// * `logits` - Raw model outputs [batch_size, num_classes]
    /// * `targets` - One-hot encoded targets [batch_size, num_classes]
    ///
    /// # Returns
    /// Average loss over the batch
    fn forward(&self, logits: &Matrix, targets: &Matrix) -> f32 {
        assert_eq!(
            MatrixExt::shape(logits),
            MatrixExt::shape(targets),
            "Logits and targets must have same shape"
        );

        let (batch_size, num_classes) = MatrixExt::shape(logits);

        // Compute log probabilities (numerically stable)
        let log_probs = self.log_softmax(logits);

        // Compute -Σ(target * log_prob) for each example
        let mut total_loss = 0.0;
        for i in 0..batch_size {
            for j in 0..num_classes {
                if targets[[i, j]] > 0.0 {
                    // Only non-zero targets contribute to loss
                    total_loss -= targets[[i, j]] * log_probs[[i, j]];
                }
            }
        }

        // Return average loss
        total_loss / batch_size as f32
    }

    /// Compute gradient: softmax(logits) - targets
    ///
    /// # Mathematical Derivation
    ///
    /// For cross-entropy with softmax:
    /// ```text
    /// ∂L/∂logits = softmax(logits) - targets
    /// ```
    ///
    /// This beautiful result makes backprop very efficient!
    ///
    /// # Arguments
    /// * `logits` - Raw model outputs [batch_size, num_classes]
    /// * `targets` - One-hot encoded targets [batch_size, num_classes]
    ///
    /// # Returns
    /// Gradient w.r.t. logits [batch_size, num_classes]
    fn backward(&self, logits: &Matrix, targets: &Matrix) -> Matrix {
        assert_eq!(MatrixExt::shape(logits), MatrixExt::shape(targets));

        let (batch_size, _) = MatrixExt::shape(logits);

        // Compute softmax probabilities
        let probs = self.softmax(logits);

        // Gradient is: (probs - targets) / batch_size
        let (rows, cols) = MatrixExt::shape(&probs);
        let mut grad = zeros_matrix(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                grad[[i, j]] = (probs[[i, j]] - targets[[i, j]]) / batch_size as f32;
            }
        }

        grad
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to convert class indices to one-hot encoding
///
/// # Example
/// ```
/// use jwall_gpt::loss::indices_to_one_hot;
///
/// let indices = vec![0, 2, 1];  // 3 examples
/// let one_hot = indices_to_one_hot(&indices, 3);
/// // Returns:
/// // [[1, 0, 0],
/// //  [0, 0, 1],
/// //  [0, 1, 0]]
/// ```
pub fn indices_to_one_hot(indices: &[usize], num_classes: usize) -> Matrix {
    let batch_size = indices.len();
    let mut one_hot = zeros_matrix(batch_size, num_classes);

    for (i, &class_idx) in indices.iter().enumerate() {
        assert!(
            class_idx < num_classes,
            "Class index {} out of bounds for {} classes",
            class_idx,
            num_classes
        );
        one_hot[[i, class_idx]] = 1.0;
    }

    one_hot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_sums_to_one() {
        let loss = CrossEntropyLoss::new();
        let logits = Matrix::from_vec(vec![vec![1.0, 2.0, 3.0], vec![0.1, 0.2, 0.3]]);

        let probs = loss.softmax(&logits);

        // Each row should sum to ~1.0
        for i in 0..2 {
            let row_sum: f32 = (0..3).map(|j| probs[[i, j]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-5, "Row {} sum: {}", i, row_sum);
        }
    }

    #[test]
    fn test_cross_entropy_perfect_prediction() {
        let loss_fn = CrossEntropyLoss::new();

        // Perfect prediction: very high logit for correct class
        let logits = Matrix::from_vec(vec![
            vec![10.0, 0.0, 0.0], // Predicts class 0
        ]);
        let targets = Matrix::from_vec(vec![
            vec![1.0, 0.0, 0.0], // True class 0
        ]);

        let loss = loss_fn.forward(&logits, &targets);

        // Loss should be very small for perfect prediction
        assert!(loss < 0.1, "Loss for perfect prediction: {}", loss);
    }

    #[test]
    fn test_cross_entropy_wrong_prediction() {
        let loss_fn = CrossEntropyLoss::new();

        // Wrong prediction: high logit for wrong class
        let logits = Matrix::from_vec(vec![
            vec![0.0, 0.0, 10.0], // Predicts class 2
        ]);
        let targets = Matrix::from_vec(vec![
            vec![1.0, 0.0, 0.0], // True class 0
        ]);

        let loss = loss_fn.forward(&logits, &targets);

        // Loss should be high for wrong prediction
        assert!(loss > 5.0, "Loss for wrong prediction: {}", loss);
    }

    #[test]
    fn test_gradient_shape() {
        let loss_fn = CrossEntropyLoss::new();

        let logits = Matrix::from_vec(vec![vec![1.0, 2.0, 3.0], vec![0.1, 0.2, 0.3]]);
        let targets = Matrix::from_vec(vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);

        let grad = loss_fn.backward(&logits, &targets);

        assert_eq!(grad.shape(), logits.shape());
    }

    #[test]
    fn test_indices_to_one_hot() {
        let indices = vec![0, 2, 1];
        let one_hot = indices_to_one_hot(&indices, 3);

        assert_eq!(one_hot[[0, 0]], 1.0);
        assert_eq!(one_hot[[0, 1]], 0.0);
        assert_eq!(one_hot[[1, 2]], 1.0);
        assert_eq!(one_hot[[2, 1]], 1.0);
    }
}
