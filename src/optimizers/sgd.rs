//! Stochastic Gradient Descent (SGD) optimizer
//!
//! The simplest optimizer - just moves parameters in the direction opposite
//! to the gradient, scaled by the learning rate.

use crate::matrix::{Matrix, MatrixExt, zeros_matrix};
use crate::optimizers::Optimizer;

/// Stochastic Gradient Descent optimizer
///
/// Updates parameters using the rule:
/// ```text
/// θ = θ - lr * ∇θ
/// ```
///
/// With optional momentum:
/// ```text
/// v = momentum * v + ∇θ
/// θ = θ - lr * v
/// ```
///
/// # Example
///
/// ```
/// use jwall_gpt::optimizers::{SGD, Optimizer};
/// use jwall_gpt::matrix::{Matrix, MatrixExt};
///
/// let mut optimizer = SGD::new(0.01);
///
/// let mut weights = Matrix::random(10, 10, 0.1);
/// let gradients = Matrix::random(10, 10, 0.01);
///
/// optimizer.step(&mut [&mut weights], &[&gradients]);
/// ```
#[derive(Debug, Clone)]
pub struct SGD {
    /// Learning rate
    pub learning_rate: f32,

    /// Momentum coefficient (0.0 = no momentum, 0.9 = high momentum)
    pub momentum: f32,

    /// Velocity buffers for momentum (one per parameter)
    velocities: Vec<Matrix>,
}

impl SGD {
    /// Create a new SGD optimizer with no momentum
    ///
    /// # Arguments
    /// * `learning_rate` - Step size for parameter updates
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            velocities: Vec::new(),
        }
    }

    /// Create SGD optimizer with momentum
    ///
    /// # Arguments
    /// * `learning_rate` - Step size for parameter updates
    /// * `momentum` - Momentum coefficient (typically 0.9)
    ///
    /// Momentum helps accelerate convergence and smooth out updates
    pub fn with_momentum(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            velocities: Vec::new(),
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut [&mut Matrix], grads: &[&Matrix]) {
        assert_eq!(
            params.len(),
            grads.len(),
            "Number of parameters must match number of gradients"
        );

        // Initialize velocities on first call if using momentum
        if self.momentum > 0.0 && self.velocities.is_empty() {
            self.velocities = grads
                .iter()
                .map(|g| {
                    let (r, c) = MatrixExt::shape(*g);
                    zeros_matrix(r, c)
                })
                .collect();
        }

        if self.momentum > 0.0 {
            // SGD with momentum
            for i in 0..params.len() {
                let (rows, cols) = MatrixExt::shape(grads[i]);

                // Update velocity: v = momentum * v + grad
                for row in 0..rows {
                    for col in 0..cols {
                        self.velocities[i][[row, col]] =
                            self.momentum * self.velocities[i][[row, col]] + grads[i][[row, col]];
                    }
                }

                // Update parameters: θ = θ - lr * v
                for row in 0..rows {
                    for col in 0..cols {
                        params[i][[row, col]] -=
                            self.learning_rate * self.velocities[i][[row, col]];
                    }
                }
            }
        } else {
            // Plain SGD without momentum
            for i in 0..params.len() {
                let (rows, cols) = MatrixExt::shape(grads[i]);

                // Update parameters: θ = θ - lr * grad
                for row in 0..rows {
                    for col in 0..cols {
                        params[i][[row, col]] -= self.learning_rate * grads[i][[row, col]];
                    }
                }
            }
        }
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::MatrixExt;

    #[test]
    fn test_sgd_basic_step() {
        let mut optimizer = SGD::new(0.1);

        let mut param: Matrix = MatrixExt::from_vec(vec![vec![1.0, 2.0, 3.0]]);
        let grad: Matrix = MatrixExt::from_vec(vec![vec![0.1, 0.2, 0.3]]);

        optimizer.step(&mut [&mut param], &[&grad]);

        // After update: param = param - 0.1 * grad
        assert!((param[[0, 0]] - 0.99).abs() < 1e-5); // 1.0 - 0.1*0.1
        assert!((param[[0, 1]] - 1.98).abs() < 1e-5); // 2.0 - 0.1*0.2
        assert!((param[[0, 2]] - 2.97).abs() < 1e-5); // 3.0 - 0.1*0.3
    }

    #[test]
    fn test_sgd_with_momentum() {
        let mut optimizer = SGD::with_momentum(0.1, 0.9);

        let mut param: Matrix = MatrixExt::from_vec(vec![vec![1.0]]);
        let grad: Matrix = MatrixExt::from_vec(vec![vec![1.0]]);

        // First step: velocity = 0 + 1.0 = 1.0
        optimizer.step(&mut [&mut param], &[&grad]);
        assert!((param[[0, 0]] - 0.9).abs() < 1e-5); // 1.0 - 0.1*1.0

        // Second step: velocity = 0.9*1.0 + 1.0 = 1.9
        optimizer.step(&mut [&mut param], &[&grad]);
        assert!((param[[0, 0]] - 0.71).abs() < 1e-5); // 0.9 - 0.1*1.9
    }
}
