//! Adam (Adaptive Moment Estimation) optimizer
//!
//! Adam combines the benefits of momentum (first moment) and RMSprop (second moment)
//! to achieve fast convergence with adaptive learning rates per parameter.
//!
//! This is the optimizer used to train GPT models.

use crate::matrix::{Matrix, MatrixExt, zeros_matrix};
use crate::optimizers::Optimizer;

/// Adam optimizer
///
/// Adapts learning rates for each parameter based on estimates of first and
/// second moments of the gradients.
///
/// # Algorithm
///
/// ```text
/// m_t = β₁ * m_{t-1} + (1 - β₁) * g_t          (first moment)
/// v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²         (second moment)
/// m̂_t = m_t / (1 - β₁ᵗ)                        (bias correction)
/// v̂_t = v_t / (1 - β₂ᵗ)                        (bias correction)
/// θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)        (parameter update)
/// ```
///
/// # Hyperparameters
///
/// - `learning_rate` (α): Step size (default: 0.001)
/// - `beta1` (β₁): Exponential decay for first moment (default: 0.9)
/// - `beta2` (β₂): Exponential decay for second moment (default: 0.999)
/// - `epsilon` (ε): Small constant for numerical stability (default: 1e-8)
///
/// # Example
///
/// ```
/// use jwall_gpt::optimizers::{Adam, Optimizer};
/// use jwall_gpt::matrix::{Matrix, MatrixExt};
///
/// let mut optimizer = Adam::new(0.001);
///
/// let mut weights = Matrix::random(10, 10, 0.1);
/// let gradients = Matrix::random(10, 10, 0.01);
///
/// optimizer.step(&mut [&mut weights], &[&gradients]);
/// ```
#[derive(Debug, Clone)]
pub struct Adam {
    /// Learning rate (α)
    pub learning_rate: f32,

    /// Exponential decay rate for first moment estimates (β₁)
    pub beta1: f32,

    /// Exponential decay rate for second moment estimates (β₂)
    pub beta2: f32,

    /// Small constant for numerical stability (ε)
    pub epsilon: f32,

    /// First moment estimates (mean of gradients)
    momentum: Vec<Matrix>,

    /// Second moment estimates (variance of gradients)
    velocity: Vec<Matrix>,

    /// Current timestep (for bias correction)
    timestep: usize,
}

impl Adam {
    /// Create a new Adam optimizer with default hyperparameters
    ///
    /// Default values match the original paper and work well for most cases:
    /// - learning_rate: 0.001
    /// - beta1: 0.9
    /// - beta2: 0.999
    /// - epsilon: 1e-8
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            momentum: Vec::new(),
            velocity: Vec::new(),
            timestep: 0,
        }
    }

    /// Create Adam optimizer with custom hyperparameters
    ///
    /// # Arguments
    /// * `learning_rate` - Step size
    /// * `beta1` - Decay rate for first moment (typically 0.9)
    /// * `beta2` - Decay rate for second moment (typically 0.999)
    /// * `epsilon` - Numerical stability constant (typically 1e-8)
    pub fn with_params(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            momentum: Vec::new(),
            velocity: Vec::new(),
            timestep: 0,
        }
    }

    /// Get current timestep (number of updates performed)
    pub fn timestep(&self) -> usize {
        self.timestep
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [&mut Matrix], grads: &[&Matrix]) {
        assert_eq!(
            params.len(),
            grads.len(),
            "Number of parameters must match number of gradients"
        );

        // Initialize momentum and velocity on first call
        if self.momentum.is_empty() {
            self.momentum = grads
                .iter()
                .map(|g| {
                    let (r, c) = MatrixExt::shape(*g);
                    zeros_matrix(r, c)
                })
                .collect();
            self.velocity = grads
                .iter()
                .map(|g| {
                    let (r, c) = MatrixExt::shape(*g);
                    zeros_matrix(r, c)
                })
                .collect();
        }

        // Increment timestep
        self.timestep += 1;

        // Precompute bias correction terms
        let bias_correction1 = 1.0 - self.beta1.powi(self.timestep as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.timestep as i32);

        // Update each parameter
        for i in 0..params.len() {
            let (rows, cols) = MatrixExt::shape(grads[i]);

            for row in 0..rows {
                for col in 0..cols {
                    let grad = grads[i][[row, col]];

                    // Update biased first moment estimate
                    // m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
                    self.momentum[i][[row, col]] =
                        self.beta1 * self.momentum[i][[row, col]] + (1.0 - self.beta1) * grad;

                    // Update biased second moment estimate
                    // v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
                    self.velocity[i][[row, col]] = self.beta2 * self.velocity[i][[row, col]]
                        + (1.0 - self.beta2) * grad * grad;

                    // Compute bias-corrected first moment
                    // m̂_t = m_t / (1 - β₁ᵗ)
                    let m_hat = self.momentum[i][[row, col]] / bias_correction1;

                    // Compute bias-corrected second moment
                    // v̂_t = v_t / (1 - β₂ᵗ)
                    let v_hat = self.velocity[i][[row, col]] / bias_correction2;

                    // Update parameter
                    // θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
                    params[i][[row, col]] -=
                        self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
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
    fn test_adam_initialization() {
        let optimizer = Adam::new(0.001);

        assert_eq!(optimizer.learning_rate, 0.001);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);
        assert_eq!(optimizer.epsilon, 1e-8);
        assert_eq!(optimizer.timestep, 0);
    }

    #[test]
    fn test_adam_step() {
        let mut optimizer = Adam::new(0.1);

        let mut param: Matrix = MatrixExt::from_vec(vec![vec![1.0]]);
        let grad: Matrix = MatrixExt::from_vec(vec![vec![0.1]]);

        let initial_value = param[[0, 0]];
        optimizer.step(&mut [&mut param], &[&grad]);

        // Parameter should have moved (decreased since gradient is positive)
        assert!(param[[0, 0]] < initial_value);

        // Timestep should have incremented
        assert_eq!(optimizer.timestep, 1);
    }

    #[test]
    fn test_adam_multiple_steps() {
        let mut optimizer = Adam::new(0.1);

        let mut param: Matrix = MatrixExt::from_vec(vec![vec![1.0]]);
        let grad: Matrix = MatrixExt::from_vec(vec![vec![1.0]]);

        // Take multiple steps
        for _ in 0..10 {
            optimizer.step(&mut [&mut param], &[&grad]);
        }

        // Parameter should have decreased significantly
        assert!(param[[0, 0]] < 0.5);
        assert_eq!(optimizer.timestep, 10);
    }

    #[test]
    fn test_adam_converges_to_minimum() {
        let mut optimizer = Adam::new(0.1);

        // Optimize f(x) = x² (minimum at x=0)
        let mut param: Matrix = MatrixExt::from_vec(vec![vec![5.0]]);

        for _ in 0..100 {
            // Gradient of x² is 2x
            let x = param[[0, 0]];
            let grad: Matrix = MatrixExt::from_vec(vec![vec![2.0 * x]]);
            optimizer.step(&mut [&mut param], &[&grad]);
        }

        // Should converge close to 0
        assert!(param[[0, 0]].abs() < 0.1, "Final value: {}", param[[0, 0]]);
    }
}
