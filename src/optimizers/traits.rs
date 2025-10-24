use crate::matrix::Matrix;

/// Core trait that all optimizers must implement
pub trait Optimizer {
    /// Perform one optimization step
    ///
    /// # Arguments
    /// * `params` - Mutable references to parameters to update
    /// * `grads` - Gradients for each parameter
    fn step(&mut self, params: &mut [&mut Matrix], grads: &[&Matrix]);

    /// Get the current learning rate
    fn learning_rate(&self) -> f32;

    /// Set a new learning rate
    fn set_learning_rate(&mut self, lr: f32);
}
