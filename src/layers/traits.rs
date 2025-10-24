use crate::matrix::Matrix;

/// Core trait that all neural network layers must implement
pub trait Layer {
    fn forward(&mut self, input: &Matrix) -> Matrix;
    fn backward(&mut self, grad_output: &Matrix) -> Matrix;
    fn parameters(&mut self) -> Vec<&mut Matrix>;
    fn zero_grad(&mut self);
}
