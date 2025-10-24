//! Optimizers for training neural networks
//!
//! This module provides various optimization algorithms for updating model parameters
//! during training. For GPT models, [`Adam`] is the recommended optimizer.
//!
//! # Example
//!
//! ```
//! use jwall_gpt::optimizers::{Adam, Optimizer};
//! use jwall_gpt::matrix::{Matrix, MatrixExt};
//!
//! // Create optimizer
//! let mut optimizer = Adam::new(0.001);
//!
//! // During training:
//! let mut weights = Matrix::random(100, 100, 0.1);
//! let gradients = Matrix::random(100, 100, 0.01);
//!
//! optimizer.step(&mut [&mut weights], &[&gradients]);
//! ```

mod adam;
mod sgd;
mod traits;

pub use adam::Adam;
pub use sgd::SGD;
pub use traits::Optimizer;
