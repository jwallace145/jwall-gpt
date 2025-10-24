//! Loss functions for training neural networks
//!
//! This module provides various loss functions used in training.
//! For GPT models, [`CrossEntropyLoss`] is the primary loss function.

mod cross_entropy;
mod mse;
mod traits;

pub use cross_entropy::{CrossEntropyLoss, indices_to_one_hot};
pub use mse::MSELoss;
pub use traits::Loss;
