//! Dataset Loading and Management
//!
//! This module provides a unified interface for loading and managing datasets
//! for training neural networks. It includes a trait-based design that allows
//! for easy integration of different datasets.
//!
//! # Examples
//!
//! ```rust,no_run
//! use jwall_gpt::datasets::{Dataset, MnistDataset};
//! use jwall_gpt::matrix::MatrixExt;
//!
//! // Load MNIST dataset (will download if needed)
//! let mnist = MnistDataset::new().expect("Failed to load MNIST");
//!
//! // Get training data
//! let (train_images, train_labels) = mnist.train_data();
//! println!("Training samples: {}", MatrixExt::rows(train_images));
//!
//! // Get test data
//! let (test_images, test_labels) = mnist.test_data();
//! println!("Test samples: {}", MatrixExt::rows(test_images));
//! ```

pub mod mnist;

pub use mnist::MnistDataset;

use crate::matrix::{Matrix, MatrixExt};

/// Trait for dataset providers
///
/// This trait defines a common interface for all datasets used in training
/// and evaluation. Datasets should provide both training and test splits,
/// with features (input data) and labels (target outputs).
///
/// # Type Parameters
///
/// Implementations should work with the matrix backend configured for the crate
/// (either BLAS-based or naive, controlled by feature flags).
///
/// # Examples
///
/// ```rust
/// use jwall_gpt::datasets::Dataset;
/// use jwall_gpt::matrix::Matrix;
///
/// struct MyDataset {
///     train_x: Matrix,
///     train_y: Matrix,
///     test_x: Matrix,
///     test_y: Matrix,
/// }
///
/// impl Dataset for MyDataset {
///     fn train_data(&self) -> (&Matrix, &Matrix) {
///         (&self.train_x, &self.train_y)
///     }
///
///     fn test_data(&self) -> (&Matrix, &Matrix) {
///         (&self.test_x, &self.test_y)
///     }
///
///     fn num_classes(&self) -> usize {
///         10
///     }
///
///     fn input_size(&self) -> usize {
///         784
///     }
/// }
/// ```
pub trait Dataset {
    /// Returns the training data as (features, labels)
    ///
    /// # Returns
    ///
    /// A tuple of references to:
    /// - Features matrix: [num_samples, input_size]
    /// - Labels matrix: [num_samples, num_classes] (one-hot encoded)
    fn train_data(&self) -> (&Matrix, &Matrix);

    /// Returns the test data as (features, labels)
    ///
    /// # Returns
    ///
    /// A tuple of references to:
    /// - Features matrix: [num_samples, input_size]
    /// - Labels matrix: [num_samples, num_classes] (one-hot encoded)
    fn test_data(&self) -> (&Matrix, &Matrix);

    /// Returns the number of classes in the dataset
    fn num_classes(&self) -> usize;

    /// Returns the size of the input features
    fn input_size(&self) -> usize;

    /// Returns the number of training samples
    fn train_size(&self) -> usize {
        let (features, _) = self.train_data();
        MatrixExt::rows(features)
    }

    /// Returns the number of test samples
    fn test_size(&self) -> usize {
        let (features, _) = self.test_data();
        MatrixExt::rows(features)
    }
}

/// Helper function to normalize image data to [0, 1] range
///
/// # Arguments
///
/// * `data` - Raw pixel data (typically 0-255)
///
/// # Returns
///
/// Normalized data in the range [0, 1]
pub fn normalize_images(data: &[u8]) -> Vec<f32> {
    data.iter().map(|&pixel| pixel as f32 / 255.0).collect()
}

/// Helper function to convert class labels to one-hot encoded format
///
/// # Arguments
///
/// * `labels` - Array of class indices
/// * `num_classes` - Total number of classes
///
/// # Returns
///
/// One-hot encoded labels as a flat vector (row-major order)
///
/// # Examples
///
/// ```
/// use jwall_gpt::datasets::labels_to_one_hot;
///
/// let labels = vec![0, 2, 1];
/// let one_hot = labels_to_one_hot(&labels, 3);
/// // Results in: [1,0,0, 0,0,1, 0,1,0]
/// assert_eq!(one_hot.len(), 9); // 3 samples * 3 classes
/// ```
pub fn labels_to_one_hot(labels: &[u8], num_classes: usize) -> Vec<f32> {
    let mut one_hot = vec![0.0; labels.len() * num_classes];

    for (i, &label) in labels.iter().enumerate() {
        let class_idx = label as usize;
        assert!(
            class_idx < num_classes,
            "Label {} out of bounds for {} classes",
            class_idx,
            num_classes
        );
        one_hot[i * num_classes + class_idx] = 1.0;
    }

    one_hot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_images() {
        let raw = vec![0, 127, 255];
        let normalized = normalize_images(&raw);

        assert!((normalized[0] - 0.0).abs() < 1e-6);
        assert!((normalized[1] - 0.498).abs() < 0.01);
        assert!((normalized[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_labels_to_one_hot() {
        let labels = vec![0, 2, 1];
        let one_hot = labels_to_one_hot(&labels, 3);

        // First sample: class 0
        assert_eq!(one_hot[0], 1.0);
        assert_eq!(one_hot[1], 0.0);
        assert_eq!(one_hot[2], 0.0);

        // Second sample: class 2
        assert_eq!(one_hot[3], 0.0);
        assert_eq!(one_hot[4], 0.0);
        assert_eq!(one_hot[5], 1.0);

        // Third sample: class 1
        assert_eq!(one_hot[6], 0.0);
        assert_eq!(one_hot[7], 1.0);
        assert_eq!(one_hot[8], 0.0);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_labels_to_one_hot_out_of_bounds() {
        let labels = vec![0, 5, 1]; // 5 is out of bounds for 3 classes
        labels_to_one_hot(&labels, 3);
    }
}
