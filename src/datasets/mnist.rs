#![allow(dead_code)]

//! MNIST Handwritten Digits Dataset
//!
//! This module provides an implementation of the Dataset trait for the MNIST
//! handwritten digits dataset. MNIST contains 70,000 grayscale images of
//! handwritten digits (0-9), with 60,000 training images and 10,000 test images.
//!
//! Each image is 28x28 pixels, flattened into a 784-dimensional vector.
//!
//! # Examples
//!
//! ```rust,no_run
//! use jwall_gpt::datasets::{Dataset, MnistDataset};
//! use jwall_gpt::matrix::MatrixExt;
//!
//! // Load the MNIST dataset (will download if not cached)
//! let mnist = MnistDataset::new().expect("Failed to load MNIST");
//!
//! println!("Training samples: {}", mnist.train_size());
//! println!("Test samples: {}", mnist.test_size());
//! println!("Input size: {}", mnist.input_size());
//! println!("Number of classes: {}", mnist.num_classes());
//!
//! // Get training data
//! let (train_images, train_labels) = mnist.train_data();
//! assert_eq!(MatrixExt::rows(train_images), 60000);
//! assert_eq!(MatrixExt::cols(train_images), 784);
//! assert_eq!(MatrixExt::rows(train_labels), 60000);
//! assert_eq!(MatrixExt::cols(train_labels), 10);
//! ```

use super::{Dataset, normalize_images};
use crate::matrix::{Matrix, MatrixExt};
use std::io;

/// MNIST handwritten digits dataset
///
/// Contains 60,000 training images and 10,000 test images of handwritten digits (0-9).
/// Each image is 28x28 pixels (784 features when flattened).
///
/// The data is automatically normalized to the range [0, 1] and labels are
/// one-hot encoded.
pub struct MnistDataset {
    train_images: Matrix,
    train_labels: Matrix,
    test_images: Matrix,
    test_labels: Matrix,
}

impl MnistDataset {
    /// Create a new MNIST dataset instance
    ///
    /// This will download the MNIST dataset if it's not already cached locally.
    /// The dataset will be downloaded to the current directory's `data/` folder.
    ///
    /// # Errors
    ///
    /// Returns an error if the dataset cannot be loaded (e.g., network issues,
    /// disk space problems, or corrupted files).
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use jwall_gpt::datasets::MnistDataset;
    ///
    /// let mnist = MnistDataset::new().expect("Failed to load MNIST");
    /// ```
    pub fn new() -> io::Result<Self> {
        // Load MNIST dataset using the mnist crate
        // This will automatically download the dataset if not present
        let mnist::Mnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
            ..
        } = mnist::MnistBuilder::new()
            .label_format_one_hot()
            .training_set_length(60_000)
            .validation_set_length(0)
            .test_set_length(10_000)
            .download_and_extract()
            .finalize();

        // Normalize images to [0, 1]
        let train_images_normalized = normalize_images(&trn_img);
        let test_images_normalized = normalize_images(&tst_img);

        // Convert labels from one-hot (already provided by mnist crate) to our format
        let train_labels_f32: Vec<f32> = trn_lbl.iter().map(|&x| x as f32).collect();
        let test_labels_f32: Vec<f32> = tst_lbl.iter().map(|&x| x as f32).collect();

        // Create matrices
        // Training: 60000 samples, 784 features
        let train_images = Self::create_matrix(train_images_normalized, 60_000, 784);
        // Training labels: 60000 samples, 10 classes
        let train_labels = Self::create_matrix(train_labels_f32, 60_000, 10);

        // Test: 10000 samples, 784 features
        let test_images = Self::create_matrix(test_images_normalized, 10_000, 784);
        // Test labels: 10000 samples, 10 classes
        let test_labels = Self::create_matrix(test_labels_f32, 10_000, 10);

        Ok(MnistDataset {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }

    /// Helper function to create a matrix from flat data
    fn create_matrix(data: Vec<f32>, rows: usize, cols: usize) -> Matrix {
        assert_eq!(
            data.len(),
            rows * cols,
            "Data length must match rows * cols"
        );

        let matrix_data: Vec<Vec<f32>> = data.chunks(cols).map(|chunk| chunk.to_vec()).collect();

        MatrixExt::from_vec(matrix_data)
    }

    /// Get a subset of training data for quick experimentation
    ///
    /// # Arguments
    ///
    /// * `num_samples` - Number of samples to return (must be <= 60,000)
    ///
    /// # Returns
    ///
    /// A tuple of (images, labels) matrices with the requested number of samples
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use jwall_gpt::datasets::MnistDataset;
    /// use jwall_gpt::matrix::MatrixExt;
    ///
    /// let mnist = MnistDataset::new().expect("Failed to load MNIST");
    /// let (small_train_x, small_train_y) = mnist.train_subset(1000);
    /// assert_eq!(MatrixExt::rows(&small_train_x), 1000);
    /// ```
    pub fn train_subset(&self, num_samples: usize) -> (Matrix, Matrix) {
        assert!(
            num_samples <= 60_000,
            "num_samples must be <= 60,000 for MNIST training data"
        );

        let images = self.get_subset(&self.train_images, num_samples);
        let labels = self.get_subset(&self.train_labels, num_samples);

        (images, labels)
    }

    /// Get a subset of test data
    ///
    /// # Arguments
    ///
    /// * `num_samples` - Number of samples to return (must be <= 10,000)
    pub fn test_subset(&self, num_samples: usize) -> (Matrix, Matrix) {
        assert!(
            num_samples <= 10_000,
            "num_samples must be <= 10,000 for MNIST test data"
        );

        let images = self.get_subset(&self.test_images, num_samples);
        let labels = self.get_subset(&self.test_labels, num_samples);

        (images, labels)
    }

    /// Helper to extract first N rows from a matrix
    fn get_subset(&self, matrix: &Matrix, num_samples: usize) -> Matrix {
        #[cfg(feature = "blas")]
        {
            // For ndarray backend, use axis_iter to iterate over rows
            use ndarray::Axis;
            let subset_data: Vec<Vec<f32>> = matrix
                .axis_iter(Axis(0))
                .take(num_samples)
                .map(|row| row.to_vec())
                .collect();
            MatrixExt::from_vec(subset_data)
        }

        #[cfg(not(feature = "blas"))]
        {
            // For naive backend, iter() works on rows
            let subset_data: Vec<Vec<f32>> = matrix.iter().take(num_samples).cloned().collect();
            MatrixExt::from_vec(subset_data)
        }
    }
}

impl Dataset for MnistDataset {
    fn train_data(&self) -> (&Matrix, &Matrix) {
        (&self.train_images, &self.train_labels)
    }

    fn test_data(&self) -> (&Matrix, &Matrix) {
        (&self.test_images, &self.test_labels)
    }

    fn num_classes(&self) -> usize {
        10 // Digits 0-9
    }

    fn input_size(&self) -> usize {
        784 // 28x28 pixels flattened
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::MatrixExt;

    // Note: These tests require downloading the MNIST dataset (~10MB).
    // Run with: cargo test -- --ignored
    // They are ignored by default to avoid download requirements in CI/CD.

    #[test]
    #[ignore = "requires MNIST dataset download"]
    fn test_mnist_loads() {
        let mnist = MnistDataset::new().expect("Failed to load MNIST");

        // Check training data dimensions
        let (train_x, train_y) = mnist.train_data();
        assert_eq!(MatrixExt::rows(train_x), 60_000);
        assert_eq!(MatrixExt::cols(train_x), 784);
        assert_eq!(MatrixExt::rows(train_y), 60_000);
        assert_eq!(MatrixExt::cols(train_y), 10);

        // Check test data dimensions
        let (test_x, test_y) = mnist.test_data();
        assert_eq!(MatrixExt::rows(test_x), 10_000);
        assert_eq!(MatrixExt::cols(test_x), 784);
        assert_eq!(MatrixExt::rows(test_y), 10_000);
        assert_eq!(MatrixExt::cols(test_y), 10);
    }

    #[test]
    #[ignore = "requires MNIST dataset download"]
    fn test_mnist_metadata() {
        let mnist = MnistDataset::new().expect("Failed to load MNIST");

        assert_eq!(mnist.num_classes(), 10);
        assert_eq!(mnist.input_size(), 784);
        assert_eq!(mnist.train_size(), 60_000);
        assert_eq!(mnist.test_size(), 10_000);
    }

    #[test]
    #[ignore = "requires MNIST dataset download"]
    fn test_mnist_normalization() {
        let mnist = MnistDataset::new().expect("Failed to load MNIST");

        let (train_x, _) = mnist.train_data();

        // Check that pixel values are normalized to [0, 1]
        #[cfg(feature = "blas")]
        {
            use ndarray::Axis;
            for row in train_x.axis_iter(Axis(0)).take(100) {
                for &pixel in row {
                    assert!(
                        (0.0..=1.0).contains(&pixel),
                        "Pixel value out of range: {}",
                        pixel
                    );
                }
            }
        }

        #[cfg(not(feature = "blas"))]
        {
            for row in train_x.iter().take(100) {
                for &pixel in row {
                    assert!(
                        (0.0..=1.0).contains(&pixel),
                        "Pixel value out of range: {}",
                        pixel
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "requires MNIST dataset download"]
    fn test_mnist_labels_one_hot() {
        let mnist = MnistDataset::new().expect("Failed to load MNIST");

        let (_, train_y) = mnist.train_data();

        // Check that labels are one-hot encoded
        #[cfg(feature = "blas")]
        {
            use ndarray::Axis;
            for label_row in train_y.axis_iter(Axis(0)).take(100) {
                let sum: f32 = label_row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-5, "One-hot encoding should sum to 1");

                // Count number of 1s (should be exactly 1)
                let num_ones = label_row
                    .iter()
                    .filter(|&&x| (x - 1.0).abs() < 1e-5)
                    .count();
                assert_eq!(num_ones, 1, "Should have exactly one 1 in one-hot encoding");
            }
        }

        #[cfg(not(feature = "blas"))]
        {
            for label_row in train_y.iter().take(100) {
                let sum: f32 = label_row.iter().sum();
                assert!((sum - 1.0).abs() < 1e-5, "One-hot encoding should sum to 1");

                // Count number of 1s (should be exactly 1)
                let num_ones = label_row
                    .iter()
                    .filter(|&&x| (x - 1.0).abs() < 1e-5)
                    .count();
                assert_eq!(num_ones, 1, "Should have exactly one 1 in one-hot encoding");
            }
        }
    }

    #[test]
    #[ignore = "requires MNIST dataset download"]
    fn test_train_subset() {
        let mnist = MnistDataset::new().expect("Failed to load MNIST");

        let (subset_x, subset_y) = mnist.train_subset(100);
        assert_eq!(MatrixExt::rows(&subset_x), 100);
        assert_eq!(MatrixExt::cols(&subset_x), 784);
        assert_eq!(MatrixExt::rows(&subset_y), 100);
        assert_eq!(MatrixExt::cols(&subset_y), 10);
    }

    #[test]
    #[ignore = "requires MNIST dataset download"]
    fn test_test_subset() {
        let mnist = MnistDataset::new().expect("Failed to load MNIST");

        let (subset_x, subset_y) = mnist.test_subset(50);
        assert_eq!(MatrixExt::rows(&subset_x), 50);
        assert_eq!(MatrixExt::cols(&subset_x), 784);
        assert_eq!(MatrixExt::rows(&subset_y), 50);
        assert_eq!(MatrixExt::cols(&subset_y), 10);
    }

    #[test]
    #[ignore = "requires MNIST dataset download"]
    #[should_panic(expected = "must be <= 60,000")]
    fn test_train_subset_too_large() {
        let mnist = MnistDataset::new().expect("Failed to load MNIST");
        mnist.train_subset(70_000);
    }

    #[test]
    #[ignore = "requires MNIST dataset download"]
    #[should_panic(expected = "must be <= 10,000")]
    fn test_test_subset_too_large() {
        let mnist = MnistDataset::new().expect("Failed to load MNIST");
        mnist.test_subset(20_000);
    }
}
