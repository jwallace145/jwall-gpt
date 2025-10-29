//! Example: Training an MLP on MNIST
//!
//! This example demonstrates training a multi-layer perceptron (MLP) on the MNIST
//! handwritten digits dataset.
//!
//! Architecture:
//! - Input: 784 features (28x28 pixels)
//! - Hidden Layer 1: 256 neurons with ReLU activation
//! - Hidden Layer 2: 128 neurons with ReLU activation
//! - Output: 10 classes (digits 0-9) with Softmax via CrossEntropy
//!
//! Run with:
//! ```bash
//! cargo run --example mnist_mlp --release
//! cargo run --example mnist_mlp --release --no-default-features  # naive backend
//! ```

use jwall_gpt::activations::relu;
use jwall_gpt::datasets::{Dataset, MnistDataset};
use jwall_gpt::layers::{Layer, Linear};
use jwall_gpt::loss::{CrossEntropyLoss, Loss};
use jwall_gpt::matrix::{Matrix, MatrixExt, zeros_matrix};
use jwall_gpt::optimizers::{Adam, Optimizer};

/// Multi-Layer Perceptron for MNIST classification
struct MnistMLP {
    fc1: Linear, // 784 -> 256
    fc2: Linear, // 256 -> 128
    fc3: Linear, // 128 -> 10

    // Cache activations for backward pass
    h1: Option<Matrix>, // After fc1 + ReLU
    h2: Option<Matrix>, // After fc2 + ReLU
}

impl MnistMLP {
    fn new() -> Self {
        Self {
            fc1: Linear::new(784, 256),
            fc2: Linear::new(256, 128),
            fc3: Linear::new(128, 10),
            h1: None,
            h2: None,
        }
    }

    /// Forward pass: Input -> fc1 -> ReLU -> fc2 -> ReLU -> fc3 -> logits
    fn forward(&mut self, input: &Matrix) -> Matrix {
        // Layer 1: Linear + ReLU
        let z1 = self.fc1.forward(input);
        let h1 = apply_relu(&z1);
        self.h1 = Some(h1.clone());

        // Layer 2: Linear + ReLU
        let z2 = self.fc2.forward(&h1);
        let h2 = apply_relu(&z2);
        self.h2 = Some(h2.clone());

        // Layer 3: Linear (logits)
        self.fc3.forward(&h2)
    }

    /// Backward pass: propagate gradients through the network
    fn backward(&mut self, grad_output: &Matrix) {
        let h2 = self.h2.as_ref().expect("forward must be called first");
        let h1 = self.h1.as_ref().expect("forward must be called first");

        // Backward through fc3
        let grad_h2 = self.fc3.backward(grad_output);

        // Backward through ReLU
        let grad_z2 = apply_relu_backward(&grad_h2, h2);

        // Backward through fc2
        let grad_h1 = self.fc2.backward(&grad_z2);

        // Backward through ReLU
        let grad_z1 = apply_relu_backward(&grad_h1, h1);

        // Backward through fc1
        let _ = self.fc1.backward(&grad_z1);
    }

    /// Get all trainable parameters
    fn parameters(&mut self) -> Vec<&mut Matrix> {
        let mut params = Vec::new();
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }

    /// Zero all gradients
    fn zero_grad(&mut self) {
        self.fc1.zero_grad();
        self.fc2.zero_grad();
        self.fc3.zero_grad();
    }
}

/// Apply ReLU activation element-wise
fn apply_relu(matrix: &Matrix) -> Matrix {
    let (rows, cols) = MatrixExt::shape(matrix);
    let mut result = zeros_matrix(rows, cols);

    for i in 0..rows {
        for j in 0..cols {
            result[[i, j]] = relu::relu(matrix[[i, j]]);
        }
    }

    result
}

/// Apply ReLU derivative element-wise
fn apply_relu_backward(grad_output: &Matrix, activation: &Matrix) -> Matrix {
    let (rows, cols) = MatrixExt::shape(grad_output);
    let mut result = zeros_matrix(rows, cols);

    for i in 0..rows {
        for j in 0..cols {
            result[[i, j]] = grad_output[[i, j]] * relu::relu_derivative(activation[[i, j]]);
        }
    }

    result
}

/// Extract a mini-batch from the dataset
fn extract_batch(
    images: &Matrix,
    labels: &Matrix,
    batch_start: usize,
    batch_size: usize,
) -> (Matrix, Matrix) {
    let total_samples = MatrixExt::rows(images);
    let batch_end = (batch_start + batch_size).min(total_samples);

    // Extract batch
    #[cfg(feature = "blas")]
    {
        use ndarray::s;
        let batch_images = images.slice(s![batch_start..batch_end, ..]).to_owned();
        let batch_labels = labels.slice(s![batch_start..batch_end, ..]).to_owned();
        (batch_images, batch_labels)
    }

    #[cfg(not(feature = "blas"))]
    {
        let mut batch_images_data = Vec::new();
        let mut batch_labels_data = Vec::new();

        for i in batch_start..batch_end {
            batch_images_data.push(images[i].clone());
            batch_labels_data.push(labels[i].clone());
        }

        (
            MatrixExt::from_vec(batch_images_data),
            MatrixExt::from_vec(batch_labels_data),
        )
    }
}

/// Compute argmax for each row (predicted class)
fn argmax_rows(matrix: &Matrix) -> Vec<usize> {
    let rows = MatrixExt::rows(matrix);
    let cols = MatrixExt::cols(matrix);

    let mut predictions = Vec::with_capacity(rows);

    for i in 0..rows {
        let mut max_idx = 0;
        let mut max_val = matrix[[i, 0]];

        for j in 1..cols {
            if matrix[[i, j]] > max_val {
                max_val = matrix[[i, j]];
                max_idx = j;
            }
        }

        predictions.push(max_idx);
    }

    predictions
}

/// Evaluate accuracy on a dataset
fn evaluate(model: &mut MnistMLP, images: &Matrix, labels: &Matrix) -> f32 {
    let num_samples = MatrixExt::rows(images);

    // Forward pass
    let logits = model.forward(images);

    // Get predictions
    let predictions = argmax_rows(&logits);
    let true_labels = argmax_rows(labels);

    // Count correct predictions
    let correct = predictions
        .iter()
        .zip(true_labels.iter())
        .filter(|(pred, true_label)| pred == true_label)
        .count();

    (correct as f32) / (num_samples as f32)
}

fn main() {
    println!("=== MNIST MLP Training ===\n");

    // Load MNIST dataset
    println!("Loading MNIST dataset...");
    let mnist = match MnistDataset::new() {
        Ok(dataset) => dataset,
        Err(e) => {
            eprintln!("Failed to load MNIST: {}", e);
            return;
        }
    };
    println!("Dataset loaded successfully!\n");

    // Get full training and test sets
    let (train_images, train_labels) = mnist.train_data();
    let (test_images, test_labels) = mnist.test_data();

    println!("Dataset Statistics:");
    println!("  Training samples: {}", MatrixExt::rows(train_images));
    println!("  Test samples:     {}", MatrixExt::rows(test_images));
    println!("  Input features:   {}", MatrixExt::cols(train_images));
    println!("  Output classes:   {}\n", MatrixExt::cols(train_labels));

    // Initialize model and optimizer
    let mut model = MnistMLP::new();
    let mut optimizer = Adam::new(0.001);
    let loss_fn = CrossEntropyLoss::new();

    println!("Model Architecture:");
    println!("  Layer 1: 784 -> 256 (ReLU)");
    println!("  Layer 2: 256 -> 128 (ReLU)");
    println!("  Layer 3: 128 -> 10  (Softmax)\n");

    // Training hyperparameters
    let batch_size = 32;
    let epochs = 10;
    let num_batches = MatrixExt::rows(train_images).div_ceil(batch_size);

    println!("Training Configuration:");
    println!("  Batch size:       {}", batch_size);
    println!("  Epochs:           {}", epochs);
    println!("  Learning rate:    {}", optimizer.learning_rate());
    println!(
        "  Optimizer:        Adam (beta1={}, beta2={})",
        optimizer.beta1, optimizer.beta2
    );
    println!("  Batches per epoch: {}\n", num_batches);

    println!("Starting training...\n");

    // Training loop
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let mut num_batches_processed = 0;

        // Mini-batch training
        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * batch_size;

            // Extract mini-batch
            let (batch_images, batch_labels) =
                extract_batch(train_images, train_labels, batch_start, batch_size);

            // Zero gradients
            model.zero_grad();

            // Forward pass
            let logits = model.forward(&batch_images);

            // Compute loss
            let loss = loss_fn.forward(&logits, &batch_labels);
            epoch_loss += loss;

            // Backward pass
            let grad_loss = loss_fn.backward(&logits, &batch_labels);
            model.backward(&grad_loss);

            // Update parameters
            // Clone gradients to avoid borrow conflicts
            let grad_copies = [
                model.fc1.weight_grad.clone(),
                model.fc1.bias_grad.clone(),
                model.fc2.weight_grad.clone(),
                model.fc2.bias_grad.clone(),
                model.fc3.weight_grad.clone(),
                model.fc3.bias_grad.clone(),
            ];
            let grad_refs: Vec<&Matrix> = grad_copies.iter().collect();

            // Get mutable parameters
            let mut params = model.parameters();
            optimizer.step(&mut params, &grad_refs);

            num_batches_processed += 1;

            // Print progress every 500 batches
            if (batch_idx + 1) % 500 == 0 {
                let avg_loss = epoch_loss / num_batches_processed as f32;
                println!(
                    "  Epoch {}/{}, Batch {}/{}: Loss = {:.4}",
                    epoch + 1,
                    epochs,
                    batch_idx + 1,
                    num_batches,
                    avg_loss
                );
            }
        }

        // Compute average loss for the epoch
        let avg_epoch_loss = epoch_loss / num_batches_processed as f32;

        // Evaluate on a subset of training data for speed
        let (train_subset_images, train_subset_labels) = mnist.train_subset(1000);
        let train_acc = evaluate(&mut model, &train_subset_images, &train_subset_labels);

        println!(
            "Epoch {}/{}: Loss = {:.4}, Train Acc = {:.2}%",
            epoch + 1,
            epochs,
            avg_epoch_loss,
            train_acc * 100.0
        );
    }

    println!("\n=== Training Complete ===\n");

    // Final evaluation on full test set
    println!("Evaluating on test set...");
    let test_acc = evaluate(&mut model, test_images, test_labels);

    println!("\n=== Final Results ===");
    println!("Test Accuracy: {:.2}%", test_acc * 100.0);

    if test_acc > 0.95 {
        println!("\nExcellent! The model achieved >95% accuracy!");
    } else if test_acc > 0.90 {
        println!("\nGood! The model achieved >90% accuracy.");
    } else {
        println!("\nThe model may need more training or hyperparameter tuning.");
    }
}
