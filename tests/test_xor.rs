//! Integration test: Train a neural network on XOR problem
//!
//! This validates that the entire training pipeline works:
//! - Matrix operations
//! - Linear layers
//! - Activations
//! - Loss functions
//! - Optimizers
//! - Backpropagation

use jwall_gpt::activations::{relu, relu_derivative, sigmoid, sigmoid_derivative};
use jwall_gpt::layers::{Layer, Linear};
use jwall_gpt::loss::{CrossEntropyLoss, Loss, MSELoss, indices_to_one_hot};
use jwall_gpt::matrix::{Matrix, MatrixExt};
use jwall_gpt::optimizers::{Adam, Optimizer, SGD};

/// Test XOR learning with cross-entropy loss and Adam optimizer
#[test]
fn test_xor_with_adam() {
    println!("\n🧪 Testing XOR with Adam optimizer and CrossEntropy loss");

    // XOR dataset
    let inputs: Matrix = Matrix::from_vec(vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ]);

    // XOR outputs as class indices
    let target_indices = vec![0, 1, 1, 0];
    let targets: Matrix = indices_to_one_hot(&target_indices, 2); // 2 classes

    // Create network: 2 → 8 → 2
    let mut layer1: Linear = Linear::new(2, 8);
    let mut layer2: Linear = Linear::new(8, 2);

    // Loss and optimizer
    let loss_fn: CrossEntropyLoss = CrossEntropyLoss::new();
    let mut optimizer: Adam = Adam::new(0.1);

    // Training loop
    let epochs = 1000;
    let mut final_loss = 0.0;

    for epoch in 0..epochs {
        // Forward pass
        let hidden = layer1.forward(&inputs);
        let hidden_activated = relu(&hidden);
        let output = layer2.forward(&hidden_activated);

        // Compute loss
        let loss = loss_fn.forward(&output, &targets);
        final_loss = loss;

        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, loss);
        }

        // Backward pass
        let grad_output = loss_fn.backward(&output, &targets);
        let grad_hidden_activated = layer2.backward(&grad_output);
        let grad_hidden = &grad_hidden_activated * &relu_derivative(&hidden);
        let _grad_input = layer1.backward(&grad_hidden);

        // Update parameters
        optimizer.step(
            &mut [
                &mut layer1.weights,
                &mut layer1.bias,
                &mut layer2.weights,
                &mut layer2.bias,
            ],
            &[
                &layer1.weight_grad,
                &layer1.bias_grad,
                &layer2.weight_grad,
                &layer2.bias_grad,
            ],
        );

        // Zero gradients
        layer1.zero_grad();
        layer2.zero_grad();
    }

    println!("✅ Final loss: {:.4}", final_loss);

    // Test predictions
    println!("\n🔍 Testing predictions:");
    let hidden = layer1.forward(&inputs);
    let hidden_activated = relu(&hidden);
    let output = layer2.forward(&hidden_activated);

    for i in 0..4 {
        let input_vals = [inputs[[i, 0]], inputs[[i, 1]]];
        let pred_class = if output[[i, 0]] > output[[i, 1]] {
            0
        } else {
            1
        };
        let target_class = target_indices[i];
        let correct = if pred_class == target_class {
            "✓"
        } else {
            "✗"
        };

        println!(
            "{:?} → predicted: {}, target: {} {}",
            input_vals, pred_class, target_class, correct
        );
    }

    // Loss should be low if training succeeded
    assert!(
        final_loss < 0.3,
        "Failed to learn XOR! Final loss: {}",
        final_loss
    );
}

/// Test XOR learning with MSE loss and SGD optimizer
#[test]
fn test_xor_with_sgd_mse() {
    println!("\n🧪 Testing XOR with SGD optimizer and MSE loss");

    // XOR dataset
    let inputs = Matrix::from_vec(vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ]);

    // XOR outputs (single output for regression)
    let targets = Matrix::from_vec(vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]]);

    // Create network: 2 → 4 → 1
    let mut layer1 = Linear::new(2, 4);
    let mut layer2 = Linear::new(4, 1);

    // Loss and optimizer
    let loss_fn = MSELoss::new();
    let mut optimizer = SGD::with_momentum(0.5, 0.9);

    // Training loop
    let epochs = 1000;
    let mut final_loss = 0.0;

    for epoch in 0..epochs {
        // Forward pass
        let hidden = layer1.forward(&inputs);
        let hidden_activated = sigmoid(&hidden);
        let output = layer2.forward(&hidden_activated);
        let output_activated = sigmoid(&output);

        // Compute loss
        let loss = loss_fn.forward(&output_activated, &targets);
        final_loss = loss;

        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, loss);
        }

        // Backward pass
        let grad_output_activated = loss_fn.backward(&output_activated, &targets);
        let grad_output = &grad_output_activated * &sigmoid_derivative(&output);
        let grad_hidden_activated = layer2.backward(&grad_output);
        let grad_hidden = &grad_hidden_activated * &sigmoid_derivative(&hidden);
        let _grad_input = layer1.backward(&grad_hidden);

        // Update parameters
        optimizer.step(
            &mut [
                &mut layer1.weights,
                &mut layer1.bias,
                &mut layer2.weights,
                &mut layer2.bias,
            ],
            &[
                &layer1.weight_grad,
                &layer1.bias_grad,
                &layer2.weight_grad,
                &layer2.bias_grad,
            ],
        );

        // Zero gradients
        layer1.zero_grad();
        layer2.zero_grad();
    }

    println!("✅ Final loss: {:.4}", final_loss);

    // Test predictions
    println!("\n🔍 Testing predictions:");
    let hidden = layer1.forward(&inputs);
    let hidden_activated = sigmoid(&hidden);
    let output = layer2.forward(&hidden_activated);
    let output_activated = sigmoid(&output);

    for i in 0..4 {
        let input_vals = [inputs[[i, 0]], inputs[[i, 1]]];
        let prediction = output_activated[[i, 0]];
        let target = targets[[i, 0]];
        let correct = if (prediction - target).abs() < 0.3 {
            "✓"
        } else {
            "✗"
        };

        println!(
            "{:?} → predicted: {:.3}, target: {:.1} {}",
            input_vals, prediction, target, correct
        );
    }

    // Loss should be low if training succeeded
    assert!(
        final_loss < 0.1,
        "Failed to learn XOR! Final loss: {}",
        final_loss
    );
}

/// Quick sanity check: ensure one training step doesn't crash
#[test]
fn test_single_training_step() {
    let inputs = Matrix::from_vec(vec![vec![1.0, 0.0]]);
    let targets = Matrix::from_vec(vec![vec![1.0, 0.0]]);

    let mut layer = Linear::new(2, 2);
    let loss_fn = CrossEntropyLoss::new();
    let mut optimizer = Adam::new(0.01);

    // One step shouldn't crash
    let output = layer.forward(&inputs);
    let _loss = loss_fn.forward(&output, &targets);
    let grad = loss_fn.backward(&output, &targets);
    layer.backward(&grad);

    optimizer.step(
        &mut [&mut layer.weights, &mut layer.bias],
        &[&layer.weight_grad, &layer.bias_grad],
    );

    // If we get here without panicking, success!
}
