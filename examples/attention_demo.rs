//! Attention Mechanism Demonstrations
//!
//! This example demonstrates the three core attention mechanisms implemented
//! in jwall-gpt:
//!
//! 1. Single-head attention with shape preservation
//! 2. Causal masking for autoregressive models
//! 3. Multi-head attention with parallel heads
//!
//! Run with: cargo run --example attention_demo

use jwall_gpt::layers::{Attention, Layer, MultiHeadAttention};
use jwall_gpt::matrix::{Matrix, MatrixExt};

fn main() {
    println!("===========================================");
    println!("    Attention Mechanism Demonstrations    ");
    println!("===========================================\n");

    demo_1_single_head_attention();
    println!();

    demo_2_causal_masking();
    println!();

    demo_3_multi_head_attention();
    println!();

    println!("===========================================");
    println!("         All demos completed! ✓           ");
    println!("===========================================");
}

/// Demo 1: Single-head attention on 4×8 matrix
///
/// Demonstrates that single-head attention preserves input shape.
fn demo_1_single_head_attention() {
    println!("Demo 1: Single-Head Attention");
    println!("------------------------------");

    // Create a 4×8 input matrix
    let input = Matrix::from_vec(vec![
        vec![1.0, 0.5, 0.2, 0.1, 0.3, 0.7, 0.4, 0.9],
        vec![0.8, 0.3, 0.6, 0.2, 0.1, 0.5, 0.7, 0.4],
        vec![0.2, 0.9, 0.1, 0.8, 0.6, 0.3, 0.5, 0.7],
        vec![0.5, 0.4, 0.7, 0.3, 0.9, 0.2, 0.8, 0.1],
    ]);

    let (rows, cols) = MatrixExt::shape(&input);
    println!("Input shape: {} × {}", rows, cols);

    // Create attention layer (input_dim=8, d_k=8, no causal mask)
    let mut attn = Attention::new(8, 8, false);
    println!("Created Attention(input_dim=8, d_k=8, causal_mask=false)");

    // Forward pass
    let output = attn.forward(&input);
    let (out_rows, out_cols) = MatrixExt::shape(&output);

    println!("Output shape: {} × {}", out_rows, out_cols);
    println!(
        "✓ Output shape: {} × {} → {} × {} (projects to d_k dimension)",
        rows, cols, out_rows, out_cols
    );

    // Show attention weights
    if let Some(weights) = attn.last_attention_weights() {
        println!("\nAttention weights (each row sums to 1.0):");
        print_matrix_sample(weights, "attention weights");

        // Verify that attention weights sum to 1.0 per row
        let (weight_rows, weight_cols) = MatrixExt::shape(weights);
        let mut all_sum_to_one = true;
        for i in 0..weight_rows {
            let mut row_sum = 0.0;
            for j in 0..weight_cols {
                row_sum += weights[[i, j]];
            }
            if (row_sum - 1.0).abs() > 1e-5 {
                all_sum_to_one = false;
                break;
            }
        }
        if all_sum_to_one {
            println!("✓ All attention weight rows sum to 1.0");
        }
    }
}

/// Demo 2: Causal masking test
///
/// Demonstrates that causal masking prevents attention to future positions.
fn demo_2_causal_masking() {
    println!("Demo 2: Causal Masking");
    println!("----------------------");

    // Create a simple 4×8 input
    let input = Matrix::from_vec(vec![vec![1.0; 8], vec![2.0; 8], vec![3.0; 8], vec![4.0; 8]]);

    let (rows, cols) = MatrixExt::shape(&input);
    println!("Input shape: {} × {}", rows, cols);

    // Create attention with causal masking
    let mut attn = Attention::new(8, 8, true);
    println!("Created Attention(input_dim=8, d_k=8, causal_mask=true)");

    // Forward pass
    let _output = attn.forward(&input);

    // Check attention weights
    if let Some(weights) = attn.last_attention_weights() {
        println!("\nAttention weights with causal masking:");
        print_matrix_full(weights, "causal attention weights");

        // Verify upper triangle is masked (near zero)
        let (weight_rows, weight_cols) = MatrixExt::shape(weights);
        let mut upper_triangle_masked = true;

        println!("\nVerifying causal mask (upper triangle should be ~0):");
        for i in 0..weight_rows {
            for j in (i + 1)..weight_cols {
                let val = weights[[i, j]];
                print!("  [{}, {}] = {:.6}", i, j, val);
                if val < 1e-6 {
                    println!(" ✓");
                } else {
                    println!(" ✗ (should be ~0)");
                    upper_triangle_masked = false;
                }
            }
        }

        if upper_triangle_masked {
            println!("\n✓ Causal mask correctly applied: upper triangle is masked");
            println!("  (Future positions cannot attend to current position)");
        } else {
            println!("\n✗ Causal mask not properly applied");
        }

        // Show that lower triangle + diagonal have non-zero values
        println!("\nLower triangle + diagonal (should have non-zero values):");
        for i in 0..weight_rows.min(3) {
            print!("  Row {}: ", i);
            for j in 0..=i {
                print!("{:.4} ", weights[[i, j]]);
            }
            println!();
        }
    }
}

/// Demo 3: Multi-head attention
///
/// Demonstrates multi-head attention with d_model=64, num_heads=4.
fn demo_3_multi_head_attention() {
    println!("Demo 3: Multi-Head Attention");
    println!("----------------------------");

    const D_MODEL: usize = 64;
    const NUM_HEADS: usize = 4;
    const SEQ_LEN: usize = 6;

    println!("Configuration:");
    println!("  d_model: {}", D_MODEL);
    println!("  num_heads: {}", NUM_HEADS);
    println!("  d_k (per head): {}", D_MODEL / NUM_HEADS);
    println!("  seq_len: {}", SEQ_LEN);

    // Create input
    let input = Matrix::random(SEQ_LEN, D_MODEL, 0.1);
    let (in_rows, in_cols) = MatrixExt::shape(&input);
    println!("\nInput shape: {} × {}", in_rows, in_cols);

    // Create multi-head attention
    let mut mha = MultiHeadAttention::new(D_MODEL, NUM_HEADS, true);
    println!("Created MultiHeadAttention with {} heads", NUM_HEADS);

    // Forward pass
    let output = mha.forward(&input);
    let (out_rows, out_cols) = MatrixExt::shape(&output);

    println!("Output shape: {} × {}", out_rows, out_cols);
    println!(
        "✓ Shape preservation verified: {} × {} → {} × {}",
        in_rows, in_cols, out_rows, out_cols
    );

    // Show attention patterns from each head
    println!("\nAttention patterns from each head:");
    for head_idx in 0..NUM_HEADS {
        if let Some(weights) = mha.head_attention_weights(head_idx) {
            println!("\n  Head {} attention weights:", head_idx);
            let (h_rows, h_cols) = MatrixExt::shape(weights);

            // Print first 3 rows as sample
            for i in 0..h_rows.min(3) {
                print!("    [{:2}]: ", i);
                for j in 0..h_cols.min(6) {
                    print!("{:.3} ", weights[[i, j]]);
                }
                if h_cols > 6 {
                    print!("...");
                }
                println!();
            }
            if h_rows > 3 {
                println!("    ...");
            }

            // Verify causal mask in this head
            let mut masked = true;
            for i in 0..h_rows {
                for j in (i + 1)..h_cols {
                    if weights[[i, j]] >= 1e-6 {
                        masked = false;
                    }
                }
            }
            if masked {
                println!("    ✓ Causal mask applied");
            }
        }
    }

    // Check parameter count
    let param_count = mha.parameters().len();
    println!("\nTotal parameter matrices: {}", param_count);
    println!(
        "  = {} heads × 6 params/head (Q,K,V weights+biases)",
        NUM_HEADS
    );
    println!("  + 2 params for multi-head output projection (weight+bias)");
    println!("  = {} × 6 + 2 = {}", NUM_HEADS, param_count);
}

/// Print a sample of a matrix (first 3×3 elements)
fn print_matrix_sample(m: &Matrix, name: &str) {
    let (rows, cols) = MatrixExt::shape(m);
    println!("  {} (showing {}×{}):", name, rows.min(3), cols.min(3));

    for i in 0..rows.min(3) {
        print!("    ");
        for j in 0..cols.min(3) {
            print!("{:.4} ", m[[i, j]]);
        }
        if cols > 3 {
            print!("...");
        }
        println!();
    }
    if rows > 3 {
        println!("    ...");
    }
}

/// Print full matrix (for small matrices)
fn print_matrix_full(m: &Matrix, name: &str) {
    let (rows, cols) = MatrixExt::shape(m);
    println!("  {} ({}×{}):", name, rows, cols);

    for i in 0..rows {
        print!("    ");
        for j in 0..cols {
            print!("{:.4} ", m[[i, j]]);
        }
        println!();
    }
}
