//! Single-head attention mechanism for transformer architectures
//!
//! Attention allows the model to focus on different parts of the input sequence
//! when processing each element. This is the core building block of transformers.
//!
//! # Attention Formula
//!
//! ```text
//! Attention(Q, K, V) = softmax(QK^T / √d_k) V
//! ```
//!
//! Where:
//! - Q (Query): What we're looking for
//! - K (Key): What each position offers
//! - V (Value): The actual content at each position
//! - d_k: Dimension of keys (for scaling)
//!
//! # Example
//!
//! ```
//! use jwall_gpt::layers::{Attention, Layer};
//! use jwall_gpt::matrix::{Matrix, MatrixExt};
//!
//! // Create attention layer
//! let mut attn = Attention::new(512, 64, true);
//!
//! // Input: [seq_len=10, input_dim=512]
//! let input = Matrix::random(10, 512, 0.1);
//! let output = attn.forward(&input);
//!
//! // Output shape: [seq_len, d_k] = [10, 64]
//! assert_eq!(MatrixExt::shape(&output), (10, 64));
//! ```

use crate::layers::{Layer, Linear};
use crate::matrix::{Matrix, MatrixExt, zeros_matrix};

/// Single-head attention mechanism
///
/// Computes scaled dot-product attention with optional causal masking.
/// This is typically used as a building block for multi-head attention.
#[derive(Debug, Clone)]
pub struct Attention {
    /// Query projection: maps input to query space
    pub q_proj: Linear,

    /// Key projection: maps input to key space
    pub k_proj: Linear,

    /// Value projection: maps input to value space
    pub v_proj: Linear,

    /// Dimension of keys/queries (for scaling)
    d_k: usize,

    /// Whether to apply causal masking (for autoregressive models)
    use_causal_mask: bool,

    /// Cached attention weights (for visualization/analysis)
    last_attention_weights: Option<Matrix>,
}

impl Attention {
    /// Create a new single-head attention layer
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Dimension of input features
    /// * `d_k` - Dimension of queries and keys (output dimension)
    /// * `use_causal_mask` - If true, masks future positions (for autoregressive models like GPT)
    ///
    /// # Example
    ///
    /// ```
    /// use jwall_gpt::layers::Attention;
    ///
    /// // For GPT-style autoregressive attention
    /// let attn = Attention::new(512, 64, true);
    ///
    /// // For BERT-style bidirectional attention
    /// let attn_bidir = Attention::new(512, 64, false);
    /// ```
    pub fn new(input_dim: usize, d_k: usize, use_causal_mask: bool) -> Self {
        Self {
            q_proj: Linear::new(input_dim, d_k),
            k_proj: Linear::new(input_dim, d_k),
            v_proj: Linear::new(input_dim, d_k),
            d_k,
            use_causal_mask,
            last_attention_weights: None,
        }
    }

    /// Get the dimension of keys/queries
    #[inline]
    pub fn d_k(&self) -> usize {
        self.d_k
    }

    /// Get the last computed attention weights (for visualization)
    pub fn last_attention_weights(&self) -> Option<&Matrix> {
        self.last_attention_weights.as_ref()
    }

    /// Compute row-wise softmax with numerical stability
    ///
    /// Applies softmax to each row independently, so each row sums to 1.0.
    /// Uses the log-sum-exp trick for numerical stability.
    ///
    /// # Arguments
    ///
    /// * `scores` - Attention scores [seq_len, seq_len]
    ///
    /// # Returns
    ///
    /// Attention weights [seq_len, seq_len] where each row sums to 1.0
    fn softmax(&self, scores: &Matrix) -> Matrix {
        let (rows, cols) = MatrixExt::shape(scores);
        let mut probs = zeros_matrix(rows, cols);

        for i in 0..rows {
            // Find max for numerical stability
            let mut max_score = scores[[i, 0]];
            for j in 1..cols {
                max_score = max_score.max(scores[[i, j]]);
            }

            // Compute exp(score - max) and sum
            let mut sum_exp = 0.0;
            for j in 0..cols {
                let exp_val = (scores[[i, j]] - max_score).exp();
                probs[[i, j]] = exp_val;
                sum_exp += exp_val;
            }

            // Normalize to get probabilities
            for j in 0..cols {
                probs[[i, j]] /= sum_exp;
            }
        }

        probs
    }

    /// Apply causal mask to attention scores
    ///
    /// Sets scores[i][j] = -1e9 for j > i, preventing attention to future positions.
    /// This ensures that position i can only attend to positions 0..=i.
    ///
    /// # Arguments
    ///
    /// * `scores` - Attention scores [seq_len, seq_len]
    ///
    /// # Returns
    ///
    /// Masked scores where future positions are set to -∞ (approximated as -1e9)
    fn apply_causal_mask(&self, scores: &mut Matrix) {
        let (rows, cols) = MatrixExt::shape(scores);

        // Set upper triangle to -1e9 (approximates -∞)
        for i in 0..rows {
            for j in (i + 1)..cols {
                scores[[i, j]] = -1e9;
            }
        }
    }
}

impl Layer for Attention {
    /// Forward pass: compute attention output
    ///
    /// # Algorithm
    ///
    /// 1. Project input to Q, K, V using linear layers
    /// 2. Compute scores = QK^T / √d_k
    /// 3. Apply causal mask if enabled (sets future positions to -∞)
    /// 4. Compute attention_weights = softmax(scores)
    /// 5. Compute output = attention_weights × V
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor [seq_len, input_dim] or [batch, seq_len, input_dim]
    ///
    /// # Returns
    ///
    /// Output tensor [seq_len, d_k]
    ///
    /// # Panics
    ///
    /// Panics if input dimensions don't match the expected input_dim.
    fn forward(&mut self, input: &Matrix) -> Matrix {
        // 1. Project to Q, K, V
        let q = self.q_proj.forward(input); // [seq_len, d_k]
        let k = self.k_proj.forward(input); // [seq_len, d_k]
        let v = self.v_proj.forward(input); // [seq_len, d_k]

        // 2. Compute attention scores: QK^T / √d_k
        let k_t = MatrixExt::transpose(&k); // [d_k, seq_len]
        let mut scores = MatrixExt::matmul(&q, &k_t); // [seq_len, seq_len]

        // Scale by √d_k for stability
        let scale = 1.0 / (self.d_k as f32).sqrt();
        let (rows, cols) = MatrixExt::shape(&scores);
        for i in 0..rows {
            for j in 0..cols {
                scores[[i, j]] *= scale;
            }
        }

        // 3. Apply causal mask if enabled
        if self.use_causal_mask {
            self.apply_causal_mask(&mut scores);
        }

        // 4. Compute attention weights via softmax
        let attention_weights = self.softmax(&scores); // [seq_len, seq_len]
        self.last_attention_weights = Some(attention_weights.clone());

        // 5. Compute output: attention_weights × V
        MatrixExt::matmul(&attention_weights, &v) // [seq_len, d_k]
    }

    /// Backward pass: compute gradients
    ///
    /// Note: Full backpropagation through attention is complex and involves
    /// gradients through softmax and matrix multiplications. This is a simplified
    /// implementation that propagates gradients through the value projection.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient from next layer [seq_len, d_k]
    ///
    /// # Returns
    ///
    /// Gradient w.r.t. input [seq_len, input_dim]
    fn backward(&mut self, grad_output: &Matrix) -> Matrix {
        // Simplified backward pass
        // In a full implementation, we'd need to backprop through:
        // 1. V projection
        // 2. Attention weights matrix multiplication
        // 3. Softmax
        // 4. Scores scaling
        // 5. QK^T multiplication
        // 6. Q and K projections

        // For now, we'll implement a simplified version that at least
        // propagates gradients through the value projection
        let grad_v = grad_output;

        // TODO: Implement full attention backward pass
        self.v_proj.backward(grad_v)
    }

    /// Get mutable references to all learnable parameters
    ///
    /// Returns parameters from Q, K, and V projection layers.
    fn parameters(&mut self) -> Vec<&mut Matrix> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params
    }

    /// Zero out all gradients
    fn zero_grad(&mut self) {
        self.q_proj.zero_grad();
        self.k_proj.zero_grad();
        self.v_proj.zero_grad();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_float_eq(a: f32, b: f32, epsilon: f32) {
        assert!(
            (a - b).abs() < epsilon,
            "Expected {} to be close to {} (within {})",
            a,
            b,
            epsilon
        );
    }

    #[test]
    fn test_attention_shape_preservation() {
        let mut attn = Attention::new(8, 8, false);
        let input = Matrix::from_vec(vec![vec![1.0; 8], vec![2.0; 8], vec![3.0; 8], vec![4.0; 8]]);

        let output = attn.forward(&input);

        assert_eq!(MatrixExt::shape(&output), (4, 8));
    }

    #[test]
    fn test_attention_weights_sum_to_one() {
        let mut attn = Attention::new(8, 8, false);
        let input = Matrix::random(4, 8, 0.1);

        let _output = attn.forward(&input);
        let weights = attn.last_attention_weights().unwrap();

        // Each row should sum to 1.0
        let (rows, cols) = MatrixExt::shape(weights);
        for i in 0..rows {
            let mut row_sum = 0.0;
            for j in 0..cols {
                row_sum += weights[[i, j]];
            }
            assert_float_eq(row_sum, 1.0, 1e-5);
        }
    }

    #[test]
    fn test_causal_masking() {
        let mut attn = Attention::new(8, 8, true);
        let input = Matrix::from_vec(vec![vec![1.0; 8], vec![2.0; 8], vec![3.0; 8], vec![4.0; 8]]);

        let _output = attn.forward(&input);
        let weights = attn.last_attention_weights().unwrap();

        // Check that upper triangle is approximately zero (masked)
        let (rows, cols) = MatrixExt::shape(weights);
        for i in 0..rows {
            for j in (i + 1)..cols {
                assert!(
                    weights[[i, j]] < 1e-6,
                    "Position [{}, {}] should be masked but got {}",
                    i,
                    j,
                    weights[[i, j]]
                );
            }
        }
    }

    #[test]
    fn test_different_d_k() {
        // Test with d_k different from input_dim
        let mut attn = Attention::new(16, 8, false);
        let input = Matrix::random(5, 16, 0.1);

        let output = attn.forward(&input);

        // Output should have shape [seq_len, d_k]
        assert_eq!(MatrixExt::shape(&output), (5, 8));
    }

    #[test]
    fn test_parameters_accessible() {
        let mut attn = Attention::new(8, 8, false);
        let params = attn.parameters();

        // Should have 6 parameters: 3 weight matrices + 3 bias vectors (Q, K, V)
        assert_eq!(params.len(), 6);
    }
}
