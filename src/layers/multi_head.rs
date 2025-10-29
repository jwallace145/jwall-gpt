//! Multi-head attention mechanism for transformer architectures
//!
//! Multi-head attention allows the model to jointly attend to information from
//! different representation subspaces. This is a key component of transformers
//! that enables them to capture diverse relationships in the input.
//!
//! # Multi-Head Attention Formula
//!
//! ```text
//! MultiHead(X) = Concat(head₁, head₂, ..., headₕ) W^O
//!
//! where headᵢ = Attention(XW^Q_i, XW^K_i, XW^V_i)
//! ```
//!
//! # Architecture
//!
//! Instead of performing a single attention function with d_model-dimensional keys,
//! queries, and values, we project d_model dimensions into h different learned linear
//! projections to d_k dimensions. We then perform attention in parallel on each of
//! these h projections, concatenate the results, and project again.
//!
//! # Example
//!
//! ```
//! use jwall_gpt::layers::{MultiHeadAttention, Layer};
//! use jwall_gpt::matrix::{Matrix, MatrixExt};
//!
//! // Standard GPT-2 configuration: 768-dim model, 12 heads
//! let mut mha = MultiHeadAttention::new(768, 12, true);
//!
//! // Input: [seq_len=10, d_model=768]
//! let input = Matrix::random(10, 768, 0.1);
//! let output = mha.forward(&input);
//!
//! assert_eq!(output.shape(), input.shape());
//! ```

use crate::layers::{Attention, Layer, Linear};
use crate::matrix::{Matrix, MatrixExt, zeros_matrix};

/// Multi-head attention mechanism
///
/// Splits the attention computation across multiple "heads" that operate in parallel,
/// each learning to focus on different aspects of the input.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    /// Individual attention heads
    heads: Vec<Attention>,

    /// Output projection that combines all heads
    output_proj: Linear,

    /// Number of parallel attention heads
    num_heads: usize,

    /// Model dimension (input/output dimension)
    d_model: usize,

    /// Dimension of each head
    d_k: usize,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer
    ///
    /// # Arguments
    ///
    /// * `d_model` - Model dimension (must be divisible by num_heads)
    /// * `num_heads` - Number of parallel attention heads
    /// * `use_causal_mask` - If true, masks future positions (for autoregressive models)
    ///
    /// # Panics
    ///
    /// Panics if d_model is not divisible by num_heads.
    ///
    /// # Example
    ///
    /// ```
    /// use jwall_gpt::layers::MultiHeadAttention;
    ///
    /// // GPT-2 small: 768 dimensions, 12 heads
    /// let mha = MultiHeadAttention::new(768, 12, true);
    ///
    /// // GPT-2 medium: 1024 dimensions, 16 heads
    /// let mha_medium = MultiHeadAttention::new(1024, 16, true);
    /// ```
    pub fn new(d_model: usize, num_heads: usize, use_causal_mask: bool) -> Self {
        assert!(
            d_model.is_multiple_of(num_heads),
            "d_model ({}) must be divisible by num_heads ({})",
            d_model,
            num_heads
        );

        let d_k = d_model / num_heads;

        // Create attention heads
        let heads = (0..num_heads)
            .map(|_| Attention::new(d_model, d_k, use_causal_mask))
            .collect();

        // Output projection to combine heads
        let output_proj = Linear::new(d_model, d_model);

        Self {
            heads,
            output_proj,
            num_heads,
            d_model,
            d_k,
        }
    }

    /// Get the number of attention heads
    #[inline]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get the model dimension
    #[inline]
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    /// Get the dimension of each head
    #[inline]
    pub fn d_k(&self) -> usize {
        self.d_k
    }

    /// Get attention weights from a specific head (for visualization)
    pub fn head_attention_weights(&self, head_idx: usize) -> Option<&Matrix> {
        self.heads
            .get(head_idx)
            .and_then(|head| head.last_attention_weights())
    }

    /// Concatenate outputs from all heads along the feature dimension
    ///
    /// Takes outputs from each head [seq_len, d_k] and concatenates them
    /// to produce [seq_len, d_model] where d_model = num_heads * d_k.
    ///
    /// # Arguments
    ///
    /// * `head_outputs` - Vector of outputs from each head, each [seq_len, d_k]
    ///
    /// # Returns
    ///
    /// Concatenated matrix [seq_len, d_model]
    fn concat_heads(&self, head_outputs: &[Matrix]) -> Matrix {
        if head_outputs.is_empty() {
            return zeros_matrix(0, 0);
        }

        let (seq_len, d_k) = MatrixExt::shape(&head_outputs[0]);
        let d_model = d_k * self.num_heads;

        // Create output matrix
        let mut concat = zeros_matrix(seq_len, d_model);

        // Concatenate each head's output
        for (head_idx, head_output) in head_outputs.iter().enumerate() {
            let start_col = head_idx * d_k;
            for i in 0..seq_len {
                for j in 0..d_k {
                    concat[[i, start_col + j]] = head_output[[i, j]];
                }
            }
        }

        concat
    }
}

impl Layer for MultiHeadAttention {
    /// Forward pass: compute multi-head attention output
    ///
    /// # Algorithm
    ///
    /// 1. Run each attention head on the input in parallel
    /// 2. Concatenate all head outputs along feature dimension
    /// 3. Apply output projection
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor [seq_len, d_model]
    ///
    /// # Returns
    ///
    /// Output tensor with same shape as input [seq_len, d_model]
    ///
    /// # Panics
    ///
    /// Panics if input dimension doesn't match d_model.
    fn forward(&mut self, input: &Matrix) -> Matrix {
        let (_seq_len, input_dim) = MatrixExt::shape(input);

        assert_eq!(
            input_dim, self.d_model,
            "Input dimension mismatch: expected {}, got {}",
            self.d_model, input_dim
        );

        // 1. Run each head
        let head_outputs: Vec<Matrix> = self
            .heads
            .iter_mut()
            .map(|head| head.forward(input))
            .collect();

        // 2. Concatenate head outputs
        let concat = self.concat_heads(&head_outputs);

        // 3. Apply output projection
        self.output_proj.forward(&concat)
    }

    /// Backward pass: compute gradients
    ///
    /// Propagates gradients back through:
    /// 1. Output projection
    /// 2. Concatenation (splits gradient)
    /// 3. Each attention head
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient from next layer [seq_len, d_model]
    ///
    /// # Returns
    ///
    /// Gradient w.r.t. input [seq_len, d_model]
    fn backward(&mut self, grad_output: &Matrix) -> Matrix {
        // Backprop through output projection
        let grad_concat = self.output_proj.backward(grad_output);

        let (seq_len, _) = MatrixExt::shape(&grad_concat);

        // Split gradient for each head
        let mut grad_inputs = Vec::new();

        for (head_idx, head) in self.heads.iter_mut().enumerate() {
            let start_col = head_idx * self.d_k;

            // Extract gradient for this head
            let mut grad_head = zeros_matrix(seq_len, self.d_k);
            for i in 0..seq_len {
                for j in 0..self.d_k {
                    grad_head[[i, j]] = grad_concat[[i, start_col + j]];
                }
            }

            // Backprop through this head
            let grad_input = head.backward(&grad_head);
            grad_inputs.push(grad_input);
        }

        // Sum gradients from all heads (since each head sees the full input)
        let mut total_grad = grad_inputs[0].clone();
        for grad in grad_inputs.iter().skip(1) {
            let (rows, cols) = MatrixExt::shape(&total_grad);
            for i in 0..rows {
                for j in 0..cols {
                    total_grad[[i, j]] += grad[[i, j]];
                }
            }
        }

        total_grad
    }

    /// Get mutable references to all learnable parameters
    ///
    /// Returns parameters from all attention heads and the output projection.
    fn parameters(&mut self) -> Vec<&mut Matrix> {
        let mut params = Vec::new();

        // Parameters from all heads
        for head in self.heads.iter_mut() {
            params.extend(head.parameters());
        }

        // Parameters from output projection
        params.extend(self.output_proj.parameters());

        params
    }

    /// Zero out all gradients
    fn zero_grad(&mut self) {
        for head in self.heads.iter_mut() {
            head.zero_grad();
        }
        self.output_proj.zero_grad();
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
    fn test_multihead_shape_preservation() {
        let mut mha = MultiHeadAttention::new(64, 4, false);
        let input = Matrix::random(10, 64, 0.1);

        let output = mha.forward(&input);

        assert_eq!(MatrixExt::shape(&output), (10, 64));
    }

    #[test]
    fn test_multihead_with_different_configs() {
        // Test various standard configurations
        let configs = vec![
            (768, 12),  // GPT-2 small
            (1024, 16), // GPT-2 medium
            (512, 8),   // Transformer base
            (64, 4),    // Small test config
        ];

        for (d_model, num_heads) in configs {
            let mut mha = MultiHeadAttention::new(d_model, num_heads, true);
            let input = Matrix::random(5, d_model, 0.1);
            let output = mha.forward(&input);

            assert_eq!(
                MatrixExt::shape(&output),
                (5, d_model),
                "Failed for config d_model={}, num_heads={}",
                d_model,
                num_heads
            );
        }
    }

    #[test]
    #[should_panic(expected = "d_model")]
    fn test_invalid_head_count() {
        // d_model not divisible by num_heads
        MultiHeadAttention::new(100, 7, false);
    }

    #[test]
    fn test_head_attention_weights_accessible() {
        let mut mha = MultiHeadAttention::new(64, 4, false);
        let input = Matrix::random(5, 64, 0.1);

        let _output = mha.forward(&input);

        // Check that we can access attention weights from each head
        for head_idx in 0..4 {
            let weights = mha.head_attention_weights(head_idx);
            assert!(
                weights.is_some(),
                "Head {} weights should be available",
                head_idx
            );

            let weights = weights.unwrap();
            let (rows, cols) = MatrixExt::shape(weights);
            assert_eq!(
                (rows, cols),
                (5, 5),
                "Attention weights should be [seq_len, seq_len]"
            );
        }
    }

    #[test]
    fn test_parameters_count() {
        let mut mha = MultiHeadAttention::new(64, 4, false);
        let params = mha.parameters();

        // Each head has 6 parameters (3 weight matrices + 3 biases for Q, K, V)
        // Plus multi-head output projection has 2 parameters (weight + bias)
        // Total: 4 heads * 6 + 2 = 26
        assert_eq!(params.len(), 26);
    }

    #[test]
    fn test_concat_heads() {
        let mha = MultiHeadAttention::new(64, 4, false);

        // Create 4 head outputs, each [3, 16]
        let head_outputs: Vec<Matrix> = (0..4)
            .map(|i| {
                let val = (i + 1) as f32;
                Matrix::from_vec(vec![
                    vec![val; 16],
                    vec![val * 2.0; 16],
                    vec![val * 3.0; 16],
                ])
            })
            .collect();

        let concat = mha.concat_heads(&head_outputs);
        assert_eq!(MatrixExt::shape(&concat), (3, 64));

        // Verify concatenation is correct
        assert_float_eq(concat[[0, 0]], 1.0, 1e-5); // First head, first element
        assert_float_eq(concat[[0, 16]], 2.0, 1e-5); // Second head, first element
        assert_float_eq(concat[[0, 32]], 3.0, 1e-5); // Third head, first element
        assert_float_eq(concat[[0, 48]], 4.0, 1e-5); // Fourth head, first element
    }
}
