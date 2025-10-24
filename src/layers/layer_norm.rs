//! Layer Normalization Utilities
//!
//! This module provides statistical functions used for layer normalization
//! in neural networks, particularly in transformer architectures like GPT.
//!
//! Layer normalization normalizes activations across features for each example,
//! which helps stabilize training and improve convergence.

#[inline]
pub fn mean(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f32>() / x.len() as f32
}

#[inline]
pub fn variance(x: &[f32], mean_val: f32) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().map(|&val| (val - mean_val).powi(2)).sum::<f32>() / x.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================
    // HELPER FUNCTIONS
    // ============================================

    /// Helper to compare floating point numbers with tolerance
    fn assert_float_eq(a: f32, b: f32, tolerance: f32) {
        assert!(
            (a - b).abs() < tolerance,
            "Expected {}, got {} (difference: {})",
            b,
            a,
            (a - b).abs()
        );
    }

    // ============================================
    // MEAN TESTS
    // ============================================

    #[test]
    fn test_mean_simple() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_float_eq(mean(&x), 3.0, 0.0001);
    }

    #[test]
    fn test_mean_all_same() {
        let x = vec![5.0, 5.0, 5.0, 5.0];
        assert_float_eq(mean(&x), 5.0, 0.0001);
    }

    #[test]
    fn test_mean_single_element() {
        let x = vec![42.0];
        assert_float_eq(mean(&x), 42.0, 0.0001);
    }

    #[test]
    fn test_mean_negative_values() {
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        assert_float_eq(mean(&x), 0.0, 0.0001);
    }

    #[test]
    fn test_mean_mixed_values() {
        let x = vec![-10.0, 0.0, 10.0];
        assert_float_eq(mean(&x), 0.0, 0.0001);
    }

    #[test]
    fn test_mean_empty_array() {
        let x: Vec<f32> = vec![];
        assert_float_eq(mean(&x), 0.0, 0.0001);
    }

    #[test]
    fn test_mean_decimal_values() {
        let x = vec![1.5, 2.5, 3.5];
        assert_float_eq(mean(&x), 2.5, 0.0001);
    }

    #[test]
    fn test_mean_large_values() {
        let x = vec![100.0, 200.0, 300.0];
        assert_float_eq(mean(&x), 200.0, 0.01);
    }

    // ============================================
    // VARIANCE TESTS
    // ============================================

    #[test]
    fn test_variance_zero() {
        // All values are the same, so variance should be 0
        let x = vec![5.0, 5.0, 5.0, 5.0];
        let m = mean(&x);
        assert_float_eq(variance(&x, m), 0.0, 0.0001);
    }

    #[test]
    fn test_variance_simple() {
        // [1, 2, 3, 4, 5] has mean 3
        // Variance = ((1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²) / 5
        //          = (4 + 1 + 0 + 1 + 4) / 5 = 10 / 5 = 2
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let m = mean(&x);
        assert_float_eq(variance(&x, m), 2.0, 0.0001);
    }

    #[test]
    fn test_variance_single_element() {
        // Single element has variance 0
        let x = vec![42.0];
        let m = mean(&x);
        assert_float_eq(variance(&x, m), 0.0, 0.0001);
    }

    #[test]
    fn test_variance_symmetric() {
        // [-2, -1, 0, 1, 2] centered at 0
        // Variance = (4 + 1 + 0 + 1 + 4) / 5 = 2
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let m = mean(&x);
        assert_float_eq(variance(&x, m), 2.0, 0.0001);
    }

    #[test]
    fn test_variance_empty_array() {
        let x: Vec<f32> = vec![];
        let m = mean(&x);
        assert_float_eq(variance(&x, m), 0.0, 0.0001);
    }

    #[test]
    fn test_variance_two_values() {
        // [0, 10] has mean 5
        // Variance = ((0-5)² + (10-5)²) / 2 = (25 + 25) / 2 = 25
        let x = vec![0.0, 10.0];
        let m = mean(&x);
        assert_float_eq(variance(&x, m), 25.0, 0.0001);
    }

    #[test]
    fn test_variance_always_non_negative() {
        // Variance should always be >= 0
        let test_cases = vec![
            vec![1.0, 2.0, 3.0],
            vec![-5.0, -10.0, -15.0],
            vec![0.0, 0.0, 0.0],
            vec![100.0, 200.0],
            vec![-50.0, 50.0],
        ];

        for x in test_cases {
            let m = mean(&x);
            let var = variance(&x, m);
            assert!(
                var >= 0.0,
                "Variance should be non-negative, got {} for {:?}",
                var,
                x
            );
        }
    }

    #[test]
    fn test_variance_with_provided_mean() {
        // Test that variance correctly uses the provided mean
        let x = vec![2.0, 4.0, 6.0, 8.0];
        let provided_mean = 5.0;

        // Variance = ((2-5)² + (4-5)² + (6-5)² + (8-5)²) / 4
        //          = (9 + 1 + 1 + 9) / 4 = 20 / 4 = 5
        assert_float_eq(variance(&x, provided_mean), 5.0, 0.0001);
    }

    // ============================================
    // INTEGRATION TESTS (Mean and Variance Together)
    // ============================================

    #[test]
    fn test_mean_and_variance_workflow() {
        // Standard workflow: compute mean first, then variance
        let x = vec![10.0, 12.0, 23.0, 23.0, 16.0, 23.0, 21.0, 16.0];

        let m = mean(&x);
        assert_float_eq(m, 18.0, 0.01);

        let var = variance(&x, m);
        // Expected variance ≈ 24.0
        assert_float_eq(var, 24.0, 0.1);
    }

    #[test]
    fn test_standard_normal_like_distribution() {
        // Values roughly following a normal distribution centered at 0
        let x = vec![-1.0, -0.5, 0.0, 0.5, 1.0];

        let m = mean(&x);
        assert_float_eq(m, 0.0, 0.0001);

        let var = variance(&x, m);
        // Variance = (1 + 0.25 + 0 + 0.25 + 1) / 5 = 2.5 / 5 = 0.5
        assert_float_eq(var, 0.5, 0.0001);
    }

    #[test]
    fn test_layer_norm_use_case() {
        // Simulating a typical layer norm scenario with activations
        let activations = vec![0.5, 1.0, 1.5, 2.0, 2.5];

        let m = mean(&activations);
        assert_float_eq(m, 1.5, 0.0001);

        let var = variance(&activations, m);
        assert_float_eq(var, 0.5, 0.0001);

        // Standard deviation (sqrt of variance) would be used for normalization
        let std_dev = var.sqrt();
        assert_float_eq(std_dev, 0.7071, 0.001); // sqrt(0.5) ≈ 0.707
    }
}
