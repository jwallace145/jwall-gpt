//! # Matrix Backend Comparison Benchmarks
//!
//! This benchmark suite compares the performance of the BLAS-optimized matrix backend
//! against the naive pure-Rust implementation. The goal is to demonstrate the significant
//! performance improvements gained by using OpenBLAS for matrix operations in GPT models.
//!
//! ## Benchmark Categories
//!
//! 1. **Matrix Multiplication (matmul)**: The most critical operation for neural networks,
//!    tested across small, medium, and large matrix sizes
//! 2. **Transpose**: Common operation in backpropagation and attention mechanisms
//! 3. **Element-wise Addition**: Used in bias addition and residual connections
//! 4. **Element-wise Multiplication**: Used in activation functions and gating mechanisms
//!
//! ## Matrix Sizes
//!
//! - **Small**: 32x32 matrices - representative of small batch operations
//! - **Medium**: 256x256 matrices - typical for small model layers
//! - **Large**: 1024x1024 matrices - representative of larger model computations
//!
//! ## Running Benchmarks
//!
//! ```bash
//! # Run all benchmarks (requires BLAS feature)
//! cargo bench
//!
//! # Run specific benchmark group
//! cargo bench -- matmul
//! cargo bench -- transpose
//! cargo bench -- add
//! cargo bench -- mul_elementwise
//!
//! # Save baseline for comparison
//! cargo bench -- --save-baseline blas-baseline
//!
//! # Compare against baseline
//! cargo bench -- --baseline blas-baseline
//! ```
//!
//! ## Expected Results
//!
//! The BLAS backend should show:
//! - **10-100x faster** for large matrix multiplications
//! - **2-10x faster** for transpose operations
//! - **2-5x faster** for element-wise operations on large matrices
//!
//! These improvements are critical for training and inference in GPT models,
//! where matrix operations dominate computational time.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jwall_gpt::matrix::{naive::Naive, MatrixCompute};

#[cfg(feature = "blas")]
use jwall_gpt::matrix::blas::Blas;

// ============================================
// BENCHMARK CONFIGURATION
// ============================================

/// Matrix sizes to test, representing different computational scales
const MATRIX_SIZES: &[usize] = &[
    32,   // Small: quick operations, minimal cache effects
    256,  // Medium: moderate computational load
    1024, // Large: heavy computation, demonstrates BLAS advantage
];

// ============================================
// HELPER FUNCTIONS
// ============================================

/// Create a matrix filled with random-ish values for benchmarking
/// Uses a simple pattern to avoid overhead of actual random generation
fn create_test_matrix_naive(size: usize) -> <Naive as MatrixCompute>::Matrix {
    let mut matrix = Naive::zeros(size, size);
    for i in 0..size {
        for j in 0..size {
            // Simple pattern that creates non-trivial values
            Naive::set(&mut matrix, i, j, ((i + j) as f32) * 0.1);
        }
    }
    matrix
}

#[cfg(feature = "blas")]
fn create_test_matrix_blas(size: usize) -> <Blas as MatrixCompute>::Matrix {
    let mut matrix = Blas::zeros(size, size);
    for i in 0..size {
        for j in 0..size {
            Blas::set(&mut matrix, i, j, ((i + j) as f32) * 0.1);
        }
    }
    matrix
}

// ============================================
// MATRIX MULTIPLICATION BENCHMARKS
// ============================================

/// Benchmark matrix multiplication - the most critical operation for neural networks
///
/// Matrix multiplication dominates computational time in:
/// - Forward passes (X * W)
/// - Backward passes (gradient computation)
/// - Attention mechanisms (Q * K^T, attention * V)
///
/// BLAS should show 10-100x speedup for large matrices due to:
/// - Optimized memory access patterns
/// - SIMD vectorization
/// - Cache-aware blocking algorithms
/// - Multi-threaded execution (when enabled)
fn benchmark_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    for &size in MATRIX_SIZES {
        // Benchmark Naive implementation
        let a_naive = create_test_matrix_naive(size);
        let b_naive = create_test_matrix_naive(size);

        group.bench_with_input(
            BenchmarkId::new("naive", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = Naive::matmul(black_box(&a_naive), black_box(&b_naive));
                    black_box(result);
                });
            },
        );

        // Benchmark BLAS implementation (if available)
        #[cfg(feature = "blas")]
        {
            let a_blas = create_test_matrix_blas(size);
            let b_blas = create_test_matrix_blas(size);

            group.bench_with_input(
                BenchmarkId::new("blas", size),
                &size,
                |bencher, _| {
                    bencher.iter(|| {
                        let result = Blas::matmul(black_box(&a_blas), black_box(&b_blas));
                        black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================
// TRANSPOSE BENCHMARKS
// ============================================

/// Benchmark matrix transpose operation
///
/// Transpose is used in:
/// - Computing gradients (W^T * gradient)
/// - Attention mechanisms (computing K^T)
/// - Weight updates
///
/// BLAS should show 2-10x speedup due to optimized memory access patterns
fn benchmark_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose");

    for &size in MATRIX_SIZES {
        // Benchmark Naive implementation
        let matrix_naive = create_test_matrix_naive(size);

        group.bench_with_input(
            BenchmarkId::new("naive", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = Naive::transpose(black_box(&matrix_naive));
                    black_box(result);
                });
            },
        );

        // Benchmark BLAS implementation (if available)
        #[cfg(feature = "blas")]
        {
            let matrix_blas = create_test_matrix_blas(size);

            group.bench_with_input(
                BenchmarkId::new("blas", size),
                &size,
                |bencher, _| {
                    bencher.iter(|| {
                        let result = Blas::transpose(black_box(&matrix_blas));
                        black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================
// ELEMENT-WISE ADDITION BENCHMARKS
// ============================================

/// Benchmark element-wise matrix addition
///
/// Used in:
/// - Bias addition (X + b)
/// - Residual connections (X + residual)
/// - Combining gradients
///
/// BLAS should show 2-5x speedup due to SIMD vectorization
fn benchmark_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add");

    for &size in MATRIX_SIZES {
        // Benchmark Naive implementation
        let a_naive = create_test_matrix_naive(size);
        let b_naive = create_test_matrix_naive(size);

        group.bench_with_input(
            BenchmarkId::new("naive", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = Naive::add(black_box(&a_naive), black_box(&b_naive));
                    black_box(result);
                });
            },
        );

        // Benchmark BLAS implementation (if available)
        #[cfg(feature = "blas")]
        {
            let a_blas = create_test_matrix_blas(size);
            let b_blas = create_test_matrix_blas(size);

            group.bench_with_input(
                BenchmarkId::new("blas", size),
                &size,
                |bencher, _| {
                    bencher.iter(|| {
                        let result = Blas::add(black_box(&a_blas), black_box(&b_blas));
                        black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================
// ELEMENT-WISE MULTIPLICATION BENCHMARKS
// ============================================

/// Benchmark element-wise matrix multiplication
///
/// Used in:
/// - Applying masks (attention masks, dropout)
/// - Gating mechanisms (e.g., GeLU approximation)
/// - Scaling operations
///
/// BLAS should show 2-5x speedup due to SIMD vectorization
fn benchmark_mul_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_elementwise");

    for &size in MATRIX_SIZES {
        // Benchmark Naive implementation
        let a_naive = create_test_matrix_naive(size);
        let b_naive = create_test_matrix_naive(size);

        group.bench_with_input(
            BenchmarkId::new("naive", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    let result = Naive::mul_elementwise(black_box(&a_naive), black_box(&b_naive));
                    black_box(result);
                });
            },
        );

        // Benchmark BLAS implementation (if available)
        #[cfg(feature = "blas")]
        {
            let a_blas = create_test_matrix_blas(size);
            let b_blas = create_test_matrix_blas(size);

            group.bench_with_input(
                BenchmarkId::new("blas", size),
                &size,
                |bencher, _| {
                    bencher.iter(|| {
                        let result =
                            Blas::mul_elementwise(black_box(&a_blas), black_box(&b_blas));
                        black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================
// CRITERION CONFIGURATION
// ============================================

criterion_group!(
    benches,
    benchmark_matmul,
    benchmark_transpose,
    benchmark_add,
    benchmark_mul_elementwise
);
criterion_main!(benches);
