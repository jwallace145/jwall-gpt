# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`jwall-gpt` is an educational Rust implementation of GPT (Generative Pre-trained Transformer) from scratch. The project prioritizes **readability and learning** over performance, with optional BLAS acceleration available via feature flags.

## Build Commands

```bash
# Build with default features (includes BLAS optimization)
cargo build

# Build without BLAS (pure Rust, educational)
cargo build --no-default-features

# Run the project
cargo run

# Run all tests
cargo test

# Run tests for a specific module
cargo test activation
cargo test matrix
cargo test matrix::naive
cargo test matrix::blas

# Build and view documentation
cargo doc --open
```

## Architecture

### Matrix Computation Backend System

The codebase uses a **trait-based abstraction** for matrix operations, allowing runtime selection between two backends:

- **`MatrixCompute` trait** (`src/matrix/mod.rs`): Defines the interface for all matrix operations
- **`Naive` backend** (`src/matrix/naive.rs`): Pure Rust implementation using `Vec<Vec<f32>>`, optimized for educational clarity
- **`Blas` backend** (`src/matrix/blas.rs`): High-performance implementation using `ndarray` with OpenBLAS, enabled via the `blas` feature

**Key Pattern**: The `DefaultMatrixCompute` type alias automatically selects the appropriate backend at compile time based on feature flags. All convenience functions in `src/matrix/mod.rs` delegate to this default backend.

**When working with matrix operations**:
- Add new operations to the `MatrixCompute` trait first
- Implement for both `Naive` (educational, explicit) and `Blas` (optimized)
- Add convenience functions that use `DefaultMatrixCompute`
- Write tests that work with both backends

### Module Structure

- **`src/activation.rs`**: Neural network activation functions (ReLU, GELU, Sigmoid). GELU is the primary activation used in GPT models.
- **`src/layer_norm.rs`**: Layer normalization utilities (mean, variance)
- **`src/matrix/`**: Matrix computation abstraction layer with dual backend support
- **`src/lib.rs`**: Public module exports
- **`src/main.rs`**: Entry point (currently minimal)

### Feature Flags

- `default = ["blas"]`: By default, use optimized BLAS backend
- `blas`: Enables `ndarray`, `blas-src`, and `openblas-src` dependencies

To test both backends, run:
```bash
cargo test                        # Tests with BLAS
cargo test --no-default-features  # Tests without BLAS (naive)
```

## Development Notes

- **Edition**: Uses `edition = "2024"` (cutting edge, may require nightly Rust)
- **Testing Philosophy**: Comprehensive tests with helper functions for float comparison (`assert_float_eq`) and matrix comparison (`assert_matrix_eq`)
- **Documentation**: All modules include detailed rustdoc comments explaining purpose and mathematical context
- The project is in early stages - `main.rs` currently just prints "Hello, world!"
