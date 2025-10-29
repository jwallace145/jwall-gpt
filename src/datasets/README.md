# Datasets Module

This module provides a unified interface for loading and managing datasets for neural network training.

## Features

- **Trait-based design**: Easy to add new datasets
- **MNIST included**: Handwritten digits dataset with automatic download
- **Matrix integration**: Works seamlessly with both BLAS and naive backends
- **Helper utilities**: Normalization and one-hot encoding functions

## Quick Start

### Loading MNIST

```rust
use jwall_gpt::datasets::{Dataset, MnistDataset};

// Load MNIST (downloads automatically if needed)
let mnist = MnistDataset::new().expect("Failed to load MNIST");

// Get full training and test sets
let (train_images, train_labels) = mnist.train_data();
let (test_images, test_labels) = mnist.test_data();

println!("Training samples: {}", mnist.train_size());   // 60,000
println!("Test samples: {}", mnist.test_size());       // 10,000
println!("Input features: {}", mnist.input_size());    // 784 (28x28)
println!("Classes: {}", mnist.num_classes());          // 10 (digits 0-9)
```

### Using Subsets for Quick Experiments

```rust
// Work with smaller subsets for faster experimentation
let (train_x, train_y) = mnist.train_subset(1000);  // Just 1000 samples
let (test_x, test_y) = mnist.test_subset(100);      // Just 100 samples
```

## Dataset Trait

All datasets implement the `Dataset` trait:

```rust
pub trait Dataset {
    fn train_data(&self) -> (&Matrix, &Matrix);
    fn test_data(&self) -> (&Matrix, &Matrix);
    fn num_classes(&self) -> usize;
    fn input_size(&self) -> usize;
    fn train_size(&self) -> usize;
    fn test_size(&self) -> usize;
}
```

## Adding a Custom Dataset

To add your own dataset:

1. Implement the `Dataset` trait
2. Use helper functions for common operations:
   - `normalize_images()` - Normalize pixel values to [0, 1]
   - `labels_to_one_hot()` - Convert class indices to one-hot encoding

### Example Custom Dataset

```rust
use jwall_gpt::datasets::{Dataset, normalize_images, labels_to_one_hot};
use jwall_gpt::matrix::{Matrix, MatrixExt};

pub struct MyDataset {
    train_images: Matrix,
    train_labels: Matrix,
    test_images: Matrix,
    test_labels: Matrix,
}

impl MyDataset {
    pub fn new() -> Self {
        // Load your data
        let raw_train_images: Vec<u8> = load_train_images();
        let raw_train_labels: Vec<u8> = load_train_labels();

        // Normalize and convert to matrices
        let normalized = normalize_images(&raw_train_images);
        let train_images = create_matrix(normalized, num_samples, feature_size);

        let one_hot = labels_to_one_hot(&raw_train_labels, num_classes);
        let train_labels = create_matrix(one_hot, num_samples, num_classes);

        // ... similar for test data

        MyDataset {
            train_images,
            train_labels,
            test_images,
            test_labels,
        }
    }
}

impl Dataset for MyDataset {
    fn train_data(&self) -> (&Matrix, &Matrix) {
        (&self.train_images, &self.train_labels)
    }

    fn test_data(&self) -> (&Matrix, &Matrix) {
        (&self.test_images, &self.test_labels)
    }

    fn num_classes(&self) -> usize {
        10  // or your number of classes
    }

    fn input_size(&self) -> usize {
        784  // or your feature size
    }
}
```

## Data Format

All datasets use these conventions:

- **Images/Features**: Matrix of shape `[num_samples, input_size]`
  - Normalized to range [0, 1]
  - Row-major order (each row is one sample)

- **Labels**: Matrix of shape `[num_samples, num_classes]`
  - One-hot encoded (1.0 for correct class, 0.0 elsewhere)
  - Each row sums to 1.0

## MNIST Details

The MNIST dataset contains:
- **60,000** training images
- **10,000** test images
- **28×28** pixel grayscale images (flattened to 784 features)
- **10 classes** (digits 0-9)

### Downloading MNIST

**Note:** The mnist crate's automatic download uses outdated URLs. You need to download the files manually:

```bash
mkdir -p data && cd data
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
cd ..
```

The files will be downloaded to the `data/` directory in your project (~53MB total).

## Performance Tips

1. **Use subsets for development**: Start with `train_subset(1000)` for quick iteration
2. **Cache datasets**: Load once, reuse throughout your session
3. **Both backends work**: Works with both `--features blas` (fast) and `--no-default-features` (educational)

## Future Datasets

Planned additions:
- CIFAR-10 (color images)
- Fashion-MNIST (clothing items)
- Custom text datasets for GPT training
