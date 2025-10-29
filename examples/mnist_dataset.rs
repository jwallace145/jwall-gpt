//! Example: Loading and Using the MNIST Dataset
//!
//! This example demonstrates how to load the MNIST dataset and inspect its properties.
//!
//! Run with: `cargo run --example mnist_dataset --no-default-features`

use jwall_gpt::datasets::{Dataset, MnistDataset};
use jwall_gpt::matrix::MatrixExt;

fn main() {
    println!("Loading MNIST dataset...");
    println!("(This will download ~10MB on first run)\n");

    // Load the MNIST dataset
    let mnist = match MnistDataset::new() {
        Ok(dataset) => dataset,
        Err(e) => {
            eprintln!("Failed to load MNIST dataset: {}", e);
            eprintln!("Make sure you have an internet connection for the first download.");
            return;
        }
    };

    // Display dataset information
    println!("✓ MNIST dataset loaded successfully!");
    println!("\nDataset Information:");
    println!("  Training samples:   {:>6}", mnist.train_size());
    println!("  Test samples:       {:>6}", mnist.test_size());
    println!(
        "  Input features:     {:>6} (28×28 pixels)",
        mnist.input_size()
    );
    println!(
        "  Number of classes:  {:>6} (digits 0-9)",
        mnist.num_classes()
    );

    // Get training and test data
    let (train_images, train_labels) = mnist.train_data();
    let (test_images, test_labels) = mnist.test_data();

    println!("\nTraining Data:");
    println!(
        "  Images shape:  [{}, {}]",
        MatrixExt::rows(train_images),
        MatrixExt::cols(train_images)
    );
    println!(
        "  Labels shape:  [{}, {}]",
        MatrixExt::rows(train_labels),
        MatrixExt::cols(train_labels)
    );

    println!("\nTest Data:");
    println!(
        "  Images shape:  [{}, {}]",
        MatrixExt::rows(test_images),
        MatrixExt::cols(test_images)
    );
    println!(
        "  Labels shape:  [{}, {}]",
        MatrixExt::rows(test_labels),
        MatrixExt::cols(test_labels)
    );

    // Demonstrate working with a subset
    println!("\n--- Working with Subsets ---");
    let (small_train_x, _small_train_y) = mnist.train_subset(100);
    println!(
        "Created training subset: {} samples",
        MatrixExt::rows(&small_train_x)
    );

    // Inspect the first training sample
    println!("\n--- First Training Sample ---");

    #[cfg(feature = "blas")]
    {
        use ndarray::Axis;
        let first_image = train_images.index_axis(Axis(0), 0);
        let first_label = train_labels.index_axis(Axis(0), 0);

        // Find which digit this is (find the index with 1.0 in one-hot encoding)
        let digit = first_label.iter().position(|&x| x > 0.5).unwrap_or(0);

        println!("Label (one-hot): {:?}", first_label);
        println!("Digit: {}", digit);

        // Show pixel value statistics
        let min_pixel = first_image.iter().fold(f32::INFINITY, |a, b| a.min(*b));
        let max_pixel = first_image.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
        let avg_pixel = first_image.iter().sum::<f32>() / first_image.len() as f32;

        println!("\nPixel values (normalized):");
        println!("  Min: {:.3}", min_pixel);
        println!("  Max: {:.3}", max_pixel);
        println!("  Avg: {:.3}", avg_pixel);

        // Show a simple ASCII visualization of the first digit
        println!("\nASCII Visualization (28×28):");
        for i in 0..28 {
            for j in 0..28 {
                let pixel = first_image[i * 28 + j];
                let char = if pixel > 0.5 {
                    '█'
                } else if pixel > 0.25 {
                    '▓'
                } else if pixel > 0.1 {
                    '░'
                } else {
                    ' '
                };
                print!("{}", char);
            }
            println!();
        }
    }

    #[cfg(not(feature = "blas"))]
    {
        let first_image = &train_images[0];
        let first_label = &train_labels[0];

        // Find which digit this is (find the index with 1.0 in one-hot encoding)
        let digit = first_label.iter().position(|&x| x > 0.5).unwrap_or(0);

        println!("Label (one-hot): {:?}", first_label);
        println!("Digit: {}", digit);

        // Show pixel value statistics
        let min_pixel = first_image.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_pixel = first_image.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let avg_pixel = first_image.iter().sum::<f32>() / first_image.len() as f32;

        println!("\nPixel values (normalized):");
        println!("  Min: {:.3}", min_pixel);
        println!("  Max: {:.3}", max_pixel);
        println!("  Avg: {:.3}", avg_pixel);

        // Show a simple ASCII visualization of the first digit
        println!("\nASCII Visualization (28×28):");
        for i in 0..28 {
            for j in 0..28 {
                let pixel = first_image[i * 28 + j];
                let char = if pixel > 0.5 {
                    '█'
                } else if pixel > 0.25 {
                    '▓'
                } else if pixel > 0.1 {
                    '░'
                } else {
                    ' '
                };
                print!("{}", char);
            }
            println!();
        }
    }

    println!("\n✓ Example completed successfully!");
}
