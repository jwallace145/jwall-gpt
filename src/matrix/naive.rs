use super::MatrixCompute;

pub type Matrix = Vec<Vec<f32>>;

pub struct Naive;

impl MatrixCompute for Naive {
    type Matrix = Matrix;

    fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
        // Validate inputs
        assert!(!a.is_empty(), "Matrix A cannot be empty");
        assert!(!b.is_empty(), "Matrix B cannot be empty");
        assert!(!a[0].is_empty(), "Matrix A cannot have empty rows");
        assert!(!b[0].is_empty(), "Matrix B cannot have empty rows");

        let rows_a = a.len();
        let cols_a = a[0].len();
        let rows_b = b.len();
        let cols_b = b[0].len();

        // Check dimension compatibility
        assert_eq!(
            cols_a, rows_b,
            "Matrix dimensions incompatible for multiplication: ({}, {}) × ({}, {})",
            rows_a, cols_a, rows_b, cols_b
        );

        // Pre-allocate result matrix
        let mut result = vec![vec![0.0; cols_b]; rows_a];

        // Optimized loop order: i, k, j for better cache performance
        // This accesses both a and b in row-major order
        for i in 0..rows_a {
            for k in 0..cols_a {
                let a_ik = a[i][k]; // Cache this value
                for j in 0..cols_b {
                    result[i][j] += a_ik * b[k][j];
                }
            }
        }

        result
    }

    fn zeros(rows: usize, cols: usize) -> Self::Matrix {
        vec![vec![0.0; cols]; rows]
    }

    fn ones(rows: usize, cols: usize) -> Self::Matrix {
        vec![vec![1.0; cols]; rows]
    }

    fn identity(size: usize) -> Self::Matrix {
        let mut result = Self::zeros(size, size);
        for i in 0..size {
            result[i][i] = 1.0;
        }
        result
    }

    fn shape(m: &Self::Matrix) -> (usize, usize) {
        if m.is_empty() {
            (0, 0)
        } else {
            (m.len(), m[0].len())
        }
    }

    fn transpose(m: &Self::Matrix) -> Self::Matrix {
        let (rows, cols) = Self::shape(m);

        // Handle empty matrix
        if rows == 0 || cols == 0 {
            return vec![];
        }

        // Create result matrix with swapped dimensions
        let mut result = Self::zeros(cols, rows);

        // Swap rows and columns
        for i in 0..rows {
            for j in 0..cols {
                result[j][i] = m[i][j];
            }
        }

        result
    }

    fn add(a: &Self::Matrix, b: &Self::Matrix) -> Self::Matrix {
        let (rows, cols) = Self::shape(a);

        assert_eq!(
            Self::shape(a),
            Self::shape(b),
            "Matrix dimensions must match for addition: {:?} vs {:?}",
            Self::shape(a),
            Self::shape(b)
        );

        let mut result = Self::zeros(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                result[i][j] = a[i][j] + b[i][j];
            }
        }

        result
    }

    fn mul_elementwise(a: &Self::Matrix, b: &Self::Matrix) -> Self::Matrix {
        let (rows, cols) = Self::shape(a);

        assert_eq!(
            Self::shape(a),
            Self::shape(b),
            "Matrix dimensions must match for element-wise multiplication: {:?} vs {:?}",
            Self::shape(a),
            Self::shape(b)
        );

        let mut result = Self::zeros(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                result[i][j] = a[i][j] * b[i][j];
            }
        }

        result
    }

    fn get(m: &Self::Matrix, row: usize, col: usize) -> f32 {
        m[row][col]
    }

    fn set(m: &mut Self::Matrix, row: usize, col: usize, val: f32) {
        m[row][col] = val;
    }
}

/// Helper function: Get matrix dimensions (rows, cols)
#[inline]
pub fn shape(m: &Matrix) -> (usize, usize) {
    if m.is_empty() {
        (0, 0)
    } else {
        (m.len(), m[0].len())
    }
}

/// Create a matrix filled with zeros
pub fn zeros(rows: usize, cols: usize) -> Matrix {
    vec![vec![0.0; cols]; rows]
}

/// Create a matrix filled with ones
pub fn ones(rows: usize, cols: usize) -> Matrix {
    vec![vec![1.0; cols]; rows]
}

/// Create an identity matrix (ones on diagonal, zeros elsewhere)
pub fn identity(size: usize) -> Matrix {
    let mut result = zeros(size, size);
    for i in 0..size {
        result[i][i] = 1.0;
    }
    result
}

// ============================================
// TESTS
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to compare floats with tolerance
    fn assert_float_eq(a: f32, b: f32, tolerance: f32) {
        assert!(
            (a - b).abs() < tolerance,
            "Expected {}, got {} (difference: {})",
            b,
            a,
            (a - b).abs()
        );
    }

    // Helper to compare matrices
    fn assert_matrix_eq(a: &Matrix, b: &Matrix, tolerance: f32) {
        assert_eq!(
            Naive::shape(a),
            Naive::shape(b),
            "Matrix shapes don't match"
        );
        let (rows, cols) = Naive::shape(a);

        for i in 0..rows {
            for j in 0..cols {
                assert_float_eq(a[i][j], b[i][j], tolerance);
            }
        }
    }

    // ========================================
    // MATRIX MULTIPLICATION TESTS
    // ========================================

    #[test]
    fn test_matmul_simple_2x2() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

        let result = Naive::matmul(&a, &b);

        // Expected:
        // [1*5 + 2*7,  1*6 + 2*8]   [19, 22]
        // [3*5 + 4*7,  3*6 + 4*8] = [43, 50]
        assert_eq!(result[0][0], 19.0);
        assert_eq!(result[0][1], 22.0);
        assert_eq!(result[1][0], 43.0);
        assert_eq!(result[1][1], 50.0);
    }

    #[test]
    fn test_matmul_single_element() {
        let a = vec![vec![3.0]];
        let b = vec![vec![4.0]];
        let result = Naive::matmul(&a, &b);

        assert_eq!(result[0][0], 12.0);
    }

    #[test]
    fn test_matmul_non_square() {
        // (2×3) × (3×2) = (2×2)
        let a = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let b = vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]];

        let result = Naive::matmul(&a, &b);

        assert_eq!(Naive::shape(&result), (2, 2));
        assert_eq!(result[0][0], 58.0); // 1*7 + 2*9 + 3*11
        assert_eq!(result[0][1], 64.0); // 1*8 + 2*10 + 3*12
        assert_eq!(result[1][0], 139.0); // 4*7 + 5*9 + 6*11
        assert_eq!(result[1][1], 154.0); // 4*8 + 5*10 + 6*12
    }

    #[test]
    fn test_matmul_identity() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let identity = Naive::identity(2);

        let result = Naive::matmul(&a, &identity);
        assert_matrix_eq(&result, &a, 0.0001);
    }

    #[test]
    #[should_panic(expected = "Matrix dimensions incompatible")]
    fn test_matmul_incompatible_dimensions() {
        let a = vec![vec![1.0, 2.0]]; // 1×2
        let b = vec![vec![3.0, 4.0]]; // 1×2 (incompatible!)
        Naive::matmul(&a, &b);
    }

    #[test]
    #[should_panic(expected = "Matrix A cannot be empty")]
    fn test_matmul_empty_a() {
        let a: Matrix = vec![];
        let b = vec![vec![1.0]];
        Naive::matmul(&a, &b);
    }

    // ========================================
    // BASIC OPERATIONS TESTS
    // ========================================

    #[test]
    fn test_zeros() {
        let m = Naive::zeros(2, 3);
        assert_eq!(Naive::shape(&m), (2, 3));

        for row in &m {
            for &val in row {
                assert_eq!(val, 0.0);
            }
        }
    }

    #[test]
    fn test_ones() {
        let m = Naive::ones(2, 3);
        assert_eq!(Naive::shape(&m), (2, 3));

        for row in &m {
            for &val in row {
                assert_eq!(val, 1.0);
            }
        }
    }

    #[test]
    fn test_identity() {
        let id = Naive::identity(3);

        assert_eq!(Naive::shape(&id), (3, 3));
        assert_eq!(id[0][0], 1.0);
        assert_eq!(id[1][1], 1.0);
        assert_eq!(id[2][2], 1.0);
        assert_eq!(id[0][1], 0.0);
        assert_eq!(id[1][0], 0.0);
    }

    #[test]
    fn test_transpose() {
        let m = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let mt = Naive::transpose(&m);

        assert_eq!(Naive::shape(&mt), (3, 2));
        assert_eq!(mt[0][0], 1.0);
        assert_eq!(mt[0][1], 4.0);
        assert_eq!(mt[1][0], 2.0);
        assert_eq!(mt[1][1], 5.0);
        assert_eq!(mt[2][0], 3.0);
        assert_eq!(mt[2][1], 6.0);
    }

    #[test]
    fn test_add() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

        let c = Naive::add(&a, &b);

        assert_eq!(c[0][0], 6.0);
        assert_eq!(c[0][1], 8.0);
        assert_eq!(c[1][0], 10.0);
        assert_eq!(c[1][1], 12.0);
    }

    #[test]
    fn test_mul_elementwise() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

        let c = Naive::mul_elementwise(&a, &b);

        assert_eq!(c[0][0], 5.0); // 1*5
        assert_eq!(c[0][1], 12.0); // 2*6
        assert_eq!(c[1][0], 21.0); // 3*7
        assert_eq!(c[1][1], 32.0); // 4*8
    }

    #[test]
    fn test_get_set() {
        let mut m = Naive::zeros(2, 2);

        Naive::set(&mut m, 0, 1, 42.0);
        assert_eq!(Naive::get(&m, 0, 1), 42.0);

        Naive::set(&mut m, 1, 0, 3.14);
        assert_eq!(Naive::get(&m, 1, 0), 3.14);
    }

    #[test]
    fn test_shape_empty() {
        let m: Matrix = vec![];
        assert_eq!(Naive::shape(&m), (0, 0));
    }
}
