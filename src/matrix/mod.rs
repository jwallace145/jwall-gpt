#![allow(dead_code, unused_imports)]

pub mod naive;

#[cfg(feature = "blas")]
pub mod blas;

pub trait MatrixCompute {
    type Matrix;

    fn matmul(a: &Self::Matrix, b: &Self::Matrix) -> Self::Matrix;
    fn zeros(rows: usize, cols: usize) -> Self::Matrix;
    fn ones(rows: usize, cols: usize) -> Self::Matrix;
    fn identity(size: usize) -> Self::Matrix;
    fn shape(m: &Self::Matrix) -> (usize, usize);
    fn transpose(m: &Self::Matrix) -> Self::Matrix;
    fn add(a: &Self::Matrix, b: &Self::Matrix) -> Self::Matrix;
    fn mul_elementwise(a: &Self::Matrix, b: &Self::Matrix) -> Self::Matrix;
    fn get(m: &Self::Matrix, row: usize, col: usize) -> f32;
    fn set(m: &mut Self::Matrix, row: usize, col: usize, val: f32);
}

pub use naive::Naive;

#[cfg(feature = "blas")]
pub use blas::Blas;

#[cfg(feature = "blas")]
pub type DefaultMatrixCompute = Blas;

#[cfg(not(feature = "blas"))]
pub type DefaultMatrixCompute = Naive;

pub type Matrix = <DefaultMatrixCompute as MatrixCompute>::Matrix;

// Convenience functions that use the default backend and trait
pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    DefaultMatrixCompute::matmul(a, b)
}

pub fn zeros_matrix(rows: usize, cols: usize) -> Matrix {
    MatrixExt::from_vec(vec![vec![0.0; cols]; rows])
}

pub fn ones_matrix(rows: usize, cols: usize) -> Matrix {
    MatrixExt::from_vec(vec![vec![1.0; cols]; rows])
}

pub fn zeros(rows: usize, cols: usize) -> Matrix {
    zeros_matrix(rows, cols)
}

pub fn ones(rows: usize, cols: usize) -> Matrix {
    ones_matrix(rows, cols)
}

pub fn identity(size: usize) -> Matrix {
    DefaultMatrixCompute::identity(size)
}

pub fn shape(m: &Matrix) -> (usize, usize) {
    DefaultMatrixCompute::shape(m)
}

pub fn transpose(m: &Matrix) -> Matrix {
    DefaultMatrixCompute::transpose(m)
}

pub fn add(a: &Matrix, b: &Matrix) -> Matrix {
    DefaultMatrixCompute::add(a, b)
}

pub fn mul_elementwise(a: &Matrix, b: &Matrix) -> Matrix {
    DefaultMatrixCompute::mul_elementwise(a, b)
}

pub fn get(m: &Matrix, row: usize, col: usize) -> f32 {
    DefaultMatrixCompute::get(m, row, col)
}

pub fn set(m: &mut Matrix, row: usize, col: usize, val: f32) {
    DefaultMatrixCompute::set(m, row, col, val)
}

pub fn matrix_compute_name() -> &'static str {
    #[cfg(feature = "blas")]
    return "BLAS (optimized)";

    #[cfg(not(feature = "blas"))]
    return "Naive (educational)";
}

// ============================================
// MATRIX EXTENSION TRAIT
// ============================================

/// Extension trait providing common matrix operations
pub trait MatrixExt: Sized {
    fn from_vec(data: Vec<Vec<f32>>) -> Self;
    fn random(rows: usize, cols: usize, scale: f32) -> Self;
    fn matmul(&self, other: &Self) -> Self;
    fn transpose(&self) -> Self;
    fn add_broadcast(&self, bias: &Self) -> Self;
    fn sum_axis(&self, axis: usize) -> Self;
    fn fill_value(&mut self, value: f32);
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn shape(&self) -> (usize, usize);
    fn element_count(&self) -> usize;
    fn is_empty_matrix(&self) -> bool;
}

#[cfg(feature = "blas")]
impl MatrixExt for Matrix {
    fn from_vec(data: Vec<Vec<f32>>) -> Self {
        blas::from_vecs(data)
    }

    fn random(rows: usize, cols: usize, scale: f32) -> Self {
        use ndarray::{Dim, OwnedRepr};
        use ndarray_rand::rand_distr::Uniform;
        let range = scale * 3f32.sqrt();
        // Use fully qualified syntax to avoid ambiguity with MatrixExt::random
        <ndarray::Array2<f32> as ndarray_rand::RandomExt<OwnedRepr<f32>, f32, Dim<[usize; 2]>>>::random(
            (rows, cols),
            Uniform::new(-range, range)
        )
    }

    fn matmul(&self, other: &Self) -> Self {
        self.dot(other)
    }

    fn transpose(&self) -> Self {
        self.t().to_owned()
    }

    fn add_broadcast(&self, bias: &Self) -> Self {
        self + bias
    }

    fn sum_axis(&self, axis: usize) -> Self {
        use ndarray::Axis;
        self.sum_axis(Axis(axis)).insert_axis(Axis(0))
    }

    fn fill_value(&mut self, value: f32) {
        ndarray::ArrayBase::fill(self, value);
    }

    fn rows(&self) -> usize {
        self.nrows()
    }

    fn cols(&self) -> usize {
        self.ncols()
    }

    fn shape(&self) -> (usize, usize) {
        self.dim()
    }

    fn element_count(&self) -> usize {
        self.len()
    }

    fn is_empty_matrix(&self) -> bool {
        self.is_empty()
    }
}

#[cfg(not(feature = "blas"))]
impl MatrixExt for Matrix {
    fn from_vec(data: Vec<Vec<f32>>) -> Self {
        naive::Matrix::new(data)
    }

    fn random(rows: usize, cols: usize, scale: f32) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let range = scale * 3f32.sqrt();
        let data = (0..rows)
            .map(|_| (0..cols).map(|_| rng.gen_range(-range..range)).collect())
            .collect();
        naive::Matrix::new(data)
    }

    fn matmul(&self, other: &Self) -> Self {
        DefaultMatrixCompute::matmul(self, other)
    }

    fn transpose(&self) -> Self {
        DefaultMatrixCompute::transpose(self)
    }

    fn add_broadcast(&self, bias: &Self) -> Self {
        let mut result = self.clone();
        let rows = result.len();
        for i in 0..rows {
            let cols = result[i].len();
            for j in 0..cols {
                result[i][j] += bias[0][j];
            }
        }
        result
    }

    fn sum_axis(&self, axis: usize) -> Self {
        if axis == 0 {
            let cols = if self.is_empty() { 0 } else { self[0].len() };
            let mut result = vec![0.0; cols];
            for row in self.iter() {
                for j in 0..cols {
                    result[j] += row[j];
                }
            }
            naive::Matrix::new(vec![result])
        } else {
            let data = self
                .iter()
                .map(|row| vec![row.iter().sum::<f32>()])
                .collect();
            naive::Matrix::new(data)
        }
    }

    fn fill_value(&mut self, value: f32) {
        for row in self.iter_mut() {
            for elem in row.iter_mut() {
                *elem = value;
            }
        }
    }

    fn rows(&self) -> usize {
        self.len()
    }

    fn cols(&self) -> usize {
        if self.is_empty() { 0 } else { self[0].len() }
    }

    fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    fn element_count(&self) -> usize {
        self.iter().map(|row| row.len()).sum()
    }

    fn is_empty_matrix(&self) -> bool {
        self.is_empty()
    }
}

// ============================================
// TESTS
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_compute_selection() {
        let matrix_compute = matrix_compute_name();

        #[cfg(feature = "blas")]
        assert_eq!(matrix_compute, "BLAS (optimized)");

        #[cfg(not(feature = "blas"))]
        assert_eq!(matrix_compute, "Naive (educational)");
    }

    #[test]
    fn test_default_matrix_compute_works() {
        let a = zeros(2, 2);
        let b = ones(2, 2);
        let c = matmul(&a, &b);

        assert_eq!(shape(&c), (2, 2));
    }

    #[test]
    fn test_identity_property() {
        let a = ones(3, 3);
        let id = identity(3);
        let result = matmul(&a, &id);

        let (rows, cols) = shape(&result);
        assert_eq!(rows, 3);
        assert_eq!(cols, 3);
    }

    #[test]
    fn test_convenience_functions() {
        let m1 = zeros(2, 2);
        let m2 = ones(2, 2);

        let _ = add(&m1, &m2);
        let _ = mul_elementwise(&m2, &m2);
        let _ = transpose(&m2);

        let val = get(&m2, 0, 0);
        assert_eq!(val, 1.0);

        let mut m3 = zeros(2, 2);
        set(&mut m3, 0, 0, 42.0);
        assert_eq!(get(&m3, 0, 0), 42.0);
    }
}
