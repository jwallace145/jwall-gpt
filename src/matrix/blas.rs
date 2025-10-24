use super::MatrixCompute;
use ndarray::Array2;

pub type Matrix = Array2<f32>;

pub struct Blas;

impl MatrixCompute for Blas {
    type Matrix = Matrix;

    #[inline]
    fn matmul(a: &Self::Matrix, b: &Self::Matrix) -> Self::Matrix {
        a.dot(b)
    }

    #[inline]
    fn zeros(rows: usize, cols: usize) -> Self::Matrix {
        Array2::zeros((rows, cols))
    }

    #[inline]
    fn ones(rows: usize, cols: usize) -> Self::Matrix {
        Array2::ones((rows, cols))
    }

    #[inline]
    fn identity(size: usize) -> Self::Matrix {
        Array2::eye(size)
    }

    #[inline]
    fn shape(m: &Self::Matrix) -> (usize, usize) {
        m.dim()
    }

    #[inline]
    fn transpose(m: &Self::Matrix) -> Self::Matrix {
        m.t().to_owned()
    }

    #[inline]
    fn add(a: &Self::Matrix, b: &Self::Matrix) -> Self::Matrix {
        a + b
    }

    #[inline]
    fn mul_elementwise(a: &Self::Matrix, b: &Self::Matrix) -> Self::Matrix {
        a * b
    }

    #[inline]
    fn get(m: &Self::Matrix, row: usize, col: usize) -> f32 {
        m[[row, col]]
    }

    #[inline]
    fn set(m: &mut Self::Matrix, row: usize, col: usize, val: f32) {
        m[[row, col]] = val;
    }
}

/// Convert from Vec<Vec<f32>> to ndarray
pub fn from_vecs(data: Vec<Vec<f32>>) -> Matrix {
    let rows = data.len();
    if rows == 0 {
        return Array2::zeros((0, 0));
    }
    let cols = data[0].len();
    let flat: Vec<f32> = data.into_iter().flatten().collect();
    Array2::from_shape_vec((rows, cols), flat).expect("Invalid shape")
}

/// Convert from ndarray to Vec<Vec<f32>>
pub fn to_vecs(matrix: &Matrix) -> Vec<Vec<f32>> {
    matrix.outer_iter().map(|row| row.to_vec()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn assert_float_eq(a: f32, b: f32, tolerance: f32) {
        assert!(
            (a - b).abs() < tolerance,
            "Expected {}, got {} (difference: {})",
            b,
            a,
            (a - b).abs()
        );
    }

    #[test]
    fn test_matmul_simple() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let c = Blas::matmul(&a, &b);

        assert_float_eq(c[[0, 0]], 19.0, 0.001);
        assert_float_eq(c[[0, 1]], 22.0, 0.001);
        assert_float_eq(c[[1, 0]], 43.0, 0.001);
        assert_float_eq(c[[1, 1]], 50.0, 0.001);
    }

    #[test]
    fn test_zeros() {
        let m = Blas::zeros(2, 3);
        assert_eq!(Blas::shape(&m), (2, 3));
        assert_float_eq(m[[0, 0]], 0.0, 0.001);
        assert_float_eq(m[[1, 2]], 0.0, 0.001);
    }

    #[test]
    fn test_identity() {
        let id = Blas::identity(3);
        assert_eq!(Blas::shape(&id), (3, 3));
        assert_float_eq(id[[0, 0]], 1.0, 0.001);
        assert_float_eq(id[[1, 1]], 1.0, 0.001);
        assert_float_eq(id[[2, 2]], 1.0, 0.001);
        assert_float_eq(id[[0, 1]], 0.0, 0.001);
    }

    #[test]
    fn test_transpose() {
        let m = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let mt = Blas::transpose(&m);

        assert_eq!(Blas::shape(&mt), (3, 2));
        assert_float_eq(mt[[0, 0]], 1.0, 0.001);
        assert_float_eq(mt[[0, 1]], 4.0, 0.001);
        assert_float_eq(mt[[2, 1]], 6.0, 0.001);
    }

    #[test]
    fn test_add() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let c = Blas::add(&a, &b);

        assert_float_eq(c[[0, 0]], 6.0, 0.001);
        assert_float_eq(c[[1, 1]], 12.0, 0.001);
    }

    #[test]
    fn test_conversion() {
        let vec_data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let matrix = from_vecs(vec_data.clone());
        let back = to_vecs(&matrix);

        assert_eq!(vec_data, back);
    }

    #[test]
    fn test_large_matrix() {
        // This is fast with BLAS!
        let a = Blas::ones(500, 500);
        let b = Blas::ones(500, 500);

        let c = Blas::matmul(&a, &b);

        // Each element should be 500 (sum of 500 1.0s)
        assert_float_eq(c[[0, 0]], 500.0, 0.01);
    }
}
