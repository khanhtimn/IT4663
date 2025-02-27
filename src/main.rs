use nalgebra::*;

#[derive(Debug)]
struct Simplex {
    /// Vecto he so ham muc tieu
    c: DVector<f64>,
    /// Ma tran rang buoc
    A: DMatrix<f64>,
    /// Vecto rang buoc (ve phai)
    b: DVector<f64>,
}

impl Simplex {
    fn new(c: DVector<f64>, A: DMatrix<f64>, b: DVector<f64>) -> Self {
        Simplex { c, A, b }
    }

    fn create_tableau(&self) -> DMatrix<f64> {
        // Lap bang don hinh + bien phu + cot z + RHS
        #[allow(clippy::toplevel_ref_arg)]
        let tableau = stack![
            self.A, // Ma tran rang buoc
            DMatrix::identity(self.A.nrows(), self.A.nrows()), // Bien phu -> Ma tran don vi
            DMatrix::zeros(self.A.nrows(), 1), // Cot z -> full 0
            self.b; // Ve phai
        ];

        // Tao hang cuoi
        #[allow(clippy::toplevel_ref_arg)]
        let cost = stack![
            // Dao dau ham muc tieu
            RowDVector::from_vec(self.c.iter().map(|&x| -x).collect::<Vec<f64>>()),
            // Dat cac cot bien phu thanh 0
            RowDVector::from_vec(vec![0.0; self.A.nrows()]),
            // Dat gia tri z = 1 va ve phai = 0
            RowDVector::from_vec(Vec::from([1.0, 0.0]));
        ];

        // Them hang vao bang
        #[allow(clippy::toplevel_ref_arg)]
        let tableau = stack![
            tableau;
            cost;
        ];

        tableau
    }

    fn iterate(&self, tableau: &DMatrix<f64>) -> (DMatrix<f64>, DVector<f64>) {
        todo!();
    }

    fn find_pivot_column(&self, tableau: &DMatrix<f64>) -> usize {
        todo!();
    }

    fn find_pivot_row(&self, tableau: &DMatrix<f64>, pivot_col: &usize) -> usize {
        todo!();
    }

    fn pivot(
        &self,
        tableau: &DMatrix<f64>,
        pivot_row: usize,
        pivot_col: usize,
    ) -> (DMatrix<f64>, DVector<f64>) {
        todo!();
    }
}

fn main() {
    // Xet bai toan quy hoach tuyen tinh dang chuan
    // z = 3x1 + 2x2 -> max
    // 2x1 + x2 <= 7
    // x1 + 2x2 <= 8
    // x1 - x2 <= 2

    let c = DVector::from_vec(vec![3.0, 2.0]);
    let A = DMatrix::from_row_slice(
        3,
        2,
        &[
            2.0, 1.0, // 2x1 + x2 <= 7
            1.0, 2.0, // x1 + 2x2 <= 8
            1.0, -1.0, // x1 - x2 <= 2
        ],
    );
    let b = DVector::from_vec(vec![7.0, 8.0, 2.0]);

    let simplex = Simplex::new(c, A, b);

    let tableau = simplex.create_tableau();
    println!("Bang don hinh: {}", tableau);
}
