use approx::*;
use nalgebra::*;
use std::{cmp::Ordering, f64};

#[derive(Debug, Clone)]
pub struct Simplex {
    /// Vecto he so ham muc tieu
    pub c: DVector<f64>,
    /// Ma tran rang buoc
    pub tableau: DMatrix<f64>,
    /// Vecto rang buoc (ve phai)
    pub b: DVector<f64>,
}

impl Simplex {
    #[allow(non_snake_case)]
    pub fn new(c: DVector<f64>, A: DMatrix<f64>, b: DVector<f64>) -> Self {
        // Lap bang don hinh + bien phu + cot z + RHS
        // TODO: Test cot co so co san
        #[allow(clippy::toplevel_ref_arg)]
        let tableau = stack![
            A, DMatrix::identity(A.nrows(), A.nrows()), DVector::zeros(A.nrows()), b;
            RowDVector::from_vec(c.iter().map(|&x| -x).collect::<Vec<f64>>()), RowDVector::zeros(A.nrows()),
            RowDVector::from_element(1, 1.0), RowDVector::zeros(1);
        ];

        // Them cot uoc luong vao ve ben phai
        #[allow(clippy::toplevel_ref_arg)]
        let tableau = stack![
          tableau, DVector::zeros(A.nrows() + 1);
        ];

        Self { c, tableau, b }
    }

    pub fn find_pivot_column(&self) -> usize {
        self.tableau
            .row(self.tableau.nrows() - 1)
            .iter()
            .enumerate()
            .filter(|(_, x)| x.is_sign_negative())
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap_or_default()
    }

    pub fn find_pivot_row(&self) -> usize {
        self.tableau
            .column(self.tableau.ncols() - 1)
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap_or_default()
    }

    pub fn pivot(&mut self, pivot_row: usize, pivot_col: usize) {
        // Phan tu xoay
        let pivot_element = self.tableau[(pivot_row, pivot_col)];

        self.tableau
            .row_part_mut(pivot_row, self.tableau.ncols() - 1)
            .scale_mut(1.0 / pivot_element);

        for row in 0..self.tableau.nrows() {
            if row != pivot_row {
                let factor = self.tableau[(row, pivot_col)];

                for col in 0..self.tableau.ncols() - 1 {
                    let pivot_row_val = self.tableau[(pivot_row, col)];
                    let current_val = self.tableau.get_mut((row, col)).unwrap();
                    *current_val -= factor * pivot_row_val;
                }
            }
        }
    }

    pub fn update_ratio_column(&mut self, pivot_col: usize) {
        let ratio_column = self
            .tableau
            .column(self.tableau.ncols() - 2)
            .component_div(&self.tableau.column(pivot_col))
            .map(|x| {
                // TODO: <= moi chuan
                // Lien quan den bai toan khong thoai hoa
                if x.is_sign_negative() {
                    f64::INFINITY
                } else {
                    x
                }
            });

        self.tableau
            .set_column(self.tableau.ncols() - 1, &ratio_column);
    }

    pub fn iterate(&mut self) -> bool {
        // Neu hang cuoi cung khong am, ket thuc buoc lap (Tieu chuan toi uu)
        let last_row = self
            .tableau
            .row_part(self.tableau.nrows() - 1, self.tableau.ncols() - 2);
        if last_row.iter().all(|&x| x >= 0.0) {
            println!("Bai toan co patu");
            return false;
        }

        // Tim cot xoay (Cot co he so hang cuoi cung am va nho nhat)
        let pivot_column = self.find_pivot_column();

        // Update cot uoc luong
        self.update_ratio_column(pivot_column);

        // Chon hang co uoc luong nho nhat
        let pivot_row = self.find_pivot_row();

        // Neu co uong luong duong ma cac phan tu trong cot tuong ung deu khong duong
        // thi ham muc tieu khong bi chan duoi, ket thuc buoc lap (Dieu kien du)
        if self
            .tableau
            .column(self.tableau.ncols() - 1)
            .iter()
            .all(|&x| x.is_infinite())
        {
            println!("Bai toan bi chan");
            return false;
        }

        // Thuc hien xoay
        self.pivot(pivot_row, pivot_column);

        println!("{}", self.tableau);

        true
    }

    pub fn solve(&mut self) -> Option<(f64, DVector<f64>)> {
        // Thuc hien buoc lap cho den khi tim ra ket qua
        while self.iterate() {}

        let mut solution = DVector::zeros(self.c.len());

        // Kiem tra xem cot co phai bien co so khong (Cot ma chi co mot so 1 va con lai la 0)
        for column in 0..self.c.len() {
            let column_vector = self.tableau.column(column);
            // Bieu dien 1 tuple gom hang cua bien co so va
            // gia tri duong cua cac phan tu trong tung hang
            let basic_row = column_vector
                .iter()
                .enumerate()
                .filter(|&(_, &val)| val.abs().is_normal())
                .collect::<Vec<_>>();

            // Neu co chinh xac mot so 1 va con lai la so 0 (Tinh ca phan gan dung)
            if basic_row.len() == 1 && abs_diff_eq!(basic_row[0].1, &1.0) {
                let row = basic_row[0].0;
                solution[column] = self.tableau[(row, self.tableau.ncols() - 2)];
            }
        }

        let c_value = self.tableau[(self.tableau.nrows() - 1, self.tableau.ncols() - 2)];

        Some((c_value, solution))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tableau_construction() {
        let c = DVector::from_vec(vec![3.0, 2.0]);
        #[allow(non_snake_case)]
        let A = DMatrix::from_row_slice(3, 2, &[2.0, 1.0, 1.0, 2.0, 1.0, -1.0]);
        let b = DVector::from_vec(vec![7.0, 8.0, 2.0]);

        let simplex = Simplex::new(c, A, b);

        assert_eq!(simplex.tableau.nrows(), 4); // 3 hang rang buoc + 1 hang ham muc tieu
        assert_eq!(simplex.tableau.ncols(), 8); // 2 bien + 3 bien phu + 1 z + 1 RHS + 1 E
    }

    #[test]
    fn test_find_pivot_column() {
        let c = DVector::from_vec(vec![3.0, 2.0]);
        #[allow(non_snake_case)]
        let A = DMatrix::from_row_slice(3, 2, &[2.0, 1.0, 1.0, 2.0, 1.0, -1.0]);
        let b = DVector::from_vec(vec![7.0, 8.0, 2.0]);

        let simplex = Simplex::new(c, A, b);

        // He so ham muc tieu o trong bang duoc dao dau
        // vay nen -3 va -2 se xuat hien o hang cuoi cua bang don hinh
        let pivot_column = simplex.find_pivot_column();

        // Cot co he so ham muc tieu nho nhat se duoc chon
        // Trong truong hop nay la cot thu nhat
        assert_eq!(pivot_column, 0);
    }

    #[test]
    fn test_update_ratio_column() {
        let c = DVector::from_vec(vec![3.0, 2.0]);
        #[allow(non_snake_case)]
        let A = DMatrix::from_row_slice(3, 2, &[2.0, 1.0, 1.0, 2.0, 1.0, -1.0]);
        let b = DVector::from_vec(vec![7.0, 8.0, 2.0]);

        let mut simplex = Simplex::new(c, A, b);

        // Gia su cot xoay la cot 1
        let pivot_col = 0;
        simplex.update_ratio_column(pivot_col);

        // Kiem tra gia tri cot uoc luong
        // Voi 3 hang dau tien, ket qua la 7/2, 8/1, 2/1
        let expected_ratios = [3.5, 8.0, 2.0];

        for (i, &expected) in expected_ratios.iter().enumerate() {
            assert_abs_diff_eq!(simplex.tableau[(i, simplex.tableau.ncols() - 1)], expected);
        }
    }

    #[test]
    fn test_find_pivot_row() {
        let c = DVector::from_vec(vec![3.0, 2.0]);
        #[allow(non_snake_case)]
        let A = DMatrix::from_row_slice(3, 2, &[2.0, 1.0, 1.0, 2.0, 1.0, -1.0]);
        let b = DVector::from_vec(vec![7.0, 8.0, 2.0]);

        let mut simplex = Simplex::new(c, A, b);

        // Update cot uoc luong truoc
        simplex.update_ratio_column(0);

        // Tim hang xoay
        let pivot_row = simplex.find_pivot_row();

        // Hang xoay se la hang co he so cot uoc luong nho nhat
        // Trong truong hop nay la hang 3 voi E = 2/1
        assert_eq!(pivot_row, 2);
    }

    #[test]
    fn test_pivot_operation() {
        let c = DVector::from_vec(vec![3.0, 2.0]);
        #[allow(non_snake_case)]
        let A = DMatrix::from_row_slice(3, 2, &[2.0, 1.0, 1.0, 2.0, 1.0, -1.0]);
        let b = DVector::from_vec(vec![7.0, 8.0, 2.0]);

        let mut simplex = Simplex::new(c, A, b);

        // Ta biet phan tu xoay dau tien (hang 3, cot 1)
        let pivot_row = 2;
        let pivot_col = 0;

        simplex.pivot(pivot_row, pivot_col);

        // Sau khi bien doi, hang xoay se co phan tu xoay = 1.0
        assert_abs_diff_eq!(simplex.tableau[(pivot_row, pivot_col)], 1.0);

        // Cac phan tu khac trong cot xoay = 0.0
        for row in 0..simplex.tableau.nrows() {
            if row != pivot_row {
                assert_abs_diff_eq!(simplex.tableau[(row, pivot_col)], 0.0);
            }
        }
    }

    #[test]
    fn test_simplex_iteration() {
        let c = DVector::from_vec(vec![3.0, 2.0]);
        #[allow(non_snake_case)]
        let A = DMatrix::from_row_slice(3, 2, &[2.0, 1.0, 1.0, 2.0, 1.0, -1.0]);
        let b = DVector::from_vec(vec![7.0, 8.0, 2.0]);

        let mut simplex = Simplex::new(c, A, b);

        // Thuc hien 1 lan lap don hinh
        let continues = simplex.iterate();

        // Buoc lap se tiep tuc vi chua tim duoc loi giai
        assert!(continues);
    }

    #[test]
    fn test_solve_standard_lp() {
        // Xet bai toan quy hoach tuyen tinh dang chuan
        // z = 3x1 + 2x2 -> max
        // 2x1 + x2 <= 7
        // x1 + 2x2 <= 8
        // x1 - x2 <= 2
        let c = DVector::from_vec(vec![3.0, 2.0]);
        #[allow(non_snake_case)]
        let A = DMatrix::from_row_slice(3, 2, &[2.0, 1.0, 1.0, 2.0, 1.0, -1.0]);
        let b = DVector::from_vec(vec![7.0, 8.0, 2.0]);

        let mut simplex = Simplex::new(c, A, b);
        let result = simplex.solve();

        // Kiem tra su ton tai patu
        assert!(result.is_some());

        let (objective_value, solution) = result.unwrap();

        // (x1 = 2, x2 = 3)
        assert_abs_diff_eq!(solution[0], 2.0);
        assert_abs_diff_eq!(solution[1], 3.0);

        // z = 3*2 + 2*3 = 12
        assert_abs_diff_eq!(objective_value, 12.0);
    }

    #[test]
    fn test_solve_unbounded_lp() {
        // Xet bai toan quy hoach tuyen tinh dang chuan
        // Day la bai toan khong bi chan
        // z = 3x1 + 5x2 -> max
        // x1 - 2x2 <= 6
        // x1       <= 10
        //      x2  >= 1
        let c = DVector::from_vec(vec![3.0, 5.0]);
        #[allow(non_snake_case)]
        let A = DMatrix::from_row_slice(3, 2, &[1.0, -2.0, 1.0, 0.0, 0.0, -1.0]);
        let b = DVector::from_vec(vec![6.0, 10.0, -1.0]);

        let mut simplex = Simplex::new(c, A, b);
        // Kiem tra su ton tai patu
        let result = simplex.solve();
        assert!(result.is_some());
    }

    #[test]
    fn test_solve_degenerate_lp() {
        // Xet bai toan quy hoach tuyen tinh dang chuan
        // Day la bai toan thoai hoa
        // z = 2x1 + x2 -> max
        // 4x1 + 3x2 <= 12
        // 4x1 + x2  <= 8
        // 4x1 + 2x2 <= 8
        let c = DVector::from_vec(vec![2.0, 1.0]);
        #[allow(non_snake_case)]
        let A = DMatrix::from_row_slice(3, 2, &[4.0, 3.0, 4.0, 1.0, 4.0, 2.0]);
        let b = DVector::from_vec(vec![12.0, 8.0, 8.0]);

        let mut simplex = Simplex::new(c, A, b);
        let result = simplex.solve();

        // Kiem tra su ton tai patu
        assert!(result.is_some());

        let (objective_value, solution) = result.unwrap();

        // (x1 = 2, x2 = 0)
        assert_abs_diff_eq!(solution[0], 2.0);
        assert_abs_diff_eq!(solution[1], 0.0);

        // z = 2*2 + 1*0 = 4
        assert_abs_diff_eq!(objective_value, 4.0);
    }
}
