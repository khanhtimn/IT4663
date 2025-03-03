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
                .filter(|&(_, &val)| val.abs() > 1e-10)
                .collect::<Vec<_>>();

            println!("{:?}", basic_row);

            // Neu co chinh xac mot so 1 va con lai la so 0 (Tinh ca phan gan dung)
            if basic_row.len() == 1 && (basic_row[0].1 - 1.0).abs() < 1e-10 {
                let row = basic_row[0].0;
                solution[column] = self.tableau[(row, self.tableau.ncols() - 2)];
            }
        }

        let c_value = self.tableau[(self.tableau.nrows() - 1, self.tableau.ncols() - 2)];

        Some((c_value, solution))
    }
}
