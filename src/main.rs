use nalgebra::*;
use simplex::*;

fn main() {
    // Xet bai toan quy hoach tuyen tinh dang chuan
    // z = 3x1 + 2x2 -> max
    // 2x1 + x2 <= 7
    // x1 + 2x2 <= 8
    // x1 - x2 <= 2

    // z = 3x1 + 2x2 -> max
    let c = DVector::from_vec(vec![3.0, 2.0]);

    #[allow(non_snake_case)]
    let A = DMatrix::from_row_slice(
        3,
        2,
        &[
            2.0, 1.0, // 2x1 + x2 <= 7
            1.0, 2.0, // x1 + 2x2 <= 8
            1.0, -1.0, // x1 - x2 <= 2
        ],
    );

    // b = [7, 8, 2]t
    let b = DVector::from_vec(vec![7.0, 8.0, 2.0]);

    // Xet bai toan quy hoach tuyen tinh dang chuan
    // Day la bai toan khong bi chan
    // z = 3x1 + 5x2 -> max
    // x1 - 2x2 <= 6
    // x1       <= 10
    //      x2  >= 1

    // z = 3x1 + 5x2 -> max
    // let c = DVector::from_vec(vec![3.0, 5.0]);

    // #[allow(non_snake_case)]
    // let A = DMatrix::from_row_slice(
    //     3,
    //     2,
    //     &[
    //         1.0, -2.0, // x1 - 2x2 <= 6
    //         1.0, 0.0, // x1       <= 10
    //         0.0, -1.0, //    - x2  <= -1
    //     ],
    // );

    // // // b = [6, 10, -1]t
    // let b = DVector::from_vec(vec![6.0, 10.0, -1.0]);

    // Xet bai toan quy hoach tuyen tinh dang chuan
    // Day la bai toan thoai hoa
    // z = 2x1 + x2 -> max
    // 4x1 + 3x2 <= 12
    // 4x1 + x2  <= 8
    // 4x1 + 2x2 <= 8

    // z = 2x1 + x2 -> max
    // let c = DVector::from_vec(vec![2.0, 1.0]);

    // #[allow(non_snake_case)]
    // let A = DMatrix::from_row_slice(
    //     3,
    //     2,
    //     &[
    //         4.0, 3.0, // 4x1 + 3x2 <= 12
    //         4.0, 1.0, // 4x1 + x2  <= 8
    //         4.0, 2.0, // 4x1 + 2x2 <= 8
    //     ],
    // );

    // // b = [12, 8, 8]t
    // let b = DVector::from_vec(vec![12.0, 8.0, 8.0]);

    let mut simplex = Simplex::new(c, A, b);

    // println!("Lan lap thu 1:");
    // println!("Bang don hinh: {}", simplex.tableau);

    // let pivot_column = simplex.find_pivot_column();
    // println!("Cot xoay: {}", pivot_column);

    // simplex.update_ratio_column(pivot_column);

    // let pivot_row = simplex.find_pivot_row();
    // println!("Hang xoay: {}", pivot_row);
    // println!(
    //     "Phan tu xoay: {}",
    //     simplex.tableau[(pivot_row, pivot_column)]
    // );

    // simplex.pivot(pivot_row, pivot_column);
    // println!("Bang don hinh sau khi xoay: {}", simplex.tableau);
    // println!("Ket thuc lan lap thu 1");
    // println!("////////////////////");

    // println!("Lan lap thu 2:");
    // println!("Bang don hinh: {}", simplex.tableau);

    // let pivot_column = simplex.find_pivot_column();
    // println!("Cot xoay: {}", pivot_column);

    // simplex.update_ratio_column(pivot_column);

    // let pivot_row = simplex.find_pivot_row();
    // println!("Hang xoay: {}", pivot_row);
    // println!(
    //     "Phan tu xoay: {}",
    //     simplex.tableau[(pivot_row, pivot_column)]
    // );

    // simplex.pivot(pivot_row, pivot_column);
    // println!("Bang don hinh sau khi xoay: {}", simplex.tableau);
    // println!("Ket thuc lan lap thu 2");
    // println!("////////////////////");

    let result = match simplex.solve() {
        Some(result) => result,
        None => (0.0, DVector::from_vec(vec![0.0, 0.0, 0.0])),
    };

    println!("Ket qua:\nx = {} f(x) = {}", result.1, result.0);
}
