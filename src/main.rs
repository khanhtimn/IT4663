use nalgebra::*;
use simplex::*;

fn main() {
    let c = DVector::from_vec(vec![1.0, 1.0]);

    #[allow(non_snake_case)]
    let A = DMatrix::from_row_slice(
        3,
        2,
        &[
            2.0, 1.0, // x1 - 2x2 <= 6
            3.0, 4.0, // x1       <= 10
            1.0, -1.0, //    - x2  <= -1
        ],
    );

    // // b = [6, 10, -1]t
    let b = DVector::from_vec(vec![8.0, 24.0, 2.0]);

    let mut simplex = Simplex::new(c, A, b);

    if let Some(result) = simplex.solve() {
        println!("Ket qua:\nx = {} f(x) = {}", result.1, result.0);
    }
}
