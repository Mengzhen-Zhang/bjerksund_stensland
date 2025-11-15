use roots::SearchError;

pub fn find_root_newton_raphson_dual<F, Func>(
    start: F,
    mut f: Func,
    convergency: &mut dyn roots::Convergency<F>,
) -> Result<F, SearchError>
where
    F: roots::FloatType,
    Func: FnMut(F) -> (F, F),
{
    let mut x = start;

    let mut iter = 0;
    loop {
        let (f, d) = f(x);
        if convergency.is_root_found(f) {
            return Ok(x);
        }
        // Derivative is 0; try to correct the bad starting point
        if convergency.is_root_found(d) {
            if iter == 0 {
                x = x + F::one();
                iter = iter + 1;
                continue;
            } else {
                return Err(SearchError::ZeroDerivative);
            }
        }

        let x1 = x - f / d;
        if convergency.is_converged(x, x1) {
            return Ok(x1);
        }

        x = x1;
        iter = iter + 1;

        if convergency.is_iteration_limit_reached(iter) {
            return Err(SearchError::NoConvergency);
        }
    }
}
