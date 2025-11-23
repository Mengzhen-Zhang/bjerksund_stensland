use nalgebra::Const;
use num_dual::{Derivative, Dual2Vec64, DualNum, Dual2_64};
use std::f64::consts::PI;
use crate::std_norm::{std_norm_pdf, std_norm_cdf};

pub struct BvNorm<T> {
    a: T,
    b: T,
    rho: T,
}

impl<T> BvNorm<T> {
    fn new(a:T, b:T, rho:T) -> Self {
        BvNorm { a, b, rho }
    }
}

pub trait BvNormPdf<T> {
    fn pdf(&self) -> T;
}

pub trait BvNormCdf<T> {
    fn cdf(&self) -> T;
}

impl<D: DualNum<f64>> BvNormPdf<D> for BvNorm<D> {
    fn pdf(&self) -> D {
        let BvNorm { a, b, rho } = self;
        let one = D::one();
        let two = D::one() * 2.;
        let two_pi = D::one() * 2. * PI;
        let c = one / (two_pi * ( - rho.powi(2) + 1.).sqrt());
        let q = - (a.powi(2) - two * rho * a * b + b.powi(2)) / ((-rho.powi(2) + 1.) * 2.);
        c * q.exp()
    }
}

impl BvNormCdf<f64> for BvNorm<f64> {
    fn cdf(&self) -> f64 {
        let BvNorm { a, b, rho } = *self;
        mv_norm::bvnd(-a, -b, rho)
    }
}

macro_rules! calc_v2 {
    // usage: calc_v2!(dx, self, N)
    ($dx:expr, $self_val:expr, $N:ident) => {{
        // We wrap in a block to keep intermediate variables local
        let dx_v1 = $dx.v1.unwrap_generic(Const::<1>, Const::<$N>);
        let dx_v2 = $dx.v2.unwrap_generic(Const::<$N>, Const::<$N>);
        let self_v1 = $self_val.v1.unwrap_generic(Const::<1>, Const::<$N>);

        // Return the final calculation
        dx_v1.transpose() * self_v1 + dx_v2 * $dx.re
    }};
}

impl<const N: usize> BvNormCdf<Dual2Vec64<Const<N>>> for BvNorm<Dual2Vec64<Const<N>>> {
    fn cdf(&self) -> Dual2Vec64<Const<N>> {
        let BvNorm { a, b, rho } = *self;
        let w = (b - rho * a) / (-rho.powi(2) + 1.).sqrt();
        let da = std_norm_pdf(a) * std_norm_cdf(w);
        let v = (a - rho * b) / (-rho.powi(2) + 1.).sqrt();
        let db = std_norm_pdf(b) * std_norm_cdf(v);
        let drho = self.pdf();

        let re = BvNorm {a: a.re, b: b.re, rho: rho.re}.cdf();
        let v1 = a.v1 * da.re + b.v1 * db.re + rho.v1 * drho.re;
        let term_a = calc_v2!(da, a, N);
        let term_b = calc_v2!(db, b, N);
        let term_rho = calc_v2!(drho, rho, N);
        let v2 = Derivative::some(term_a + term_b + term_rho);

        Dual2Vec64::<Const<N>>::new(re, v1, v2)
    }
}

impl BvNormCdf<Dual2_64> for BvNorm<Dual2_64> {
    fn cdf(&self) -> Dual2_64 {
        let BvNorm { a, b, rho } = *self;
        let w = (b - rho * a) / (-rho.powi(2) + 1.).sqrt();
        let da = std_norm_pdf(a) * std_norm_cdf(w);
        let v = (a - rho * b) / (-rho.powi(2) + 1.).sqrt();
        let db = std_norm_pdf(b) * std_norm_cdf(v);
        let drho = self.pdf();

        let re = BvNorm {a: a.re, b: b.re, rho: rho.re}.cdf();
        let v1 = a.v1 * da.re + b.v1 * db.re + rho.v1 * drho.re;
        let term_a = da.v1 * a.v1 + da.v2 * da.re;
        let term_b = db.v1 * b.v1 + db.v2 * db.re;
        let term_rho = drho.v1 * rho.v1 + drho.v2 * drho.re;
        let v2 = term_a + term_b + term_rho;

        Dual2_64::new(re, v1, v2)
    }
}

pub fn bv_norm_cdf<T>(a: T, b:T, rho: T) -> T
where
    BvNorm<T>: BvNormCdf<T>
{
    BvNorm::new(a, b, rho).cdf()
}

pub fn bv_norm_pdf<T>(a: T, b:T, rho: T) -> T
where
    BvNorm<T>: BvNormPdf<T>
{
    BvNorm::new(a, b, rho).pdf()
}
