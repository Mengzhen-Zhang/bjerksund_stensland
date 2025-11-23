use nalgebra::Const;
use statrs::distribution::ContinuousCDF;
use num_dual::{Derivative, Dual2Vec64, Dual2_64, Dual64, DualNum, DualVec64};
use std::f64::consts::PI;

pub struct StdNorm<T>(T);

pub trait StdNormPdf<T> {
    fn pdf(&self) -> T;
}

pub trait StdNormCdf<T> {
    fn cdf(&self) -> T;
}

impl<D: DualNum<f64>> StdNormPdf<D> for StdNorm<D> {
    fn pdf(&self) -> D {
        (- self.0.powi(2) * 0.5).exp() / (2. * PI).sqrt()
    }
}

impl StdNormCdf<f64> for StdNorm<f64> {
    fn cdf(&self) -> f64 {
        use statrs::distribution::Normal as Normal;
        let cdf = Normal::new(0.0, 1.0).unwrap().cdf(self.0);
        cdf
    }
}

impl StdNormCdf<Dual64> for StdNorm<Dual64> {
    fn cdf(&self) -> Dual64 {
        let cdf = StdNorm(self.0.re).cdf();
        let dx = StdNorm(self.0.re).pdf();

        let re = cdf;
        let eps = dx * self.0.eps;

        Dual64::new(re, eps)
    }
}

impl StdNormCdf<Dual2_64> for StdNorm<Dual2_64> {
    fn cdf(&self) -> Dual2_64 {
        let cdf = StdNorm(self.0.re).cdf();

        let dx = self.pdf();

        let re = cdf;
        let v1 = dx.re * self.0.v1;
        let v2 = dx.v1 * self.0.v1 + dx.re * self.0.v2;

        Dual2_64::new(re, v1, v2 )
    }
}

impl<const N: usize> StdNormCdf<DualVec64<Const<N>>> for StdNorm<DualVec64<Const<N>>> {
    fn cdf(&self) -> DualVec64<Const<N>> {
        let cdf = StdNorm(self.0.re).cdf();
        let dx = self.pdf();

        let re = cdf;
        let eps = self.0.eps * dx.re;

        DualVec64::new(re, eps)
    }
}

impl<const N: usize> StdNormCdf<Dual2Vec64<Const<N>>> for StdNorm<Dual2Vec64<Const<N>>> {
    fn cdf(&self) -> Dual2Vec64<Const<N>> {
        let cdf = StdNorm(self.0.re).cdf();
        let dx = self.pdf();

        let re = cdf;
        let v1: Derivative<_,_,_,_> = self.0.v1 * dx.re;
        let dx_v1 = dx.v1.unwrap_generic(Const::<1>, Const::<N>);
        let dx_v2 = dx.v2.unwrap_generic(Const::<N>, Const::<N>);
        let self_v1 = self.0.v1.unwrap_generic(Const::<1>, Const::<N>);
        let v2 = dx_v1.transpose() * self_v1 + dx_v2 * dx.re;
        let v2 = Derivative::some(v2);

        Dual2Vec64::new(re, v1, v2 )
    }
}


pub fn std_norm_cdf<T>(x: T) -> T
where
    StdNorm<T>: StdNormCdf<T>
{
    StdNorm(x).cdf()
}

pub fn std_norm_pdf<T>(x: T) -> T
where
    StdNorm<T>: StdNormPdf<T>
{
    StdNorm(x).pdf()
}
