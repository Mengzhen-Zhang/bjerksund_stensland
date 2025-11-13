use ad_trait::{AD, F64};
use ad_trait::differentiable_function::DifferentiableFunctionTrait;
use ad_trait::forward_ad::adfn::adfn;
use statrs::distribution::ContinuousCDF;
use simba::scalar::RealField;
use simba::scalar::ComplexField;

fn std_norm_cdf_val(x: f64) -> f64 {
    statrs::distribution::Normal::new(0.0, 1.0).unwrap().cdf(x)
}

fn std_norm_cdf_forward<const N: usize>(x: adfn<N>) -> adfn<N> {
    let xv = x.value();
    let value = statrs::distribution::Normal::new(0.0, 1.0).unwrap().cdf(xv);
    
    let dxv = (-0.5 * xv * xv).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let mut tangent = x.tangent();
    for v in tangent.iter_mut(){
        *v *= dxv;
    };

    adfn::new(value, tangent)
}

/// Helper for the Bivariate Normal CDF using `mv_norm::bvnd`.
/// This is differentiable using analytical gradients.
fn bn_cdf_forward<const N: usize>(a: adfn<N>, b: adfn<N>, rho: adfn<N>) -> adfn<N> {
    // Extract scalar values
    let (av, bv, rhov) = (a.value(), b.value(), rho.value());

    // Forward value via the external function
    let value = mv_norm::bvnd(-av, -bv, rhov);

    // Compute analytical gradients
    let sqrt_term = (1.0 - rhov * rhov).sqrt();
    let phi_a = (-0.5 * av * av).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let phi_b = (-0.5 * bv * bv).exp() / (2.0 * std::f64::consts::PI).sqrt();

    // Standard normal CDF helper (you can replace with statrs::distribution)
    let d_a = -phi_a * std_norm_cdf_val((bv - rhov * av) / sqrt_term);
    let d_b = -phi_b * std_norm_cdf_val((av - rhov * bv) / sqrt_term);

    let phi2 = (-(av * av - 2.0 * rhov * av * bv + bv * bv) / (2.0 * (1.0 - rhov * rhov))).exp()
        / (2.0 * std::f64::consts::PI * (1.0 - rhov * rhov).sqrt());
    let d_rho = phi2;

    let mut tangent = [0.0; N];
    let a_t = a.tangent();
    let b_t = b.tangent();
    let rho_t = rho.tangent();
    for i in 0..N {
        tangent[i] = d_a * a_t[i] + d_b * b_t[i] + d_rho * rho_t[i];
    }

    adfn::new(value, tangent)
}

// fn sqrt<const N: usize>(x: adfn<N>) -> adfn<N> {
//     let xv = x.value();
//     let value = xv.sqrt();

//     let d_x = 0.5 / value;
//     let mut tangent = x.tangent();
//     for v in tangent.iter_mut() {
//       *v *= d_x
//     }

//     adfn::new(value, tangent)
// }

struct BjerksundStensland<const N: usize> {
    s: adfn<N>, k: adfn<N>, t: adfn<N>, r: adfn<N>, b: adfn<N>, v: adfn<N>,
    tau: adfn<N>, beta: adfn<N>, x_t: adfn<N>, x_tau: adfn<N>,
}

impl<const N: usize> BjerksundStensland<N> {
    fn new(
        spot: adfn<N>,
        strike: adfn<N>,
        volatility: adfn<N>,
        risk_free_rate: adfn<N>,
        dividend_rate: adfn<N>,
        time_to_maturity: adfn<N>,  // in years
    ) -> Self
    {
        let s = spot;
        let k = strike;
        let r = risk_free_rate;
        let q = dividend_rate;
        let b = r - q;
        // let (s, k, r, b) = if is_call {
        //     (s, k, r, b)
        // } else {
        //     (k, s, r - b, - b)
        // };
        let t = time_to_maturity;
        let v = volatility;
        let tau = F64(0.5 * (5.0_f64.sqrt() - 1.0)) * t;
        let beta = {
            let term = F64(0.5) - b / (v * v);
            term + (term * term + F64(2.)* r / (v * v)).sqrt()
        };
        let mut result = BjerksundStensland {
            s, k, t, r, b, v, tau, beta, x_t: 0.0.into(), x_tau: 0.0.into(),
        };
        let x_t = result.x(t);
        let x_tau = result.x(t - tau);
        result.x_t = x_t;
        result.x_tau = x_tau;
        result
    }

    fn x(&self, t: adfn<N>) -> adfn<N> {
        let Self {
            s: _, k, t:_, r, b, v,
            tau: _, beta, x_t: _, x_tau: _,
        } = *self;
        let b_inf = beta / (beta - F64(1.)) * k;
        let b_0 = (r * k / (r - b)).max(k);
        let h = - (b * t + F64(2.)* v * t.sqrt())
            * (k.powi(2) / ((b_inf - b_0) * b_0));
        b_0 + (b_inf - b_0) * (F64(1.) - h.exp())
    }

    // d_lambda / d_gamma
    fn zeta(&self, gamma: adfn<N>) -> adfn<N> {
        self.b + (gamma - F64(0.5)) * self.v.powi(2)
    }

    fn alpha(&self, x: adfn<N>) -> adfn<N> {
        (x - self.k) * x.powf(- self.beta)
    }
    
    fn lambda(&self, gamma: adfn<N>) -> adfn<N> {
        let r = self.r;
        let term = self.b + F64(0.5) * (gamma - F64(1.)) * self.v.powi(2);
        - r + gamma * term
    }

    fn kappa(&self, gamma: adfn<N>) -> adfn<N> {
        let b = self.b;
        let v = self.v;
        F64(2.) * b / v.powi(2) + (F64(2.) * gamma - F64(1.))
    }
    
    fn phi(&self, gamma: adfn<N>, h: adfn<N>) -> adfn<N> {
        let Self {s, v, tau, x_t, .. } = *self;
        let zeta = self.zeta(gamma);
        let lambda = self.lambda(gamma);
        let kappa = self.kappa(gamma);
        let denom = v * tau.sqrt();
        let d1 = {
            let num = (s / h).ln() + zeta * tau;
            num / denom
        };
        let d2 = {
            let num = (x_t.powi(2) / s / h).ln() + zeta * tau;
            num / denom
        };
        let mut result = (lambda * tau).exp() * s.powf(gamma);
        result *= std_norm_cdf_forward(- d1)
            - (x_t / s).powf(kappa) * std_norm_cdf_forward(- d2);
        result
    }

    fn psi(&self, gamma: adfn<N>, h: adfn<N>) -> adfn<N> {
        let Self {s, k, t, r, b, v,
            tau, beta, x_t, x_tau} = *self;
        let rho = (tau / t).sqrt();
        let kappa = self.kappa(gamma);
        let zeta = self.zeta(gamma);
        let zeta_t = zeta * t;
        let zeta_tau = zeta * tau;
        let v_t = v * t.sqrt();
        let v_tau = v * tau.sqrt();
        let m1 = {
            let d = -((s / x_tau).ln() + zeta_tau) / v_tau;
            let dd = -((s / h).ln() + zeta_t) / v_t;
            bn_cdf_forward(d, dd, rho)
        };
        let m2 = {
            let d = -((x_t.powi(2) / (s * x_tau)).ln() + zeta_tau) / v_tau;
            let dd = -((x_t.powi(2) / (s * h)).ln() + zeta_t) / v_t;
            - (x_t / s).powf(kappa) * bn_cdf_forward(d, dd, rho)
        };
        let m3 = {
            let d = -((s / x_tau).ln() - zeta_tau) / v_tau;
            let dd = -((x_tau.powi(2) / (s * h)).ln() + zeta_t) / v_t;
            - (x_tau / s).powf(kappa) * bn_cdf_forward(d, dd, - rho)
        };
        let m4 = {
            let d = -((x_t.powi(2) / (s * x_tau)).ln() - zeta_tau) / v_tau;
            let dd = -((s * x_tau.powi(2) / (h * x_t.powi(2))).ln() + zeta_t) / v_t;
            (x_tau / x_t).powf(kappa) * bn_cdf_forward(d, dd, - rho)
        };
        let lambda = self.lambda(gamma);
        (lambda * t).exp() * s.powf(gamma) * (m1 + m2 + m3 + m4)
    }

    fn price_call(&self) -> adfn<N> {
        let Self {s, k, t, r, b, v,
            tau, beta, x_t, x_tau} = *self;
        
        let one = 1.0.into();
        let zero = 0.0.into();
        
        let row1 = self.alpha(x_t) * s.powf(beta)
            - self.alpha(x_t) * self.phi(beta, x_t);
        let row2 = self.phi(one, x_t)
            - self.phi(one, x_tau);
        let row3 = - k * self.phi(zero, x_t)
            + k * self.phi(zero, x_tau);
        let row4 = self.alpha(x_tau) * self.phi(beta, x_tau)
            - self.alpha(x_tau) * self.psi(beta, x_tau);
        let row5 = self.psi(one, x_tau)
            - self.psi(one, k);
        let row6 = - k * self.psi(zero, x_tau)
            + k * self.psi(zero, k);
        row1 + row2 + row3 + row4 + row5 + row6
    }

}

pub fn bs_call_price_forward<const N: usize>(
    spot: adfn<N>,
    strike: adfn<N>,
    volatility: adfn<N>,
    risk_free_rate: adfn<N>,
    dividend_rate: adfn<N>,
    time_to_maturity: adfn<N>,
) -> adfn<N> {
    BjerksundStensland::new(spot, strike, volatility, risk_free_rate, dividend_rate, time_to_maturity).price_call()
}


pub fn bs_put_price_forward<const N: usize>(
    spot: adfn<N>,
    strike: adfn<N>,
    volatility: adfn<N>,
    risk_free_rate: adfn<N>,
    dividend_rate: adfn<N>,
    time_to_maturity: adfn<N>,
) -> adfn<N> {
    // use put-call parity to price american put
    let (spot, strike, risk_free_rate, dividend_rate) = (strike, spot, dividend_rate, risk_free_rate);
    BjerksundStensland::new(spot, strike, volatility, risk_free_rate, dividend_rate, time_to_maturity).price_call()
}

#[cfg(test)]
mod lib_tests {
    use super::*;

    #[test]
    fn test_bs_call_price() {
        let s: adfn<1> = 100.0.into();
        let k: adfn<1> = 90.0.into();
        let v: adfn<1> = 0.25.into();
        let r: adfn<1> = 0.05.into();
        let q: adfn<1> = 0.03.into();
        let t: adfn<1> = 1.0.into();
        let price = bs_call_price_forward(s, k, v, r, q, t);
        assert_eq!(price, 1.0.into(), "price is {}", price);
    }
}
