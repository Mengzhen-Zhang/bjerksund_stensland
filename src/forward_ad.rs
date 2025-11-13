use autodiff::One;
use num_dual::*;
use nalgebra::SVector;
use nalgebra::RealField;
use statrs::distribution::ContinuousCDF;
use statrs::distribution::Continuous;
use std::f64::consts::PI;

fn std_norm_pdf(x: f64) -> f64 {
    (- x.powi(2) * 0.5).exp() / (2. * PI).sqrt()
}

fn std_norm_cdf(x: f64) -> f64 {
    use statrs::distribution::Normal as Normal;
    let cdf = Normal::new(0.0, 1.0).unwrap().cdf(x);
    cdf
}

fn std_norm_pdf_forward<D: DualNum<f64>>(x: &D) -> D {
    (- x.powi(2) * 0.5).exp() / (2. * PI).sqrt()
}

fn std_norm_cdf_dual(x: &Dual64) -> Dual64 {
    let cdf = std_norm_cdf(x.re);

    let x_dual = Dual::from_re(x.re).derivative();
    let dx = std_norm_pdf_forward(&x_dual);

    let re = cdf;
    let eps = dx.re * x.eps;

    Dual64::new(re, eps)
}

fn std_norm_cdf_dual2(x: &Dual2_64) -> Dual2_64 {
    let cdf = std_norm_cdf(x.re);

    let x_dual = Dual::from_re(x.re).derivative();
    let dx = std_norm_pdf_forward(&x_dual);

    let re = cdf;
    let v1 = dx.re * x.v1;
    let v2 = dx.eps * x.v1.powi(2) + dx.re * x.v2;

    Dual2_64::new(re, v1, v2 )
}

fn bn_d_rho_forward<D: DualNum<f64>>(a: &D, b: &D, rho: &D) -> D {
    let one: D = 1.0.into();
    let two: D = 2.0.into();
    let two_pi: D = (2. * PI).into();
    let c = one / (two_pi * ( - rho.powi(2) + 1.).sqrt());
    let q = - (a.powi(2) - two * rho * a * b + b.powi(2)) / ((-rho.powi(2) + 1.) * 2.);
    c * q.exp()
}

fn bn_d_a_forward(a: &Dual64, b: &Dual64, rho: &Dual64) -> Dual64 {
    let w = (b - rho * a) / (-rho.powi(2) + 1.).sqrt();
    std_norm_pdf_forward(a) * std_norm_cdf_dual(&w)
}

fn bn_d_b_forward(a: &Dual64, b: &Dual64, rho: &Dual64) -> Dual64 {
    let v = (a - rho * b) / (-rho.powi(2) + 1.).sqrt();
    std_norm_pdf_forward(b) * std_norm_cdf_dual(&v)
}

/// Helper for the Bivariate Normal CDF using `mv_norm::bvnd`.
/// This is differentiable using analytical gradients.
fn bn_cdf_dual2(a: &Dual2_64, b: &Dual2_64, rho: &Dual2_64) -> Dual2_64 {
    let a_dual = Dual::from_re(a.re);
    let b_dual = Dual::from_re(b.re);
    let rho_dual = Dual::from_re(rho.re);

    let d_a = bn_d_a_forward(&a_dual, &b_dual, &rho_dual).re;
    let d_b = bn_d_b_forward(&a_dual, &b_dual, &rho_dual).re;
    let d_rho = bn_d_rho_forward(&a_dual, &b_dual, &rho_dual).re;

    let d_a_a = bn_d_a_forward(&a_dual.derivative(), &b_dual, &rho_dual).eps;
    let d_a_b = bn_d_a_forward(&a_dual, &b_dual.derivative(), &rho_dual).eps;
    let d_a_rho = bn_d_a_forward(&a_dual, &b_dual, &rho_dual.derivative()).eps;

    let d_b_a = bn_d_b_forward(&a_dual.derivative(), &b_dual, &rho_dual).eps;
    let d_b_b = bn_d_b_forward(&a_dual, &b_dual.derivative(), &rho_dual).eps;
    let d_b_rho = bn_d_b_forward(&a_dual, &b_dual, &rho_dual.derivative()).eps;

    let d_rho_a = bn_d_rho_forward(&a_dual.derivative(), &b_dual, &rho_dual).eps;
    let d_rho_b = bn_d_rho_forward(&a_dual, &b_dual.derivative(), &rho_dual).eps;
    let d_rho_rho = bn_d_rho_forward(&a_dual, &b_dual, &rho_dual.derivative()).eps;

    // Forward value via the external function
    let re = mv_norm::bvnd(-a.re, -b.re, rho.re);
    let v1 = d_a * a.v1 + d_b * b.v1 + d_rho * rho.v1;
    let v2: f64 = {
        let term1 = d_a * a.v2 + d_b * b.v2 + d_rho * rho.v2;
        let term2 = a.v1 * (d_a_a * a.v1 + d_a_b * b.v1 + d_a_rho * rho.v1);
        let term3 = b.v1 * (d_b_a * a.v1 + d_b_b * b.v1 + d_b_rho * rho.v1);
        let term4 = rho.v1 * (d_rho_a * a.v1 + d_rho_b * b.v1 + d_rho_rho * rho.v1);
        term1 + term2 + term3 + term4
    };

    Dual2_64::new(re, v1, v2)
}

struct BjerksundStensland {
    s: Dual2_64, k: Dual2_64, t: Dual2_64, r: Dual2_64, b: Dual2_64, v: Dual2_64,
    tau: Dual2_64, beta: Dual2_64, x_t: Dual2_64, x_tau: Dual2_64,
}

impl BjerksundStensland {
    fn new(
        spot: Dual2_64,
        strike: Dual2_64,
        volatility: Dual2_64,
        risk_free_rate: Dual2_64,
        dividend_rate: Dual2_64,
        time_to_maturity: Dual2_64,  // in years
    ) -> Self
    {
        let s = spot;
        let k = strike;
        let r = risk_free_rate;
        let q = dividend_rate;
        let b = r - q;
        let t = time_to_maturity;
        let v = volatility;
        let tau = t * 0.5 * (5.0_f64.sqrt() - 1.0);
        let beta = {
            let term = - b / (v * v) + 0.5;
            term + (term.powi(2) + r * 2. / v.powi(2)).sqrt()
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

    fn x(&self, t: Dual2_64) -> Dual2_64 {
        let Self {
            s: _, k, t:_, r, b, v,
            tau: _, beta, x_t: _, x_tau: _,
        } = *self;
        let b_inf = beta / (beta - 1.) * k;
        let b_0 = (r * k / (r - b)).max(k);
        let h = - (b * t + v * t.sqrt() * 2.)
            * (k.powi(2) / ((b_inf - b_0) * b_0));
        b_0 + (b_inf - b_0) * (- h.exp() + 1.)
    }

    // d_lambda / d_gamma
    fn zeta(&self, gamma: Dual2_64) -> Dual2_64 {
        self.b + (gamma - 0.5) * self.v.powi(2)
    }

    fn alpha(&self, x: Dual2_64) -> Dual2_64 {
        (x - self.k) * (- self.beta * x.ln()).exp()
    }
    
    fn lambda(&self, gamma: Dual2_64) -> Dual2_64 {
        let r = self.r;
        let term = self.b + (gamma - 1.) * self.v.powi(2) * 0.5;
        - r + gamma * term
    }

    fn kappa(&self, gamma: Dual2_64) -> Dual2_64 {
        let b = self.b;
        let v = self.v;
        b * 2. / v.powi(2) + (gamma * 2. - 1.)
    }
    
    fn phi(&self, gamma: Dual2_64, h: Dual2_64) -> Dual2_64 {
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
        let mut result = (lambda * tau).exp() * (gamma * s.ln()).exp();
        result *= std_norm_cdf_dual2(&(- d1))
            - (kappa * (x_t / s).ln()).exp() * std_norm_cdf_dual2(&(- d2));
        result
    }

    fn psi(&self, gamma: Dual2_64, h: Dual2_64) -> Dual2_64 {
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
            bn_cdf_dual2(&d, &dd, &rho)
        };
        let m2 = {
            let d = -((x_t.powi(2) / (s * x_tau)).ln() + zeta_tau) / v_tau;
            let dd = -((x_t.powi(2) / (s * h)).ln() + zeta_t) / v_t;
            - (kappa * (x_t / s).ln()).exp() * bn_cdf_dual2(&d, &dd, &rho)
        };
        let m3 = {
            let d = -((s / x_tau).ln() - zeta_tau) / v_tau;
            let dd = -((x_tau.powi(2) / (s * h)).ln() + zeta_t) / v_t;
            - (kappa * (x_tau / s).ln()).exp() * bn_cdf_dual2(&d, &dd, &(- rho))
        };
        let m4 = {
            let d = -((x_t.powi(2) / (s * x_tau)).ln() - zeta_tau) / v_tau;
            let dd = -((s * x_tau.powi(2) / (h * x_t.powi(2))).ln() + zeta_t) / v_t;
            (kappa * (x_tau / x_t).ln()).exp() * bn_cdf_dual2(&d, &dd, &(- rho))
        };
        let lambda = self.lambda(gamma);
        (lambda * t).exp() * (gamma * s.ln()).exp() * (m1 + m2 + m3 + m4)
    }

    fn price_call(&self) -> Dual2_64 {
        let Self {s, k, t, r, b, v,
            tau, beta, x_t, x_tau} = *self;
        
        let one = 1.0.into();
        let zero = 0.0.into();
        
        let row1 = self.alpha(x_t) * (beta * s.ln()).exp()
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

pub fn bs_call_price_dual2(
    spot: Dual2_64,
    strike: Dual2_64,
    volatility: Dual2_64,
    risk_free_rate: Dual2_64,
    dividend_rate: Dual2_64,
    time_to_maturity: Dual2_64,
) -> Dual2_64 {
    BjerksundStensland::new(spot, strike, volatility, risk_free_rate, dividend_rate, time_to_maturity).price_call()
}

pub fn bs_call_delta(
    spot: f64,
    strike: f64,
    volatility: f64,
    risk_free_rate: f64,
    dividend_rate: f64,
    time_to_maturity: f64,
) -> f64 {
    let price = bs_call_price_dual2(
        Dual2_64::from_re(spot).derivative(),
        strike.into(),
        volatility.into(),
        risk_free_rate.into(),
        dividend_rate.into(),
        time_to_maturity.into());
    price.v1
}

pub fn bs_call_theta(
    spot: f64,
    strike: f64,
    volatility: f64,
    risk_free_rate: f64,
    dividend_rate: f64,
    time_to_maturity: f64,
) -> f64 {
    let price = bs_call_price_dual2(
        Dual2_64::from_re(spot),
        strike.into(),
        volatility.into(),
        risk_free_rate.into(),
        dividend_rate.into(),
        Dual2_64::from_re(time_to_maturity).derivative());
    - price.v1 / 365.
}

pub fn bs_call_rho(
    spot: f64,
    strike: f64,
    volatility: f64,
    risk_free_rate: f64,
    dividend_rate: f64,
    time_to_maturity: f64,
) -> f64 {
    let price = bs_call_price_dual2(
        Dual2_64::from_re(spot),
        strike.into(),
        volatility.into(),
        Dual2_64::from_re(risk_free_rate).derivative(),
        dividend_rate.into(),
        Dual2_64::from_re(time_to_maturity)
    );
    price.v1 / 100.
}

pub fn bs_call_vega(
    spot: f64,
    strike: f64,
    volatility: f64,
    risk_free_rate: f64,
    dividend_rate: f64,
    time_to_maturity: f64,
) -> f64 {
    let price = bs_call_price_dual2(
        spot.into(),
        strike.into(),
        Dual2_64::from_re(volatility).derivative(),
        risk_free_rate.into(),
        dividend_rate.into(),
        time_to_maturity.into()
    );
    price.v1 / 100.0
}

pub fn bs_call_gamma(
    spot: f64,
    strike: f64,
    volatility: f64,
    risk_free_rate: f64,
    dividend_rate: f64,
    time_to_maturity: f64,
) -> f64 {
    let price = bs_call_price_dual2(
        Dual2_64::from_re(spot).derivative(),
        strike.into(),
        volatility.into(),
        risk_free_rate.into(),
        dividend_rate.into(),
        time_to_maturity.into()
    );
    price.v2
}


pub fn bs_put_price_dual2(
    spot: Dual2_64,
    strike: Dual2_64,
    volatility: Dual2_64,
    risk_free_rate: Dual2_64,
    dividend_rate: Dual2_64,
    time_to_maturity: Dual2_64,
) -> Dual2_64 {
    // use put-call parity to price american put
    let (spot, strike, risk_free_rate, dividend_rate) = (strike, spot, dividend_rate, risk_free_rate);
    BjerksundStensland::new(spot, strike, volatility, risk_free_rate, dividend_rate, time_to_maturity).price_call()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bs_call_price() {
        let s = 100.0;
        let k = 90.0;
        let v = 0.25;
        let r = 0.05;
        let q = 0.03;
        let t = 1.0;
        let price = bs_call_price_dual2(s.into(), k.into(), v.into(), r.into(), q.into(), t.into());
        let delta = bs_call_delta(s, k, v, r, q, t);
        let theta = bs_call_theta(s, k, v, r, q, t);
        let gamma = bs_call_gamma(s, k, v, r, q, t);
        let vega = bs_call_vega(s, k, v, r, q, t);
        let rho = bs_call_rho(s, k, v, r, q, t);
        assert_eq!(price, delta.into(),
            "price is {}\n delta is  {}\n theta is {}\n gamma is {}\n vega is {}\n rho is {}",
            price,
            delta,
            theta,
            gamma,
            vega,
            rho,
        );
    }

    use mv_norm::bvnd; // Ensure this is in scope for the test

    #[test]
    fn test_bn_cdf_dual2_finite_difference() {
        // 1. Define base values and derivative components
        let a_val = 0.5;
        let b_val = -0.2;
        let rho_val = 0.3;

        let a_v1 = 0.1;
        let a_v2 = 0.05;
        let b_v1 = -0.2;
        let b_v2 = -0.03;
        let rho_v1 = 0.05;
        let rho_v2 = 0.01;

        // 2. Get the Automatic Differentiation (AD) results
        let a_dual = Dual2_64::new(a_val, a_v1, a_v2);
        let b_dual = Dual2_64::new(b_val, b_v1, b_v2);
        let rho_dual = Dual2_64::new(rho_val, rho_v1, rho_v2);
        
        let ad_result = bn_cdf_dual2(&a_dual, &b_dual, &rho_dual);
        let (ad_re, ad_v1, ad_v2) = (ad_result.re, ad_result.v1, ad_result.v2);

        // 3. Setup for Finite Difference Method (FDM)
        
        // Helper function to evaluate the composite function g(t) at a given t
        // g(t) = f(a(t), b(t), rho(t))
        let g = |t: f64| -> f64 {
            // Reconstruct inputs at time 't' using 2nd-order Taylor expansion
            let a_t = a_val + a_v1 * t + 0.5 * a_v2 * t.powi(2);
            let b_t = b_val + b_v1 * t + 0.5 * b_v2 * t.powi(2);
            let rho_t = rho_val + rho_v1 * t + 0.5 * rho_v2 * t.powi(2);

            // Return the base value of the function f(a, b, rho)
            // This MUST match the base function used in bn_cdf_dual2
            mv_norm::bvnd(-a_t, -b_t, rho_t)
        };

        let h = 1e-5; // Step size for finite difference

        // 4. Calculate FDM approximations
        
        // Get function values at t=0, t=h, and t=-h
        let g_zero = g(0.0);
        let g_plus_h = g(h);
        let g_minus_h = g(-h);

        // Central difference for first derivative
        let fdm_v1 = (g_plus_h - g_minus_h) / (2.0 * h);
        
        // Central difference for second derivative
        let fdm_v2 = (g_plus_h - 2.0 * g_zero + g_minus_h) / (h * h);

        // 5. Assert results
        
        // Value (re) should be almost identical
        assert!(
            (ad_re - g_zero).abs() < 1e-15,
            "Value (re) mismatch: AD = {}, FDM = {}",
            ad_re, g_zero
        );

        // First derivative (v1) should be very close
        let v1_precision = 1e-7;
        assert!(
            (ad_v1 - fdm_v1).abs() < v1_precision,
            "V1 mismatch: AD = {}, FDM = {}, Diff = {}",
            ad_v1, fdm_v1, (ad_v1 - fdm_v1).abs()
        );

        // Second derivative (v2) is less stable, so use a looser tolerance
        let v2_precision = 1e-4;
         assert!(
            (ad_v2 - fdm_v2).abs() < v2_precision,
            "V2 mismatch: AD = {}, FDM = {}, Diff = {}",
            ad_v2, fdm_v2, (ad_v2 - fdm_v2).abs()
        );
    }
    
}
