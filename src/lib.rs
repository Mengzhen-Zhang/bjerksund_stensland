mod forward_ad;

/// Helper for the Bivariate Normal CDF using `mv_norm::bvnd`.
/// This is the corrected, more efficient implementation.
fn bn_cdf(a: f64, b: f64, rho: f64) -> f64 {
    mv_norm::bvnd(-a, -b, rho)
}

fn std_norm_cdf(x: f64) -> f64 {
    use statrs::distribution::ContinuousCDF;
    statrs::distribution::Normal::new(0.0, 1.0).unwrap().cdf(x)
}

struct BjerksundStensland {
    s: f64, k: f64, t: f64, r: f64, b: f64, v: f64,
    tau: f64, beta: f64, x_t: f64, x_tau: f64,
}

impl BjerksundStensland {
    fn new(
        spot: f64,
        strike: f64,
        volatility: f64,
        risk_free_rate: f64,
        dividend_rate: f64,
        time_to_maturity: f64,  // in years
    ) -> Self
    {
        let s = spot;
        let k = strike;
        let r = risk_free_rate;
        let q = dividend_rate;
        let b = r - q;
        let t = time_to_maturity;
        let v = volatility;
        let tau = 0.5 * (5.0_f64.sqrt() - 1.0) * t;
        let beta = {
            let term = 0.5 - b / v.powi(2);
            term + (term.powi(2) + 2.* r / v.powi(2)).sqrt()
        };
        let mut result = BjerksundStensland {
            s, k, t, r, b, v, tau, beta, x_t: 0.0, x_tau: 0.0,
        };
        let x_t = result.x(t);
        let x_tau = result.x(t - tau);
        result.x_t = x_t;
        result.x_tau = x_tau;
        result
    }

    fn x(&self, t: f64) -> f64 {
        let Self {
            s: _, k, t:_, r, b, v,
            tau: _, beta, x_t: _, x_tau: _,
        } = self;
        let b_inf = beta / (beta - 1.) * k;
        let b_0 = (r * k / (r - b)).max(*k);
        let h = - (b * t + 2.* v * t.sqrt())
            * (k.powi(2) / ((b_inf - b_0) * b_0));
        b_0 + (b_inf - b_0) * (1. - h.exp())
    }

    // d_lambda / d_gamma
    fn zeta(&self, gamma: f64) -> f64 {
        self.b + (gamma - 0.5) * self.v.powi(2)
    }

    fn alpha(&self, x: f64) -> f64 {
        (x - self.k) * x.powf(- self.beta)
    }
    
    fn lambda(&self, gamma: f64) -> f64 {
        let r = self.r;
        let term = self.b + 0.5 * (gamma - 1.) * self.v.powi(2);
        - r + gamma * term
    }

    fn kappa(&self, gamma: f64) -> f64 {
        let b = self.b;
        let v = self.v;
        2. * b / v.powi(2) + (2. * gamma - 1.)
    }
    
    fn phi(&self, gamma: f64, h: f64) -> f64 {
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
        result *= std_norm_cdf(- d1)
            - (x_t / s).powf(kappa) * std_norm_cdf(- d2);
        result
    }

    fn psi(&self, gamma: f64, h: f64) -> f64 {
        let Self {s, k, t, r, b, v,
            tau, beta, x_t, x_tau} = self;
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
            bn_cdf(d, dd, rho)
        };
        let m2 = {
            let d = -((x_t.powi(2) / (s * x_tau)).ln() + zeta_tau) / v_tau;
            let dd = -((x_t.powi(2) / (s * h)).ln() + zeta_t) / v_t;
            - (x_t / s).powf(kappa) * bn_cdf(d, dd, rho)
        };
        let m3 = {
            let d = -((s / x_tau).ln() - zeta_tau) / v_tau;
            let dd = -((x_tau.powi(2) / (s * h)).ln() + zeta_t) / v_t;
            - (x_tau / s).powf(kappa) * bn_cdf(d, dd, - rho)
        };
        let m4 = {
            let d = -((x_t.powi(2) / (s * x_tau)).ln() - zeta_tau) / v_tau;
            let dd = -((s * x_tau.powi(2) / (h * x_t.powi(2))).ln() + zeta_t) / v_t;
            (x_tau / x_t).powf(kappa) * bn_cdf(d, dd, - rho)
        };
        let lambda = self.lambda(gamma);
        (lambda * t).exp() * s.powf(gamma) * (m1 + m2 + m3 + m4)
    }

    fn price_call(&self) -> f64 {
        let Self {s, k, t, r, b, v,
            tau, beta, x_t, x_tau} = *self;
        let row1 = self.alpha(x_t) * s.powf(beta)
            - self.alpha(x_t) * self.phi(beta, x_t);
        let row2 = self.phi(1., x_t)
            - self.phi(1., x_tau);
        let row3 = - k * self.phi(0., x_t)
            + k * self.phi(0., x_tau);
        let row4 = self.alpha(x_tau) * self.phi(beta, x_tau)
            - self.alpha(x_tau) * self.psi(beta, x_tau);
        let row5 = self.psi(1., x_tau)
            - self.psi(1., k);
        let row6 = - k * self.psi(0., x_tau)
            + k * self.psi(0., k);
        row1 + row2 + row3 + row4 + row5 + row6
    }

}

pub fn bs_call_price(
    spot: f64,
    strike: f64,
    volatility: f64,
    risk_free_rate: f64,
    dividend_rate: f64,
    time_to_maturity: f64,
) -> f64 {
    BjerksundStensland::new(spot, strike, volatility, risk_free_rate, dividend_rate, time_to_maturity).price_call()
}


pub fn bs_put_price(
    spot: f64,
    strike: f64,
    volatility: f64,
    risk_free_rate: f64,
    dividend_rate: f64,
    time_to_maturity: f64,
) -> f64 {
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
        let price = bs_call_price(s, k, v, r, q, t);
        assert_eq!(price, 1.0_f64, "price is {}", price);
    }
}
