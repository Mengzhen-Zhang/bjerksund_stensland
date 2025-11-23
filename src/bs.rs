use num_dual::Dual2_64;
use num_dual::{Dual2SVec64, DualNum};
use nalgebra::{Const, RealField};

use crate::std_norm::*;
use crate::bv_norm::*;

struct BjerksundStensland<T> {
    s: T, k: T, t: T, r: T, b: T, v: T,
    tau: T, beta: T, x_t: T, x_tau: T,
}

type DVec = Dual2SVec64<6>;

const S_IDX: usize = 0;
const K_IDX: usize = 1;
const R_IDX: usize = 2;
const Q_IDX: usize = 3;
const T_IDX: usize = 4;
const V_IDX: usize = 5;

impl<D: DualNum<f64> + Copy + RealField> BjerksundStensland<D>
where
    BvNorm<D>: BvNormCdf<D>,
    StdNorm<D>: StdNormCdf<D>
{
    fn new(
        spot: D,
        strike: D,
        volatility: D,
        risk_free_rate: D,
        dividend_rate: D,
        time_to_maturity: D,  // in years
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
            let term = - b / v.powi(2) + 0.5;
            term + (term.powi(2) + r * 2. / v.powi(2)).sqrt()
        };
        let mut result = BjerksundStensland {
            s, k, t: t, r, b, v, tau: tau, beta, x_t: 0.0.into(), x_tau: 0.0.into(),
        };
        let x_t = result.x(t);
        let x_tau = result.x(t - tau);
        result.x_t = x_t;
        result.x_tau = x_tau;
        result
    }

    fn x(&self, t: D) -> D {
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
    fn zeta(&self, gamma: D) -> D {
        self.b + (gamma - 0.5) * self.v.powi(2)
    }

    fn alpha(&self, x: D) -> D {
        (x - self.k) * (- self.beta * x.ln()).exp()
    }
    
    fn lambda(&self, gamma: D) -> D {
        let r = self.r;
        let term = self.b + (gamma - 1.) * self.v.powi(2) * 0.5;
        - r + gamma * term
    }

    fn kappa(&self, gamma: D) -> D {
        let b = self.b;
        let v = self.v;
        b * 2. / v.powi(2) + (gamma * 2. - 1.)
    }
    
    fn phi(&self, gamma: D, h: D) -> D {
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
        result *= std_norm_cdf(- d1)
            - (kappa * (x_t / s).ln()).exp() * std_norm_cdf(- d2);
        result
    }

    fn psi(&self, gamma: D, h: D) -> D {
        let Self {s, k: _, t, r: _, b: _, v,
            tau, beta: _, x_t, x_tau} = *self;
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
            bv_norm_cdf(d, dd, rho)
        };
        let m2 = {
            let d = -((x_t.powi(2) / (s * x_tau)).ln() + zeta_tau) / v_tau;
            let dd = -((x_t.powi(2) / (s * h)).ln() + zeta_t) / v_t;
            - (kappa * (x_t / s).ln()).exp() * bv_norm_cdf(d, dd, rho)
        };
        let m3 = {
            let d = -((s / x_tau).ln() - zeta_tau) / v_tau;
            let dd = -((x_tau.powi(2) / (s * h)).ln() + zeta_t) / v_t;
            - (kappa * (x_tau / s).ln()).exp() * bv_norm_cdf(d, dd, - rho)
        };
        let m4 = {
            let d = -((x_t.powi(2) / (s * x_tau)).ln() - zeta_tau) / v_tau;
            let dd = -((s * x_tau.powi(2) / (h * x_t.powi(2))).ln() + zeta_t) / v_t;
            (kappa * (x_tau / x_t).ln()).exp() * bv_norm_cdf(d, dd, - rho)
        };
        let lambda = self.lambda(gamma);
        (lambda * t).exp() * (gamma * s.ln()).exp() * (m1 + m2 + m3 + m4)
    }

    fn price_call(&self) -> D {
        let Self {s, k, t: _, r: _, b: _, v: _,
            tau: _, beta, x_t, x_tau} = *self;
        
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

#[derive(Debug, Clone)]
pub enum OptionType {
    Put,
    Call,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct OptionOutcomes {
    price: f64,
    delta: f64,
    gamma: f64,
    vega: f64,
    theta: f64,
    rho: f64,
}

pub struct AmericanOption {
    spot: DVec,
    strike: DVec,
    risk_free_rate: DVec,
    dividend_rate: DVec,
    time_to_maturity: DVec,
    option_type: OptionType,
}

impl AmericanOption {
    pub fn new(
        spot: f64,
        strike: f64,
        risk_free_rate: f64,
        dividend_rate: f64,
        time_to_maturity: f64,
        is_call: bool,
    ) -> Self {
        let spot = DVec::from_re(spot).derivative(S_IDX);
        let strike = DVec::from_re(strike).derivative(K_IDX);
        let risk_free_rate = DVec::from_re(risk_free_rate).derivative(R_IDX);
        let dividend_rate = DVec::from_re(dividend_rate).derivative(Q_IDX);
        let time_to_maturity = DVec::from_re(time_to_maturity).derivative(T_IDX);
        let option_type = if is_call { OptionType::Call } else { OptionType::Put };
        Self { spot, strike, risk_free_rate, dividend_rate, time_to_maturity, option_type }
    }

    pub fn price(&self, volatility: f64) -> OptionOutcomes {
        let Self {
            spot: s,
            strike: k,
            risk_free_rate: r,
            dividend_rate: q,
            time_to_maturity: t,
            ..
        } = *self;
        let v = DVec::from_re(volatility).derivative(V_IDX);
        let result = match self.option_type {
            OptionType::Call => BjerksundStensland::new(s, k, v, r, q, t).price_call(),
            OptionType::Put => BjerksundStensland::new(k, s, v, q, r, t).price_call(),
        };
        let v1 = result.v1.unwrap_generic(Const::<1>, Const::<6>);
        let v2 = result.v2.unwrap_generic(Const::<6>, Const::<6>);
        let model_price = result.re;
        let delta = v1[S_IDX];
        let gamma = v2[(S_IDX, S_IDX)];
        let theta = - v1[T_IDX] / 365.;
        let rho = v1[R_IDX] / 100.;
        let vega = v1[V_IDX] / 100.;
        OptionOutcomes { price: model_price, delta, gamma, theta, rho, vega }
    }

    fn price_vega(&self, volatility: f64) -> Dual2_64 {
        let spot = self.spot.re.into();
        let strike = self.strike.re.into();
        let volatility = Dual2_64::from_re(volatility).derivative();
        let risk_free_rate = self.risk_free_rate.re.into();
        let dividend_rate = self.dividend_rate.re.into();
        let time_to_maturity = self.time_to_maturity.re.into();
        let price = match self.option_type {
            OptionType::Call => BjerksundStensland::new(
                spot, strike, volatility, risk_free_rate, dividend_rate, time_to_maturity
            ).price_call(),
            OptionType::Put => BjerksundStensland::new(
                strike, spot, volatility, dividend_rate, risk_free_rate, time_to_maturity
            ).price_call(),
        };
        price
    }

    pub fn iv(&self, price: f64) -> Result<f64, roots::SearchError> {
        let f = |x| {
            let price_dual = self.price_vega(x);
            (price_dual.re - price, price_dual.v1)
        };

        use crate::newton::find_root_newton_raphson_dual;
        let mut convergency = roots::SimpleConvergency { eps:1e-3f64, max_iter:1000 };
        let mut root = find_root_newton_raphson_dual(1e-3f64,  &f, &mut convergency);
        if root.is_err() {
            root = find_root_newton_raphson_dual(1e-2f64, &f, &mut convergency);
        }
        if root.is_err() {
            root = find_root_newton_raphson_dual(1e-1f64, &f, &mut convergency);
        }
        if root.is_err() {
            root = find_root_newton_raphson_dual(1f64, &f, &mut convergency);
        }
        if root.is_err() {
            root = find_root_newton_raphson_dual(10f64, &f, &mut convergency);
        }
        root
    }
}
