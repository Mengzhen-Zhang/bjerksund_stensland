use bjerksund_stensland::{bs_call_price, AmericanOption};

fn main() {
    use std::time::Instant;
    let now = Instant::now();

    let s = 100.0;
    let k = 90.0;
    let v = 0.25;
    let r = 0.05;
    let q = 0.03;
    let t = 1.0;
    let market_price = 12.0;

    let no_ad_price = bs_call_price(s, k, v, r, q, t);
    println!("Call Optoin Price (no autodiff): {}", no_ad_price);

    let call_option = AmericanOption::new(s, k, r, q, t, true);
    let outcome = call_option.price(v);
    let iv = call_option.iv(market_price).unwrap();
    println!("Call Option Pricing Outcomes: {:?}", outcome);
    println!("Market Price: {}", market_price);
    println!("Call Option IV: {}", iv);

    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
}
