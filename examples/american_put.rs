use bjerksund_stensland::{bs_put_price, AmericanPut};

fn main() {
    let s = 90.0;
    let k = 100.0;
    let v = 0.25;
    let r = 0.05;
    let q = 0.03;
    let t = 1.0;
    let market_price = 12.0;

    let no_ad_price = bs_put_price(s, k, v, r, q, t);
    println!("Put Optoin Price (no autodiff): {}", no_ad_price);

    let call_option = AmericanPut::new(s, k, r, q, t);
    let outcome = call_option.price(v);
    let iv = call_option.iv(market_price).unwrap();
    println!("Put Option Pricing Outcomes: {:?}", outcome);
    println!("Market Price: {}", market_price);
    println!("Put Option IV: {}", iv);
}
