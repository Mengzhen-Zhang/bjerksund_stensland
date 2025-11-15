mod no_ad;
mod forward_ad;
mod newton;

pub use no_ad::{
    bs_call_price,
    bs_put_price,
    bs_call_iv,
    bs_put_iv
};

pub use forward_ad::{
    AmericanCall,
    AmericanPut,
    bs_call_price_dual2,
};
