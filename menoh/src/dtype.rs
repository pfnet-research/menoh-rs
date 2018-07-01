use menoh_sys;

pub trait Dtype {
    const ID: menoh_sys::menoh_dtype;
}

impl Dtype for f32 {
    const ID: menoh_sys::menoh_dtype = menoh_sys::menoh_dtype_float as _;
}
