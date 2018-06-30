use menoh_sys;

pub type DtypeCode = menoh_sys::menoh_dtype_constant;

pub trait Dtype {
    const CODE: DtypeCode;
}

impl Dtype for f32 {
    const CODE: DtypeCode = menoh_sys::menoh_dtype_float as _;
}
