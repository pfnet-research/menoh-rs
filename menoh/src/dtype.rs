use menoh_sys;

#[derive(Clone, Copy)]
pub enum Dtype {
    Float,
}

impl Dtype {
    pub fn from_raw(dtype: menoh_sys::menoh_dtype) -> Self {
        match dtype as _ {
            menoh_sys::menoh_dtype_float => Dtype::Float,
            _ => unreachable!(),
        }
    }
}

impl Into<menoh_sys::menoh_dtype> for Dtype {
    fn into(self) -> menoh_sys::menoh_dtype {
        match self {
            Dtype::Float => menoh_sys::menoh_dtype_float as _,
        }
    }
}
