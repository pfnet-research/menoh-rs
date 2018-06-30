use menoh_sys;

#[derive(Clone, Copy)]
pub enum Dtype {
    Float,
}

impl Into<menoh_sys::menoh_dtype> for Dtype {
    fn into(self) -> menoh_sys::menoh_dtype {
        match self {
            Dtype::Float => menoh_sys::menoh_dtype_float as _,
        }
    }
}
