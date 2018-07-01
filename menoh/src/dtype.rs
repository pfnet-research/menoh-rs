use menoh_sys;

use Error;

pub trait Dtype {
    const ID: menoh_sys::menoh_dtype;

    fn check(dtype: menoh_sys::menoh_dtype) -> Result<(), Error> {
        if dtype == Self::ID {
            Ok(())
        } else {
            Err(Error::InvalidDtype(format!("menoh invalid dtype error: {}", dtype)))
        }
    }
}

impl Dtype for f32 {
    const ID: menoh_sys::menoh_dtype = menoh_sys::menoh_dtype_float as _;
}
