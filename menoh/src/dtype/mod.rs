use menoh_sys;

use Error;

/// Trait representing scalar types supported by Menoh.
pub unsafe trait Dtype {
    const ID: menoh_sys::menoh_dtype;

    fn check(dtype: menoh_sys::menoh_dtype) -> Result<(), Error> {
        if dtype == Self::ID {
            Ok(())
        } else {
            Err(Error::DtypeMismatch {
                    actual: dtype,
                    expected: Self::ID,
                })
        }
    }
}

unsafe impl Dtype for f32 {
    const ID: menoh_sys::menoh_dtype = menoh_sys::menoh_dtype_float as _;
}

#[cfg(test)]
mod tests;
