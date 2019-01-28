use crate::Error;

/// Representation of scalar types supported by Menoh.
pub unsafe trait Dtype {
    /// Integer specifying the scalar type.
    ///
    /// ```
    /// # extern crate menoh;
    /// # extern crate menoh_sys;
    /// # use menoh::*;
    /// assert_eq!(f32::ID, menoh_sys::menoh_dtype_float as menoh_sys::menoh_dtype);
    /// ```
    const ID: menoh_sys::menoh_dtype;

    /// Verify a scalar type.
    ///
    /// ```
    /// # use menoh::*;
    /// assert!(f32::check(f32::ID).is_ok());
    /// assert!(f32::check(f32::ID + 1).is_err());
    /// ```
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
