use menoh_sys;
use std::ffi;
use std::ptr;

use Dtype;
use Error;
use error::check;

pub struct VariableProfileTableBuilder {
    handle: menoh_sys::menoh_variable_profile_table_builder_handle,
}

impl VariableProfileTableBuilder {
    pub fn new() -> Result<Self, Error> {
        let mut handle = ptr::null_mut();
        unsafe { check(menoh_sys::menoh_make_variable_profile_table_builder(&mut handle))? };
        Ok(Self { handle })
    }

    pub fn add_input_profile_dims_2(&mut self,
                                    name: &str,
                                    dtype: Dtype,
                                    num: usize,
                                    size: usize)
                                    -> Result<(), Error> {
        let name = ffi::CString::new(name).map_err(|_| Error::NulError)?;
        unsafe {
            check(menoh_sys::menoh_variable_profile_table_builder_add_input_profile_dims_2(
                self.handle,
                name.as_ptr(),
                dtype.into(),
                num as _,
                size as _))?;
        }
        Ok(())
    }
}

impl Drop for VariableProfileTableBuilder {
    fn drop(&mut self) {
        unsafe { menoh_sys::menoh_delete_variable_profile_table_builder(self.handle) }
    }
}
