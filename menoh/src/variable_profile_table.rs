use menoh_sys;
use std::ffi;
use std::mem;

use Dtype;
use Error;
use error::check;

pub struct VariableProfile {
    pub dtype: Dtype,
    pub dims: Vec<usize>,
}

pub struct VariableProfileTable {
    handle: menoh_sys::menoh_variable_profile_table_handle,
}

impl VariableProfileTable {
    pub unsafe fn from_handle(handle: menoh_sys::menoh_variable_profile_table_handle) -> Self {
        Self { handle }
    }

    pub fn get(&self, name: &str) -> Result<VariableProfile, Error> {
        let name = ffi::CString::new(name).map_err(|_| Error::NulError)?;
        unsafe {
            let mut dtype = mem::uninitialized();
            check(menoh_sys::menoh_variable_profile_table_get_dtype(self.handle,
                                                                    name.as_ptr(),
                                                                    &mut dtype))?;
            let mut size = 0;
            check(menoh_sys::menoh_variable_profile_table_get_dims_size(self.handle,
                                                                        name.as_ptr(),
                                                                        &mut size))?;
            let mut dims = Vec::with_capacity(size as _);
            for index in 0..size {
                let mut dim = 0;
                check(menoh_sys::menoh_variable_profile_table_get_dims_at(self.handle,
                                                                          name.as_ptr(),
                                                                          index,
                                                                          &mut dim))?;
                dims.push(dim as _);
            }
            Ok(VariableProfile {
                   dtype: Dtype::from_raw(dtype),
                   dims,
               })
        }
    }

    pub unsafe fn handle(&self) -> menoh_sys::menoh_variable_profile_table_handle {
        self.handle
    }
}

impl Drop for VariableProfileTable {
    fn drop(&mut self) {
        unsafe { menoh_sys::menoh_delete_variable_profile_table(self.handle) }
    }
}
