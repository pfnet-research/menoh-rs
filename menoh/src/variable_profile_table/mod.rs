use menoh_sys;
use std::ffi;
use std::mem;

use dtype::Dtype;
use Error;
use error::check;
use handler::Handler;

pub struct VariableProfileTable {
    handle: menoh_sys::menoh_variable_profile_table_handle,
}

impl VariableProfileTable {
    pub fn get_variable_dims<T>(&self, name: &str) -> Result<Vec<usize>, Error>
        where T: Dtype
    {
        let name = ffi::CString::new(name)?;
        unsafe {
            let mut dtype = mem::uninitialized();
            check(menoh_sys::menoh_variable_profile_table_get_dtype(self.handle,
                                                                    name.as_ptr(),
                                                                    &mut dtype))?;
            T::check(dtype)?;
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
            Ok(dims)
        }
    }
}

impl Handler for VariableProfileTable {
    type Handle = menoh_sys::menoh_variable_profile_table_handle;
    unsafe fn from_handle(handle: Self::Handle) -> Self {
        Self { handle }
    }
    unsafe fn handle(&self) -> Self::Handle {
        self.handle
    }
}

impl Drop for VariableProfileTable {
    fn drop(&mut self) {
        unsafe { menoh_sys::menoh_delete_variable_profile_table(self.handle) }
    }
}
