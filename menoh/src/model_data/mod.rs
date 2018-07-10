use menoh_sys;
use std::ffi;
use std::path;
use std::ptr;

use Error;
use handler::Handler;
use error::check;
use VariableProfileTable;

/// Container of operators and values of constant variables.
pub struct ModelData {
    handle: menoh_sys::menoh_model_data_handle,
}

impl ModelData {
    pub fn from_onnx<P>(path: P) -> Result<Self, Error>
        where P: AsRef<path::Path>
    {
        let path = ffi::CString::new::<&str>(&path.as_ref().to_string_lossy())?;
        let mut handle = ptr::null_mut();
        unsafe { check(menoh_sys::menoh_make_model_data_from_onnx(path.as_ptr(), &mut handle))? };
        Ok(Self { handle })
    }

    pub fn optimize(&mut self, variable_profile_table: &VariableProfileTable) -> Result<(), Error> {
        unsafe {
            check(menoh_sys::menoh_model_data_optimize(self.handle,
                                                       variable_profile_table.handle()))
        }
    }
}

impl Handler for ModelData {
    type Handle = menoh_sys::menoh_model_data_handle;
    unsafe fn from_handle(handle: Self::Handle) -> Self {
        Self { handle }
    }
    unsafe fn handle(&self) -> Self::Handle {
        self.handle
    }
}

impl Drop for ModelData {
    fn drop(&mut self) {
        unsafe { menoh_sys::menoh_delete_model_data(self.handle) }
    }
}

#[cfg(test)]
mod tests;
