use menoh_sys;
use std::ffi;
use std::ptr;

use dtype::Dtype;
use Error;
use error::check;
use Model;
use ModelData;
use VariableProfileTable;

pub struct ModelBuilder {
    handle: menoh_sys::menoh_model_builder_handle,
}

impl ModelBuilder {
    pub fn new(variable_profile_table: &VariableProfileTable) -> Result<Self, Error> {
        let mut handle = ptr::null_mut();
        unsafe {
            check(menoh_sys::menoh_make_model_builder(variable_profile_table.handle(),
                                                      &mut handle))?;
        }
        Ok(Self { handle })
    }

    pub fn build(&self,
                 model_data: &ModelData,
                 backend_name: &str,
                 backend_config: &str)
                 -> Result<Model, Error> {
        let backend_name = ffi::CString::new(backend_name)?;
        let backend_config = ffi::CString::new(backend_config)?;
        let mut handle = ptr::null_mut();
        unsafe {
            check(menoh_sys::menoh_build_model(self.handle,
                                               model_data.handle(),
                                               backend_name.as_ptr(),
                                               backend_config.as_ptr(),
                                               &mut handle))?;
            Ok(Model::from_handle(handle))
        }
    }
}

impl Drop for ModelBuilder {
    fn drop(&mut self) {
        unsafe { menoh_sys::menoh_delete_model_builder(self.handle) }
    }
}
