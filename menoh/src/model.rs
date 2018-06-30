use menoh_sys;
use std::ffi;
use std::os::raw::c_void;
use std::ptr;

use Error;
use error::check;

pub struct Model {
    handle: menoh_sys::menoh_model_handle,
}

impl Model {
    pub unsafe fn from_handle(handle: menoh_sys::menoh_model_handle) -> Self {
        Self { handle }
    }

    pub fn get_variable_buffer_handle(&self, name: &str) -> Result<*const c_void, Error> {
        let name = ffi::CString::new(name).map_err(|_| Error::NulError)?;
        let mut buffer = ptr::null_mut();
        unsafe {
            check(menoh_sys::menoh_model_get_variable_buffer_handle(self.handle,
                                                                    name.as_ptr(),
                                                                    &mut buffer))?;
        }
        Ok(buffer)
    }

    pub fn run(&mut self) -> Result<(), Error> {
        unsafe { check(menoh_sys::menoh_model_run(self.handle)) }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { menoh_sys::menoh_delete_model(self.handle) }
    }
}
