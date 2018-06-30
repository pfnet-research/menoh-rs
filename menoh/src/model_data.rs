use menoh_sys;
use std::ffi;
use std::mem;
use std::path;
use std::ptr;

use error::check;
use Error;

pub struct ModelData {
    pub model_data: *mut menoh_sys::menoh_model_data,
}

impl ModelData {
    pub fn from_onnx<P>(path: P) -> Result<Self, Error>
        where P: AsRef<path::Path>
    {
        let path = ffi::CString::new(path.as_ref().as_os_str().to_string_lossy().as_ref())
            .map_err(|_| Error::InvalidFilename)?;
        let mut model_data = ptr::null_mut();
        unsafe {
            check(menoh_sys::menoh_make_model_data_from_onnx(path.as_ptr(), &mut model_data))?;
        }
        Ok(Self { model_data })
    }
}

impl Drop for ModelData {
    fn drop(&mut self) {
        unsafe { menoh_sys::menoh_delete_model_data(self.model_data) }
    }
}
