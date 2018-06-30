use menoh_sys;
use std::ffi;
use std::mem;

use error::check;
use Error;

pub struct ModelData {
    pub model_data: *mut menoh_sys::menoh_model_data,
}

impl ModelData {
    pub fn from_onnx(path: &str) -> Result<Self, Error> {
        let path = ffi::CString::new(path).unwrap();
        let mut model_data = unsafe { mem::uninitialized() };
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
