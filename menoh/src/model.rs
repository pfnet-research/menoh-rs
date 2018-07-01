use menoh_sys;
use std::ffi;
use std::mem;
use std::ptr;
use std::slice;

use dtype::Dtype;
use Error;
use error::check;

pub struct Model {
    handle: menoh_sys::menoh_model_handle,
}

impl Model {
    pub fn get_variable_dims<T>(&self, name: &str) -> Result<Vec<usize>, Error>
        where T: Dtype
    {
        let name = ffi::CString::new(name).map_err(|_| Error::NulError)?;
        unsafe {
            let mut dtype = mem::uninitialized();
            check(menoh_sys::menoh_model_get_variable_dtype(self.handle,
                                                            name.as_ptr(),
                                                            &mut dtype))?;
            T::check(dtype)?;
            let mut size = 0;
            check(menoh_sys::menoh_model_get_variable_dims_size(self.handle,
                                                                name.as_ptr(),
                                                                &mut size))?;
            let mut dims = Vec::with_capacity(size as _);
            for index in 0..size {
                let mut dim = 0;
                check(menoh_sys::menoh_model_get_variable_dims_at(self.handle,
                                                                  name.as_ptr(),
                                                                  index,
                                                                  &mut dim))?;
                dims.push(dim as _);
            }
            Ok(dims)
        }
    }

    pub fn get_variable<T>(&self, name: &str) -> Result<(Vec<usize>, &[T]), Error>
        where T: Dtype
    {
        let dims = self.get_variable_dims::<T>(name)?;
        let name = ffi::CString::new(name).map_err(|_| Error::NulError)?;
        let mut buffer = ptr::null_mut();
        unsafe {
            check(menoh_sys::menoh_model_get_variable_buffer_handle(self.handle,
                                                                    name.as_ptr(),
                                                                    &mut buffer))?;
            let buffer = slice::from_raw_parts(buffer as _, dims.iter().product());
            Ok((dims, buffer))
        }
    }

    pub fn get_variable_mut<T>(&mut self, name: &str) -> Result<(Vec<usize>, &mut [T]), Error>
        where T: Dtype
    {
        let dims = self.get_variable_dims::<T>(name)?;
        let name = ffi::CString::new(name).map_err(|_| Error::NulError)?;
        let mut buffer = ptr::null_mut();
        unsafe {
            check(menoh_sys::menoh_model_get_variable_buffer_handle(self.handle,
                                                                    name.as_ptr(),
                                                                    &mut buffer))?;
            let buffer = slice::from_raw_parts_mut(buffer as _, dims.iter().product());
            Ok((dims, buffer))
        }
    }

    pub fn run(&mut self) -> Result<(), Error> {
        unsafe { check(menoh_sys::menoh_model_run(self.handle)) }
    }

    pub unsafe fn from_handle(handle: menoh_sys::menoh_model_handle) -> Self {
        Self { handle }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { menoh_sys::menoh_delete_model(self.handle) }
    }
}
