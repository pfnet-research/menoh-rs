use crate::error::check;
use crate::Dtype;
use crate::Error;
use std::ffi::CString;
use std::mem;
use std::ptr;
use std::slice;

/// Model, which executes computation.
pub struct Model {
    pub(crate) handle: menoh_sys::menoh_model_handle,
}

impl Model {
    /// Fetch the shape of a variable.
    ///
    /// ```
    /// # use menoh::*;
    /// # fn main() -> Result<(), Error> {
    /// # let model = Builder::from_onnx("MLP.onnx")?
    /// #     .add_input::<f32>("input", &[2, 3])?
    /// #     .add_output("fc2")?
    /// #     .build("mkldnn", "")?;
    /// let dims = model.get_variable_dims("fc2")?;
    /// # assert_eq!(dims, &[2, 5]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_variable_dims(&self, name: &str) -> Result<Vec<usize>, Error> {
        let name = CString::new(name)?;
        unsafe {
            let mut size = 0;
            check(menoh_sys::menoh_model_get_variable_dims_size(
                self.handle,
                name.as_ptr(),
                &mut size,
            ))?;
            let mut dims = Vec::with_capacity(size as _);
            for index in 0..size {
                let mut dim = 0;
                check(menoh_sys::menoh_model_get_variable_dims_at(
                    self.handle,
                    name.as_ptr(),
                    index,
                    &mut dim,
                ))?;
                dims.push(dim as _);
            }
            Ok(dims)
        }
    }

    fn get_variable_dtype(&self, name: &str) -> Result<menoh_sys::menoh_dtype, Error> {
        let name = CString::new(name)?;
        unsafe {
            let mut dtype = mem::uninitialized();
            check(menoh_sys::menoh_model_get_variable_dtype(
                self.handle,
                name.as_ptr(),
                &mut dtype,
            ))?;
            Ok(dtype)
        }
    }

    /// Fetch the shape and read-only view of a variable.
    ///
    /// ```
    /// # use menoh::*;
    /// # fn main() -> Result<(), Error> {
    /// # let model = Builder::from_onnx("MLP.onnx")?
    /// #     .add_input::<f32>("input", &[2, 3])?
    /// #     .add_output("fc2")?
    /// #     .build("mkldnn", "")?;
    /// let (dims, buf) = model.get_variable::<f32>("fc2")?;
    /// # assert_eq!(dims, &[2, 5]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_variable<T>(&self, name: &str) -> Result<(Vec<usize>, &[T]), Error>
    where
        T: Dtype,
    {
        T::check(self.get_variable_dtype(name)?)?;
        let dims = self.get_variable_dims(name)?;

        let name = CString::new(name)?;
        let mut buffer = ptr::null_mut();
        unsafe {
            check(menoh_sys::menoh_model_get_variable_buffer_handle(
                self.handle,
                name.as_ptr(),
                &mut buffer,
            ))?;
            let buffer = slice::from_raw_parts(buffer as _, dims.iter().product());
            Ok((dims, buffer))
        }
    }

    /// Fetch the shape and read/write view of a variable.
    ///
    /// ```
    /// # use menoh::*;
    /// # fn main() -> Result<(), Error> {
    /// # let mut model = Builder::from_onnx("MLP.onnx")?
    /// #     .add_input::<f32>("input", &[2, 3])?
    /// #     .add_output("fc2")?
    /// #     .build("mkldnn", "")?;
    /// let (dims, buf) = model.get_variable_mut::<f32>("fc2")?;
    /// # assert_eq!(dims, &[2, 5]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_variable_mut<T>(&mut self, name: &str) -> Result<(Vec<usize>, &mut [T]), Error>
    where
        T: Dtype,
    {
        T::check(self.get_variable_dtype(name)?)?;
        let dims = self.get_variable_dims(name)?;

        let name = CString::new(name)?;
        let mut buffer = ptr::null_mut();
        unsafe {
            check(menoh_sys::menoh_model_get_variable_buffer_handle(
                self.handle,
                name.as_ptr(),
                &mut buffer,
            ))?;
            let buffer = slice::from_raw_parts_mut(buffer as _, dims.iter().product());
            Ok((dims, buffer))
        }
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
