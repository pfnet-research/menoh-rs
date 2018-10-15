use menoh_sys;
use std::ffi;

use error::check;
use handler::Handler;
use Error;

/// Container of variable profiles (type, shape and flag of input/output).
pub struct VariableProfileTable {
    handle: menoh_sys::menoh_variable_profile_table_handle,
}

impl VariableProfileTable {
    /// Fetch the shape of variable.
    ///
    /// ```
    /// # use menoh::*;
    /// # fn main() -> Result<(), Error> {
    /// # let mut model_data = ModelData::from_onnx("MLP.onnx")?;
    /// # let mut vpt_builder = VariableProfileTableBuilder::new()?;
    /// # vpt_builder.add_input::<f32>("input", &[2, 3])?;
    /// # vpt_builder.add_output("fc2")?;
    /// # let vpt = vpt_builder.build(&model_data)?;
    /// let dims = vpt.get_variable_dims("fc2")?;
    /// # assert_eq!(dims, &[2, 5]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_variable_dims(&self, name: &str) -> Result<Vec<usize>, Error> {
        let name = ffi::CString::new(name)?;
        unsafe {
            let mut size = 0;
            check(menoh_sys::menoh_variable_profile_table_get_dims_size(
                self.handle,
                name.as_ptr(),
                &mut size,
            ))?;
            let mut dims = Vec::with_capacity(size as _);
            for index in 0..size {
                let mut dim = 0;
                check(menoh_sys::menoh_variable_profile_table_get_dims_at(
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
