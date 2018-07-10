use menoh_sys;
use std::ffi;

use Error;
use error::check;
use handler::Handler;

/// Container of variable profiles (type, shape and flag of input/output).
pub struct VariableProfileTable {
    handle: menoh_sys::menoh_variable_profile_table_handle,
}

impl VariableProfileTable {
    /// Fetch the shape of variable.
    ///
    /// ```
    /// # fn main() -> Result<(), menoh::Error> {
    /// # let mut model_data = menoh::ModelData::from_onnx("test.onnx")?;
    /// # let mut vpt_builder = menoh:: VariableProfileTableBuilder::new()?;
    /// # vpt_builder.add_input::<f32>("139830916504208", &[2, 3])?;
    /// # vpt_builder.add_output::<f32>("139830916504880")?;
    /// # let vpt = vpt_builder.build(&model_data)?;
    /// let dims = vpt.get_variable_dims("139830916504880")?;
    /// # assert_eq!(dims, &[2, 5]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_variable_dims(&self, name: &str) -> Result<Vec<usize>, Error> {
        let name = ffi::CString::new(name)?;
        unsafe {
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
