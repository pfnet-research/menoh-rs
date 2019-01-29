use crate::error::check;
use crate::Error;
use crate::VariableProfileTable;
use std::ffi::CString;
use std::path::Path;
use std::ptr;

/// Container of operators and values of constant variables.
pub struct ModelData {
    pub(crate) handle: menoh_sys::menoh_model_data_handle,
}

impl ModelData {
    /// Load data from a ONNX file.
    ///
    /// ```
    /// # use menoh::*;
    /// # fn main() -> Result<(), Error> {
    /// let model_data = ModelData::from_onnx("MLP.onnx")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_onnx<P>(path: P) -> Result<Self, Error>
    where
        P: AsRef<Path>,
    {
        let path = CString::new::<&str>(&path.as_ref().to_string_lossy())?;
        let mut handle = ptr::null_mut();
        unsafe {
            check(menoh_sys::menoh_make_model_data_from_onnx(
                path.as_ptr(),
                &mut handle,
            ))?
        };
        Ok(Self { handle })
    }

    /// Load data from a ONNX data.
    ///
    /// ```
    /// # use menoh::*;
    /// # fn main() -> Result<(), Error> {
    /// # let onnx_data = include_bytes!("../MLP.onnx");
    /// let model_data = ModelData::from_onnx_bytes(onnx_data)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_onnx_bytes(data: &[u8]) -> Result<Self, Error> {
        let mut handle = ptr::null_mut();
        unsafe {
            check(menoh_sys::menoh_make_model_data_from_onnx_data_on_memory(
                data.as_ptr(),
                data.len() as _,
                &mut handle,
            ))?
        };
        Ok(Self { handle })
    }

    /// Remove unused data using a `VariableProfileTable`.
    ///
    /// ```
    /// # use menoh::*;
    /// # fn main() -> Result<(), Error> {
    /// # let mut model_data = ModelData::from_onnx("MLP.onnx")?;
    /// # let mut vpt_builder = VariableProfileTableBuilder::new()?;
    /// # vpt_builder.add_input::<f32>("input", &[2, 3])?;
    /// # vpt_builder.add_output("fc2")?;
    /// # let vpt = vpt_builder.build(&model_data)?;
    /// model_data.optimize(&vpt)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn optimize(&mut self, variable_profile_table: &VariableProfileTable) -> Result<(), Error> {
        unsafe {
            check(menoh_sys::menoh_model_data_optimize(
                self.handle,
                variable_profile_table.handle,
            ))
        }
    }
}

impl Drop for ModelData {
    fn drop(&mut self) {
        unsafe { menoh_sys::menoh_delete_model_data(self.handle) }
    }
}

#[cfg(test)]
mod tests;
