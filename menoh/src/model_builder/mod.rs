use menoh_sys;
use std::ffi;
use std::ptr;

use Error;
use error::check;
use handler::Handler;
use Model;
use ModelData;
use VariableProfileTable;

/// Builder for `Model`.
pub struct ModelBuilder {
    handle: menoh_sys::menoh_model_builder_handle,
}

impl ModelBuilder {
    /// Create a builder using a `VariableProfileTable`.
    ///
    /// ```
    /// # use menoh::*;
    /// # fn main() -> Result<(), Error> {
    /// # let mut model_data = ModelData::from_onnx("MLP.onnx")?;
    /// # let mut vpt_builder = VariableProfileTableBuilder::new()?;
    /// # vpt_builder.add_input::<f32>("input", &[2, 3])?;
    /// # vpt_builder.add_output::<f32>("fc2")?;
    /// # let vpt = vpt_builder.build(&model_data)?;
    /// let model_builder = ModelBuilder::new(&vpt)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(variable_profile_table: &VariableProfileTable) -> Result<Self, Error> {
        let mut handle = ptr::null_mut();
        unsafe {
            check(menoh_sys::menoh_make_model_builder(variable_profile_table.handle(),
                                                      &mut handle))?;
        }
        Ok(Self { handle })
    }

    /// Build a `Model` from a `ModelData`.
    ///
    /// ```
    /// # use menoh::*;
    /// # fn main() -> Result<(), Error> {
    /// # let mut model_data = ModelData::from_onnx("MLP.onnx")?;
    /// # let mut vpt_builder = VariableProfileTableBuilder::new()?;
    /// # vpt_builder.add_input::<f32>("input", &[2, 3])?;
    /// # vpt_builder.add_output::<f32>("fc2")?;
    /// # let vpt = vpt_builder.build(&model_data)?;
    /// # model_data.optimize(&vpt)?;
    /// # let model_builder = ModelBuilder::new(&vpt)?;
    /// let model = model_builder.build(model_data, "mkldnn", "")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self,
                 model_data: ModelData,
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
