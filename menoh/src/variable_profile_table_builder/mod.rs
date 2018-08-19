use menoh_sys;
use std::ffi;
use std::ptr;

use error::check;
use handler::Handler;
use Dtype;
use Error;
use ModelData;
use VariableProfileTable;

/// Builder for `VariableProfileTable`.
pub struct VariableProfileTableBuilder {
    handle: menoh_sys::menoh_variable_profile_table_builder_handle,
}

impl VariableProfileTableBuilder {
    /// Create a builder.
    ///
    /// ```
    /// use menoh::*;
    /// # fn main() -> Result<(), Error> {
    /// let vpt_builder = VariableProfileTableBuilder::new()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new() -> Result<Self, Error> {
        let mut handle = ptr::null_mut();
        unsafe {
            check(menoh_sys::menoh_make_variable_profile_table_builder(
                &mut handle,
            ))?
        };
        Ok(Self { handle })
    }

    /// Register a variable as input.
    ///
    /// ```
    /// use menoh::*;
    /// # fn main() -> Result<(), Error> {
    /// # let mut vpt_builder = VariableProfileTableBuilder::new()?;
    /// vpt_builder.add_input::<f32>("input", &[2, 3])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_input<T>(&mut self, name: &str, dims: &[usize]) -> Result<(), Error>
    where
        T: Dtype,
    {
        let c_name = ffi::CString::new(name)?;
        match dims.len() {
            2 => unsafe {
                check(
                    menoh_sys::menoh_variable_profile_table_builder_add_input_profile_dims_2(
                        self.handle,
                        c_name.as_ptr(),
                        T::ID,
                        dims[0] as _,
                        dims[1] as _,
                    ),
                )
            },
            4 => unsafe {
                check(
                    menoh_sys::menoh_variable_profile_table_builder_add_input_profile_dims_4(
                        self.handle,
                        c_name.as_ptr(),
                        T::ID,
                        dims[0] as _,
                        dims[1] as _,
                        dims[2] as _,
                        dims[3] as _,
                    ),
                )
            },
            _ => Err(Error::InvalidDimsSize {
                name: name.to_owned(),
                size: dims.len(),
            }),
        }
    }

    /// Register a variable as output.
    ///
    /// ```
    /// use menoh::*;
    /// # fn main() -> Result<(), Error> {
    /// # let mut vpt_builder = VariableProfileTableBuilder::new()?;
    /// # vpt_builder.add_input::<f32>("input", &[2, 3])?;
    /// vpt_builder.add_output::<f32>("fc2")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_output<T>(&mut self, name: &str) -> Result<(), Error>
    where
        T: Dtype,
    {
        let name = ffi::CString::new(name)?;
        unsafe {
            check(
                menoh_sys::menoh_variable_profile_table_builder_add_output_profile(
                    self.handle,
                    name.as_ptr(),
                    T::ID,
                ),
            )
        }
    }

    /// Build a `VariableProfileTable` using a `ModelData`.
    ///
    /// ```
    /// use menoh::*;
    /// # fn main() -> Result<(), Error> {
    /// # let mut model_data = ModelData::from_onnx("MLP.onnx")?;
    /// # let mut vpt_builder = VariableProfileTableBuilder::new()?;
    /// # vpt_builder.add_input::<f32>("input", &[2, 3])?;
    /// # vpt_builder.add_output::<f32>("fc2")?;
    /// let vpt = vpt_builder.build(&model_data)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self, model_data: &ModelData) -> Result<VariableProfileTable, Error> {
        let mut handle = ptr::null_mut();
        unsafe {
            check(menoh_sys::menoh_build_variable_profile_table(
                self.handle,
                model_data.handle(),
                &mut handle,
            ))?;
            Ok(VariableProfileTable::from_handle(handle))
        }
    }
}

impl Drop for VariableProfileTableBuilder {
    fn drop(&mut self) {
        unsafe { menoh_sys::menoh_delete_variable_profile_table_builder(self.handle) }
    }
}

#[cfg(test)]
mod tests;
