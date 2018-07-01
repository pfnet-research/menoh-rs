use menoh_sys;
use std::ffi;
use std::ptr;

use dtype::Dtype;
use Error;
use error::check;
use ModelData;
use VariableProfileTable;

pub struct VariableProfileTableBuilder {
    handle: menoh_sys::menoh_variable_profile_table_builder_handle,
}

impl VariableProfileTableBuilder {
    pub fn new() -> Result<Self, Error> {
        let mut handle = ptr::null_mut();
        unsafe { check(menoh_sys::menoh_make_variable_profile_table_builder(&mut handle))? };
        Ok(Self { handle })
    }

    pub fn add_input<T>(&mut self, name: &str, dims: &[usize]) -> Result<(), Error>
        where T: Dtype
    {
        let c_name = ffi::CString::new(name).map_err(|_| Error::NulError)?;
        match dims.len() {
            2 => unsafe {
                check(menoh_sys::menoh_variable_profile_table_builder_add_input_profile_dims_2(
                    self.handle, c_name.as_ptr(), T::ID,
                    dims[0] as _, dims[1] as _))
            },
            4 => unsafe {
                check(menoh_sys::menoh_variable_profile_table_builder_add_input_profile_dims_4(
                    self.handle, c_name.as_ptr(), T::ID,
                    dims[0] as _, dims[1] as _, dims[2] as _, dims[3] as _))
            },
            _ => {
                Err(Error::InvalidDimsSize(format!("menoh invalid dims size error (2 or 4 is valid): dims size of {} is specified {}",
                                                   name,
                                                   dims.len())))
            }
        }
    }

    pub fn add_output<T>(&mut self, name: &str) -> Result<(), Error>
        where T: Dtype
    {
        let name = ffi::CString::new(name).map_err(|_| Error::NulError)?;
        unsafe {
            check(menoh_sys::menoh_variable_profile_table_builder_add_output_profile(
                self.handle, name.as_ptr(), T::ID))
        }
    }

    pub fn build(&self, model_data: &ModelData) -> Result<VariableProfileTable, Error> {
        let mut handle = ptr::null_mut();
        unsafe {
            check(menoh_sys::menoh_build_variable_profile_table(self.handle,
                                                                model_data.handle(),
                                                                &mut handle))?;
            Ok(VariableProfileTable::from_handle(handle))
        }
    }
}

impl Drop for VariableProfileTableBuilder {
    fn drop(&mut self) {
        unsafe { menoh_sys::menoh_delete_variable_profile_table_builder(self.handle) }
    }
}
