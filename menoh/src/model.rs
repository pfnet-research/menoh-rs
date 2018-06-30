use menoh_sys;

pub struct Model {
    handle: menoh_sys::menoh_model_handle,
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { menoh_sys::menoh_delete_model(self.handle) }
    }
}
