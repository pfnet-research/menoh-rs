/// This trait makes it impossible to access internal handles from outside.
pub trait Handler {
    type Handle;
    unsafe fn from_handle(handle: Self::Handle) -> Self;
    unsafe fn handle(&self) -> Self::Handle;
}
