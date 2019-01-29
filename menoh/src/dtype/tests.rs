use super::Dtype;

#[test]
fn check_f32() {
    f32::check(menoh_sys::menoh_dtype_float as _).unwrap();
}

#[test]
#[should_panic(expected = "DtypeMismatch")]
fn check_f32_invalid() {
    f32::check((menoh_sys::menoh_dtype_float + 1) as _).unwrap();
}
