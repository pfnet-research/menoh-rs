#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

/* automatically generated by rust-bindgen */

pub const menoh_dtype_float: menoh_dtype_constant = 0;
pub type menoh_dtype_constant = u32;
pub type menoh_dtype = i32;
pub const menoh_error_code_success: menoh_error_code_constant = 0;
pub const menoh_error_code_std_error: menoh_error_code_constant = 1;
pub const menoh_error_code_unknown_error: menoh_error_code_constant = 2;
pub const menoh_error_code_invalid_filename: menoh_error_code_constant = 3;
pub const menoh_error_code_unsupported_onnx_opset_version: menoh_error_code_constant = 4;
pub const menoh_error_code_onnx_parse_error: menoh_error_code_constant = 5;
pub const menoh_error_code_invalid_dtype: menoh_error_code_constant = 6;
pub const menoh_error_code_invalid_attribute_type: menoh_error_code_constant = 7;
pub const menoh_error_code_unsupported_operator_attribute: menoh_error_code_constant = 8;
pub const menoh_error_code_dimension_mismatch: menoh_error_code_constant = 9;
pub const menoh_error_code_variable_not_found: menoh_error_code_constant = 10;
pub const menoh_error_code_index_out_of_range: menoh_error_code_constant = 11;
pub const menoh_error_code_json_parse_error: menoh_error_code_constant = 12;
pub const menoh_error_code_invalid_backend_name: menoh_error_code_constant = 13;
pub const menoh_error_code_unsupported_operator: menoh_error_code_constant = 14;
pub const menoh_error_code_failed_to_configure_operator: menoh_error_code_constant = 15;
pub const menoh_error_code_backend_error: menoh_error_code_constant = 16;
pub const menoh_error_code_same_named_variable_already_exist: menoh_error_code_constant = 17;
pub const menoh_error_code_unsupported_input_dims: menoh_error_code_constant = 18;
pub const menoh_error_code_same_named_parameter_already_exist: menoh_error_code_constant = 19;
pub const menoh_error_code_same_named_attribute_already_exist: menoh_error_code_constant = 20;
pub const menoh_error_code_invalid_backend_config_error: menoh_error_code_constant = 21;
pub const menoh_error_code_input_not_found_error: menoh_error_code_constant = 22;
pub const menoh_error_code_output_not_found_error: menoh_error_code_constant = 23;
pub type menoh_error_code_constant = u32;
pub type menoh_error_code = i32;
extern "C" {
    pub fn menoh_get_last_error_message() -> *const ::std::os::raw::c_char;
}
#[repr(C)]
pub struct menoh_model_data {
    _unused: [u8; 0],
}
pub type menoh_model_data_handle = *mut menoh_model_data;
extern "C" {
    pub fn menoh_delete_model_data(model_data: menoh_model_data_handle);
}
extern "C" {
    pub fn menoh_make_model_data_from_onnx(
        onnx_filename: *const ::std::os::raw::c_char,
        dst_handle: *mut menoh_model_data_handle,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_make_model_data_from_onnx_data_on_memory(
        onnx_data: *const u8,
        size: i32,
        dst_handle: *mut menoh_model_data_handle,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_make_model_data(dst_handle: *mut menoh_model_data_handle) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_model_data_add_parameter(
        model_data: menoh_model_data_handle,
        parameter_name: *const ::std::os::raw::c_char,
        dtype: menoh_dtype,
        dims_size: i32,
        dims: *const i32,
        buffer_handle: *mut ::std::os::raw::c_void,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_model_data_add_new_node(
        model_data: menoh_model_data_handle,
        op_type: *const ::std::os::raw::c_char,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_model_data_add_input_name_to_current_node(
        model_data: menoh_model_data_handle,
        input_name: *const ::std::os::raw::c_char,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_model_data_add_output_name_to_current_node(
        model_data: menoh_model_data_handle,
        output_name: *const ::std::os::raw::c_char,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_model_data_add_attribute_int_to_current_node(
        model_data: menoh_model_data_handle,
        attribute_name: *const ::std::os::raw::c_char,
        value: i32,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_model_data_add_attribute_float_to_current_node(
        model_data: menoh_model_data_handle,
        attribute_name: *const ::std::os::raw::c_char,
        value: f32,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_model_data_add_attribute_ints_to_current_node(
        model_data: menoh_model_data_handle,
        attribute_name: *const ::std::os::raw::c_char,
        size: i32,
        value: *const i32,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_model_data_add_attribute_floats_to_current_node(
        model_data: menoh_model_data_handle,
        attribute_name: *const ::std::os::raw::c_char,
        size: i32,
        value: *const f32,
    ) -> menoh_error_code;
}
#[repr(C)]
pub struct menoh_variable_profile_table_builder {
    _unused: [u8; 0],
}
pub type menoh_variable_profile_table_builder_handle = *mut menoh_variable_profile_table_builder;
extern "C" {
    pub fn menoh_make_variable_profile_table_builder(
        dst_handle: *mut menoh_variable_profile_table_builder_handle,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_delete_variable_profile_table_builder(
        builder: menoh_variable_profile_table_builder_handle,
    );
}
extern "C" {
    pub fn menoh_variable_profile_table_builder_add_input_profile(
        builder: menoh_variable_profile_table_builder_handle,
        name: *const ::std::os::raw::c_char,
        dtype: menoh_dtype,
        dims_size: i32,
        dims: *const i32,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_variable_profile_table_builder_add_input_profile_dims_2(
        builder: menoh_variable_profile_table_builder_handle,
        name: *const ::std::os::raw::c_char,
        dtype: menoh_dtype,
        num: i32,
        size: i32,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_variable_profile_table_builder_add_input_profile_dims_4(
        builder: menoh_variable_profile_table_builder_handle,
        name: *const ::std::os::raw::c_char,
        dtype: menoh_dtype,
        num: i32,
        channel: i32,
        height: i32,
        width: i32,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_variable_profile_table_builder_add_output_name(
        builder: menoh_variable_profile_table_builder_handle,
        name: *const ::std::os::raw::c_char,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_variable_profile_table_builder_add_output_profile(
        builder: menoh_variable_profile_table_builder_handle,
        name: *const ::std::os::raw::c_char,
        dtype: menoh_dtype,
    ) -> menoh_error_code;
}
#[repr(C)]
pub struct menoh_variable_profile_table {
    _unused: [u8; 0],
}
pub type menoh_variable_profile_table_handle = *mut menoh_variable_profile_table;
extern "C" {
    pub fn menoh_build_variable_profile_table(
        builder: menoh_variable_profile_table_builder_handle,
        model_data: menoh_model_data_handle,
        dst_handle: *mut menoh_variable_profile_table_handle,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_delete_variable_profile_table(
        variable_profile_table: menoh_variable_profile_table_handle,
    );
}
extern "C" {
    pub fn menoh_variable_profile_table_get_dtype(
        variable_profile_table: menoh_variable_profile_table_handle,
        variable_name: *const ::std::os::raw::c_char,
        dst_dtype: *mut menoh_dtype,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_variable_profile_table_get_dims_size(
        variable_profile_table: menoh_variable_profile_table_handle,
        variable_name: *const ::std::os::raw::c_char,
        dst_size: *mut i32,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_variable_profile_table_get_dims_at(
        variable_profile_table: menoh_variable_profile_table_handle,
        variable_name: *const ::std::os::raw::c_char,
        index: i32,
        dst_size: *mut i32,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_model_data_optimize(
        model_data: menoh_model_data_handle,
        variable_profile_table: menoh_variable_profile_table_handle,
    ) -> menoh_error_code;
}
#[repr(C)]
pub struct menoh_model_builder {
    _unused: [u8; 0],
}
pub type menoh_model_builder_handle = *mut menoh_model_builder;
extern "C" {
    pub fn menoh_make_model_builder(
        variable_profile_table: menoh_variable_profile_table_handle,
        dst_handle: *mut menoh_model_builder_handle,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_delete_model_builder(model_builder: menoh_model_builder_handle);
}
extern "C" {
    pub fn menoh_model_builder_attach_external_buffer(
        builder: menoh_model_builder_handle,
        variable_name: *const ::std::os::raw::c_char,
        buffer_handle: *mut ::std::os::raw::c_void,
    ) -> menoh_error_code;
}
#[repr(C)]
pub struct menoh_model {
    _unused: [u8; 0],
}
pub type menoh_model_handle = *mut menoh_model;
extern "C" {
    pub fn menoh_build_model(
        builder: menoh_model_builder_handle,
        model_data: menoh_model_data_handle,
        backend_name: *const ::std::os::raw::c_char,
        backend_config: *const ::std::os::raw::c_char,
        dst_model_handle: *mut menoh_model_handle,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_delete_model(model: menoh_model_handle);
}
extern "C" {
    pub fn menoh_model_get_variable_buffer_handle(
        model: menoh_model_handle,
        variable_name: *const ::std::os::raw::c_char,
        dst_data: *mut *mut ::std::os::raw::c_void,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_model_get_variable_dtype(
        model: menoh_model_handle,
        variable_name: *const ::std::os::raw::c_char,
        dst_dtype: *mut menoh_dtype,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_model_get_variable_dims_size(
        model: menoh_model_handle,
        variable_name: *const ::std::os::raw::c_char,
        dst_size: *mut i32,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_model_get_variable_dims_at(
        model: menoh_model_handle,
        variable_name: *const ::std::os::raw::c_char,
        index: i32,
        dst_size: *mut i32,
    ) -> menoh_error_code;
}
extern "C" {
    pub fn menoh_model_run(model: menoh_model_handle) -> menoh_error_code;
}
