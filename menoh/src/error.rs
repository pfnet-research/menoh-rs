use menoh_sys;

#[derive(Debug)]
pub enum Error {
    StdError,
    UnknownError,
    InvalidFilename,
    UnsupportedOnnxOpsetVersion,
    OnnxParseError,
    InvalidDtype,
    InvalidAttributeType,
    UnsupportedOperatorAttribute,
    DimensionMismatch,
    VariableNotFound,
    IndexOutOfRange,
    JsonParseError,
    InvalidBackendName,
    UnsupportedOperator,
    FailedToConfigureOperator,
    BackendError,
    SameNamedVariableAlreadyExist,
    InvalidDimsSize,
    NulError,
}

impl Error {
    fn from_raw(code: menoh_sys::menoh_error_code) -> Option<Self> {
        match code as _ {
            menoh_sys::menoh_error_code_success => None,
            menoh_sys::menoh_error_code_std_error => Some(Error::StdError),
            menoh_sys::menoh_error_code_unknown_error => Some(Error::UnknownError),
            menoh_sys::menoh_error_code_invalid_filename => Some(Error::InvalidFilename),
            menoh_sys::menoh_error_code_unsupported_onnx_opset_version => {
                Some(Error::UnsupportedOnnxOpsetVersion)
            }
            menoh_sys::menoh_error_code_onnx_parse_error => Some(Error::OnnxParseError),
            menoh_sys::menoh_error_code_invalid_dtype => Some(Error::InvalidDtype),
            menoh_sys::menoh_error_code_invalid_attribute_type => Some(Error::InvalidAttributeType),
            menoh_sys::menoh_error_code_unsupported_operator_attribute => {
                Some(Error::UnsupportedOperatorAttribute)
            }
            menoh_sys::menoh_error_code_dimension_mismatch => Some(Error::DimensionMismatch),
            menoh_sys::menoh_error_code_variable_not_found => Some(Error::VariableNotFound),
            menoh_sys::menoh_error_code_index_out_of_range => Some(Error::IndexOutOfRange),
            menoh_sys::menoh_error_code_json_parse_error => Some(Error::JsonParseError),
            menoh_sys::menoh_error_code_invalid_backend_name => Some(Error::InvalidBackendName),
            menoh_sys::menoh_error_code_unsupported_operator => Some(Error::UnsupportedOperator),
            menoh_sys::menoh_error_code_failed_to_configure_operator => {
                Some(Error::FailedToConfigureOperator)
            }
            menoh_sys::menoh_error_code_backend_error => Some(Error::BackendError),
            menoh_sys::menoh_error_code_same_named_variable_already_exist => {
                Some(Error::SameNamedVariableAlreadyExist)
            }
            _ => unreachable!(),
        }
    }
}

pub fn check(code: menoh_sys::menoh_error_code) -> Result<(), Error> {
    match Error::from_raw(code) {
        Some(err) => Err(err),
        None => Ok(()),
    }
}
