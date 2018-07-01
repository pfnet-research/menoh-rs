use menoh_sys;
use std::error;
use std::ffi;
use std::fmt;

#[derive(Debug)]
pub enum Error {
    StdError(String),
    UnknownError(String),
    InvalidFilename(String),
    UnsupportedOnnxOpsetVersion(String),
    OnnxParseError(String),
    InvalidDtype(String),
    InvalidAttributeType(String),
    UnsupportedOperatorAttribute(String),
    DimensionMismatch(String),
    VariableNotFound(String),
    IndexOutOfRange(String),
    JsonParseError(String),
    InvalidBackendName(String),
    UnsupportedOperator(String),
    FailedToConfigureOperator(String),
    BackendError(String),
    SameNamedVariableAlreadyExist(String),
    InvalidDimsSize(String),
    NulError,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl error::Error for Error {}

pub fn check(code: menoh_sys::menoh_error_code) -> Result<(), Error> {
    let code = code as menoh_sys::menoh_error_code_constant;

    if code == menoh_sys::menoh_error_code_success {
        Ok(())
    } else {
        let message = unsafe {
            ffi::CStr::from_ptr(menoh_sys::menoh_get_last_error_message())
                .to_owned()
                .into_string()
                .unwrap_or("[failed to decode message]".to_owned())
        };
        match code {
            menoh_sys::menoh_error_code_std_error => Err(Error::StdError(message)),
            menoh_sys::menoh_error_code_unknown_error => Err(Error::UnknownError(message)),
            menoh_sys::menoh_error_code_invalid_filename => Err(Error::InvalidFilename(message)),
            menoh_sys::menoh_error_code_unsupported_onnx_opset_version => {
                Err(Error::UnsupportedOnnxOpsetVersion(message))
            }
            menoh_sys::menoh_error_code_onnx_parse_error => Err(Error::OnnxParseError(message)),
            menoh_sys::menoh_error_code_invalid_dtype => Err(Error::InvalidDtype(message)),
            menoh_sys::menoh_error_code_invalid_attribute_type => {
                Err(Error::InvalidAttributeType(message))
            }
            menoh_sys::menoh_error_code_unsupported_operator_attribute => {
                Err(Error::UnsupportedOperatorAttribute(message))
            }
            menoh_sys::menoh_error_code_dimension_mismatch => {
                Err(Error::DimensionMismatch(message))
            }
            menoh_sys::menoh_error_code_variable_not_found => Err(Error::VariableNotFound(message)),
            menoh_sys::menoh_error_code_index_out_of_range => Err(Error::IndexOutOfRange(message)),
            menoh_sys::menoh_error_code_json_parse_error => Err(Error::JsonParseError(message)),
            menoh_sys::menoh_error_code_invalid_backend_name => {
                Err(Error::InvalidBackendName(message))
            }
            menoh_sys::menoh_error_code_unsupported_operator => {
                Err(Error::UnsupportedOperator(message))
            }
            menoh_sys::menoh_error_code_failed_to_configure_operator => {
                Err(Error::FailedToConfigureOperator(message))
            }
            menoh_sys::menoh_error_code_backend_error => Err(Error::BackendError(message)),
            menoh_sys::menoh_error_code_same_named_variable_already_exist => {
                Err(Error::SameNamedVariableAlreadyExist(message))
            }
            _ => unreachable!(),
        }
    }
}
