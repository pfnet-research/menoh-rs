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

impl Error {
    fn from_raw(code: menoh_sys::menoh_error_code) -> Option<Self> {
        let code = code as menoh_sys::menoh_error_code_constant;

        if code == menoh_sys::menoh_error_code_success {
            return None;
        }

        let message = unsafe {
            ffi::CStr::from_ptr(menoh_sys::menoh_get_last_error_message())
                .to_owned()
                .into_string()
                .unwrap()
        };
        match code {
            menoh_sys::menoh_error_code_std_error => Some(Error::StdError(message)),
            menoh_sys::menoh_error_code_unknown_error => Some(Error::UnknownError(message)),
            menoh_sys::menoh_error_code_invalid_filename => Some(Error::InvalidFilename(message)),
            menoh_sys::menoh_error_code_unsupported_onnx_opset_version => {
                Some(Error::UnsupportedOnnxOpsetVersion(message))
            }
            menoh_sys::menoh_error_code_onnx_parse_error => Some(Error::OnnxParseError(message)),
            menoh_sys::menoh_error_code_invalid_dtype => Some(Error::InvalidDtype(message)),
            menoh_sys::menoh_error_code_invalid_attribute_type => {
                Some(Error::InvalidAttributeType(message))
            }
            menoh_sys::menoh_error_code_unsupported_operator_attribute => {
                Some(Error::UnsupportedOperatorAttribute(message))
            }
            menoh_sys::menoh_error_code_dimension_mismatch => {
                Some(Error::DimensionMismatch(message))
            }
            menoh_sys::menoh_error_code_variable_not_found => {
                Some(Error::VariableNotFound(message))
            }
            menoh_sys::menoh_error_code_index_out_of_range => Some(Error::IndexOutOfRange(message)),
            menoh_sys::menoh_error_code_json_parse_error => Some(Error::JsonParseError(message)),
            menoh_sys::menoh_error_code_invalid_backend_name => {
                Some(Error::InvalidBackendName(message))
            }
            menoh_sys::menoh_error_code_unsupported_operator => {
                Some(Error::UnsupportedOperator(message))
            }
            menoh_sys::menoh_error_code_failed_to_configure_operator => {
                Some(Error::FailedToConfigureOperator(message))
            }
            menoh_sys::menoh_error_code_backend_error => Some(Error::BackendError(message)),
            menoh_sys::menoh_error_code_same_named_variable_already_exist => {
                Some(Error::SameNamedVariableAlreadyExist(message))
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

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl error::Error for Error {}
