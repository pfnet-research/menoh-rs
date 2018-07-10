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

    InvalidDimsSize {
        /// Name of the variable.
        name: String,
        /// Size of the specified dims.
        size: usize,
    },
    DtypeMismatch {
        /// Actual dtype.
        actual: menoh_sys::menoh_dtype,
        /// Requested dtype.
        expected: menoh_sys::menoh_dtype,
    },
    NulError(ffi::NulError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::StdError(message) => write!(f, "{}", message),
            Error::UnknownError(message) => write!(f, "{}", message),
            Error::InvalidFilename(message) => write!(f, "{}", message),
            Error::UnsupportedOnnxOpsetVersion(message) => write!(f, "{}", message),
            Error::OnnxParseError(message) => write!(f, "{}", message),
            Error::InvalidDtype(message) => write!(f, "{}", message),
            Error::InvalidAttributeType(message) => write!(f, "{}", message),
            Error::UnsupportedOperatorAttribute(message) => write!(f, "{}", message),
            Error::DimensionMismatch(message) => write!(f, "{}", message),
            Error::VariableNotFound(message) => write!(f, "{}", message),
            Error::IndexOutOfRange(message) => write!(f, "{}", message),
            Error::JsonParseError(message) => write!(f, "{}", message),
            Error::InvalidBackendName(message) => write!(f, "{}", message),
            Error::UnsupportedOperator(message) => write!(f, "{}", message),
            Error::FailedToConfigureOperator(message) => write!(f, "{}", message),
            Error::BackendError(message) => write!(f, "{}", message),
            Error::SameNamedVariableAlreadyExist(message) => write!(f, "{}", message),
            Error::InvalidDimsSize { name, size } => {
                write!(f,
                       "menoh invalid dims size error (2 or 4 is valid): dims size of {} is specified {}",
                       name,
                       size)
            }
            Error::DtypeMismatch { actual, expected } => {
                write!(f,
                       "menoh dtype mismatch error: actural {}, expected {}",
                       actual,
                       expected)
            }
            Error::NulError(err) => err.fmt(f),
        }
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

impl From<ffi::NulError> for Error {
    fn from(value: ffi::NulError) -> Self {
        Error::NulError(value)
    }
}
