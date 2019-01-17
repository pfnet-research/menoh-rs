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
    UnsupportedInputDims(String),
    SameNamedParameterAlreadyExist(String),
    SameNamedAttributeAlreadyExist(String),
    InvalidBackendConfigError(String),
    InputNotFoundError(String),
    OutputNotFoundError(String),

    DtypeMismatch {
        /// Actual dtype.
        actual: menoh_sys::menoh_dtype,
        /// Requested dtype.
        expected: menoh_sys::menoh_dtype,
    },
    NulError(ffi::NulError),
}
use Error::*;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            StdError(message)
            | UnknownError(message)
            | InvalidFilename(message)
            | UnsupportedOnnxOpsetVersion(message)
            | OnnxParseError(message)
            | InvalidDtype(message)
            | InvalidAttributeType(message)
            | UnsupportedOperatorAttribute(message)
            | DimensionMismatch(message)
            | VariableNotFound(message)
            | IndexOutOfRange(message)
            | JsonParseError(message)
            | InvalidBackendName(message)
            | UnsupportedOperator(message)
            | FailedToConfigureOperator(message)
            | BackendError(message)
            | SameNamedVariableAlreadyExist(message)
            | UnsupportedInputDims(message)
            | SameNamedParameterAlreadyExist(message)
            | SameNamedAttributeAlreadyExist(message)
            | InvalidBackendConfigError(message)
            | InputNotFoundError(message)
            | OutputNotFoundError(message) => write!(f, "{}", message),

            DtypeMismatch { actual, expected } => write!(
                f,
                "menoh dtype mismatch error: actural {}, expected {}",
                actual, expected
            ),
            NulError(err) => err.fmt(f),
        }
    }
}

impl error::Error for Error {}

pub fn check(code: menoh_sys::menoh_error_code) -> Result<(), Error> {
    let code = code as menoh_sys::menoh_error_code_constant;

    if code == menoh_sys::menoh_error_code_success {
        Ok(())
    } else {
        let message = String::from_utf8_lossy(
            unsafe { ffi::CStr::from_ptr(menoh_sys::menoh_get_last_error_message()) }.to_bytes(),
        )
        .into_owned();
        let err = match code {
            menoh_sys::menoh_error_code_std_error => StdError(message),
            menoh_sys::menoh_error_code_unknown_error => UnknownError(message),
            menoh_sys::menoh_error_code_invalid_filename => InvalidFilename(message),
            menoh_sys::menoh_error_code_unsupported_onnx_opset_version => {
                UnsupportedOnnxOpsetVersion(message)
            }
            menoh_sys::menoh_error_code_onnx_parse_error => OnnxParseError(message),
            menoh_sys::menoh_error_code_invalid_dtype => InvalidDtype(message),
            menoh_sys::menoh_error_code_invalid_attribute_type => InvalidAttributeType(message),
            menoh_sys::menoh_error_code_unsupported_operator_attribute => {
                UnsupportedOperatorAttribute(message)
            }
            menoh_sys::menoh_error_code_dimension_mismatch => DimensionMismatch(message),
            menoh_sys::menoh_error_code_variable_not_found => VariableNotFound(message),
            menoh_sys::menoh_error_code_index_out_of_range => IndexOutOfRange(message),
            menoh_sys::menoh_error_code_json_parse_error => JsonParseError(message),
            menoh_sys::menoh_error_code_invalid_backend_name => InvalidBackendName(message),
            menoh_sys::menoh_error_code_unsupported_operator => UnsupportedOperator(message),
            menoh_sys::menoh_error_code_failed_to_configure_operator => {
                FailedToConfigureOperator(message)
            }
            menoh_sys::menoh_error_code_backend_error => BackendError(message),
            menoh_sys::menoh_error_code_same_named_variable_already_exist => {
                SameNamedVariableAlreadyExist(message)
            }
            menoh_sys::menoh_error_code_unsupported_input_dims => UnsupportedInputDims(message),
            menoh_sys::menoh_error_code_same_named_parameter_already_exist => {
                SameNamedParameterAlreadyExist(message)
            }
            menoh_sys::menoh_error_code_same_named_attribute_already_exist => {
                SameNamedAttributeAlreadyExist(message)
            }
            menoh_sys::menoh_error_code_invalid_backend_config_error => {
                InvalidBackendConfigError(message)
            }
            menoh_sys::menoh_error_code_input_not_found_error => InputNotFoundError(message),
            menoh_sys::menoh_error_code_output_not_found_error => OutputNotFoundError(message),
            _ => unreachable!(),
        };
        Err(err)
    }
}

impl From<ffi::NulError> for Error {
    fn from(value: ffi::NulError) -> Self {
        Error::NulError(value)
    }
}
