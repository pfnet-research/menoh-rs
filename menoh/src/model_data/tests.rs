use super::ModelData;

#[test]
#[should_panic(expected = "InvalidFilename")]
fn from_onnx_invalid_path() {
    ModelData::from_onnx("invalid.onnx").unwrap();
}

#[test]
#[should_panic(expected = "OnnxParseError")]
fn from_onnx_slice_invalid_data() {
    ModelData::from_onnx_slice(b"0123456789").unwrap();
}
