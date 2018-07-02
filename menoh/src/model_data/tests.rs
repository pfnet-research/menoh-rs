use super::ModelData;

#[test]
fn from_onnx() {
    ModelData::from_onnx("test.onnx").unwrap();
}

#[test]
#[should_panic(expected="InvalidFilename")]
fn from_onnx_invalid_path() {
    ModelData::from_onnx("invalid.onnx").unwrap();
}
