use super::ModelData;

#[test]
#[should_panic(expected = "InvalidFilename")]
fn from_onnx_invalid_path() {
    ModelData::from_onnx("invalid.onnx").unwrap();
}
