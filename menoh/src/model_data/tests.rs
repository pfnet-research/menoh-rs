extern crate mktemp;

use super::ModelData;

#[test]
fn from_onnx() {
    let file = mktemp::Temp::new_file().unwrap();
    ModelData::from_onnx(file).unwrap();
}

#[test]
#[should_panic(expected="InvalidFilename")]
fn from_onnx_invalid_path() {
    ModelData::from_onnx("invalid.onnx").unwrap();
}
