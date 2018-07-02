use super::VariableProfileTableBuilder;

#[test]
fn new() {
    VariableProfileTableBuilder::new().unwrap();
}

#[test]
fn add_input() {
    let mut vpt_builder = VariableProfileTableBuilder::new().unwrap();
    vpt_builder
        .add_input::<f32>("input", &[1, 3, 224, 224])
        .unwrap();
}

#[test]
#[should_panic(expected="InvalidDimsSize")]
fn add_input_invalid_dims() {
    let mut vpt_builder = VariableProfileTableBuilder::new().unwrap();
    vpt_builder
        .add_input::<f32>("input", &[1, 3, 224])
        .unwrap();
}

#[test]
#[should_panic(expected="NulError")]
fn add_input_invalid_name() {
    let mut vpt_builder = VariableProfileTableBuilder::new().unwrap();
    vpt_builder
        .add_input::<f32>("in\0put", &[1, 3, 224, 224])
        .unwrap();
}

#[test]
fn add_output() {
    let mut vpt_builder = VariableProfileTableBuilder::new().unwrap();
    vpt_builder.add_output::<f32>("output").unwrap();
}

#[test]
#[should_panic(expected="NulError")]
fn add_output_invalid_name() {
    let mut vpt_builder = VariableProfileTableBuilder::new().unwrap();
    vpt_builder.add_output::<f32>("out\0put").unwrap();
}
