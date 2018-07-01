extern crate menoh;

fn main() -> Result<(), menoh::Error> {
    let conv1_1_in_name = "140326425860192";
    let fc6_out_name = "140326200777584";
    let softmax_out_name = "140326200803680";

    let mut model_data = menoh::ModelData::from_onnx("VGG16.onnx")?;

    let mut vpt_builder = menoh::VariableProfileTableBuilder::new()?;
    vpt_builder
        .add_input::<f32>(conv1_1_in_name, &[1, 3, 224, 224])?;
    vpt_builder.add_output::<f32>(fc6_out_name)?;
    vpt_builder.add_output::<f32>(softmax_out_name)?;

    let vpt = vpt_builder.build(&model_data)?;
    model_data.optimize(&vpt)?;

    let mut model_builder = menoh::ModelBuilder::new(&vpt)?;
    let mut input_data = [0.5_f32; 1 * 3 * 224 * 224];
    unsafe {
        model_builder
            .attach_external_buffer(conv1_1_in_name, input_data.as_mut_ptr() as _)?;
    }

    let mut model = model_builder.build(&model_data, "mkldnn", "")?;
    model.run()?;

    let (_, fc6_data) = model.get_variable::<f32>(fc6_out_name)?;
    println!("{:?}", &fc6_data[..10]);

    let (softmax_dims, softmax_data) = model.get_variable::<f32>(softmax_out_name)?;
    for n in 0..softmax_dims[0] {
        println!("{:?}",
                 &softmax_data[n * softmax_dims[1]..(n + 1) * softmax_dims[1]]);
    }

    Ok(())
}
