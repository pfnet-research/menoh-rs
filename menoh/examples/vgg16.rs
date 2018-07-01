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
    let mut input_buf = [0.5_f32; 1 * 3 * 224 * 224];
    unsafe {
        model_builder
            .attach_external(conv1_1_in_name, &mut input_buf)?;
    }

    let mut model = model_builder.build(&model_data, "mkldnn", "")?;
    model.run()?;

    let (_, fc6_buf) = model.get_variable::<f32>(fc6_out_name)?;
    println!("{:?}", &fc6_buf[..10]);

    let (softmax_dims, softmax_buf) = model.get_variable::<f32>(softmax_out_name)?;
    for n in 0..softmax_dims[0] {
        println!("{:?}",
                 &softmax_buf[n * softmax_dims[1]..(n + 1) * softmax_dims[1]]);
    }

    Ok(())
}
