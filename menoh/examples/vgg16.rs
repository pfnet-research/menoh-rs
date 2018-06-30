extern crate menoh;

use std::slice;

fn main() -> Result<(), menoh::Error> {
    let conv1_1_in_name = "140326425860192";
    let fc6_out_name = "140326200777584";
    let softmax_out_name = "140326200803680";

    let mut model_data = menoh::ModelData::from_onnx("VGG16.onnx")?;

    let mut vpt_builder = menoh::VariableProfileTableBuilder::new()?;
    vpt_builder
        .add_input(conv1_1_in_name, menoh::Dtype::Float, &[1, 3, 224, 224])?;
    vpt_builder
        .add_output(fc6_out_name, menoh::Dtype::Float)?;
    vpt_builder
        .add_output(softmax_out_name, menoh::Dtype::Float)?;

    let variable_profile_table = vpt_builder.build(&model_data)?;
    let softmax_out = variable_profile_table.get(softmax_out_name)?;
    model_data.optimize(&variable_profile_table)?;

    let mut model_builder = menoh::ModelBuilder::new(&variable_profile_table)?;
    let mut input_buff = [0.5_f32; 1 * 3 * 224 * 224];
    unsafe {
        model_builder
            .attach_external_buffer(conv1_1_in_name, input_buff.as_mut_ptr() as _)?;
    }

    let mut model = model_builder.build(&model_data, "mkldnn", "")?;
    let fc6_output_buff = model.get_variable_buffer_handle(fc6_out_name)?;
    let softmax_output_buff = model.get_variable_buffer_handle(softmax_out_name)?;

    model.run()?;

    println!("{:?}",
             unsafe { slice::from_raw_parts(fc6_output_buff as *const f32, 10) });

    let softmax_output_buff = unsafe {
        slice::from_raw_parts(softmax_output_buff as *const f32,
                              softmax_out.dims[0] * softmax_out.dims[1])
    };
    for n in 0..softmax_out.dims[0] {
        println!("{:?}",
                 &softmax_output_buff[n * softmax_out.dims[1]..(n + 1) * softmax_out.dims[1]]);
    }

    Ok(())
}
