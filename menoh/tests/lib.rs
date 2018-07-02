extern crate menoh;

const IN_NAME: &'static str = "139830916504208";
const FC1_NAME: &'static str = "139830916504600";
const FC2_NAME: &'static str = "139830916504880";

fn assert_almost_eq(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for i in 0..expected.len() {
        assert!((actual[i] - expected[i]).abs() < 1e-6);
    }
}

#[test]
fn inference() {
    let mut model = (|| {
                         menoh::Builder::new("test.onnx")?
                             .add_input::<f32>(IN_NAME, &[2, 3])?
                             .add_output::<f32>(FC1_NAME)?
                             .add_output::<f32>(FC2_NAME)?
                             .build("mkldnn", "")
                     })()
            .unwrap();
    {
        let (in_dims, in_buf) = model.get_variable_mut::<f32>(IN_NAME).unwrap();
        assert_eq!(in_dims, &[2, 3]);
        assert_eq!(in_buf.len(), 2 * 3);
        in_buf.copy_from_slice(&[0.3595079, 0.43703195, 0.6976312, 0.06022547, 0.6667667,
                                 0.67063785]);
    }

    model.run().unwrap();

    {
        let (fc1_dims, fc1_buf) = model.get_variable::<f32>(FC1_NAME).unwrap();
        assert_eq!(fc1_dims, &[2, 4]);
        assert_almost_eq(fc1_buf,
                         &[0.86133176, 0.54272187, 0.11743745, 0.7073185, 0.5943423, 0.41845477,
                           0., 0.63281214]);
    }
    {
        let (fc2_dims, fc2_buf) = model.get_variable::<f32>(FC2_NAME).unwrap();
        assert_eq!(fc2_dims, &[2, 5]);
        assert_almost_eq(fc2_buf,
                         &[0.5048409, 0.30410108, 0., 0.5193354, 1.2016813, 0.35719275,
                           0.13083139, 0., 0.31098637, 0.88258076]);
    }
}
