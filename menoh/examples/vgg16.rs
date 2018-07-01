extern crate docopt;
extern crate image;
extern crate menoh;
#[macro_use]
extern crate serde_derive;

use image::GenericImage;
use std::cmp;
use std::error;
use std::fs;
use std::io;
use std::io::BufRead;
use std::path;

fn crop_and_resize(mut img: image::DynamicImage, size: usize) -> image::DynamicImage {
    let (h, w) = (img.height(), img.width());
    let min = cmp::min(w, h);
    img.crop((w - min) / 2, (h - min) / 2, min, min)
        .resize_exact(size as _, size as _, image::FilterType::Nearest)
}

fn reorder_to_chw(img: &image::DynamicImage) -> Vec<f32> {
    let mut data = Vec::new();
    // rev: RGB -> BGR
    for c in (0..3).rev() {
        for y in 0..img.height() {
            for x in 0..img.width() {
                data.push(img.get_pixel(x, y).data[c] as f32);
            }
        }
    }
    data
}

fn load_category_list<P>(path: P) -> io::Result<Vec<String>>
    where P: AsRef<path::Path>
{
    let mut categories = Vec::new();
    for line in io::BufReader::new(fs::File::open(path)?).lines() {
        categories.push(line?);
    }
    Ok(categories)
}

const USAGE: &'static str = r#"
vgg16 example

Usage: vgg16 [options]

Options:
  -i --image PATH         input image path [default: Light_sussex_hen.jpg]
  -m --model PATH         onnx model path [default: VGG16.onnx]
  -s --synset-words PATH  synset words path [default: synset_words.txt]
"#;

#[derive(Debug, Deserialize)]
struct Args {
    flag_i: path::PathBuf,
    flag_m: path::PathBuf,
    flag_s: path::PathBuf,
}

fn main() -> Result<(), Box<dyn(error::Error)>> {
    let args: Args = docopt::Docopt::new(USAGE)
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());

    const INSIZE: usize = 224;
    const CONV1_1_IN_NAME: &'static str = "140326425860192";
    const FC6_OUT_NAME: &'static str = "140326200777584";
    const SOFTMAX_OUT_NAME: &'static str = "140326200803680";

    let mut vpt_builder = menoh::VariableProfileTableBuilder::new()?;
    vpt_builder
        .add_input::<f32>(CONV1_1_IN_NAME, &[1, 3, INSIZE, INSIZE])?;
    vpt_builder.add_output::<f32>(FC6_OUT_NAME)?;
    vpt_builder.add_output::<f32>(SOFTMAX_OUT_NAME)?;

    let mut model_data = menoh::ModelData::from_onnx(args.flag_m)?;
    let vpt = vpt_builder.build(&model_data)?;
    model_data.optimize(&vpt)?;

    let model_builder = menoh::ModelBuilder::new(&vpt)?;
    let mut model = model_builder.build(&model_data, "mkldnn", "")?;
    let img = image::open(args.flag_i)?;
    model
        .get_variable_mut(CONV1_1_IN_NAME)?
        .1
        .copy_from_slice(&reorder_to_chw(&crop_and_resize(img, INSIZE)));
    model.run()?;

    let (_, fc6_buf) = model.get_variable::<f32>(FC6_OUT_NAME)?;
    println!("{:?}", &fc6_buf[..10]);

    let (softmax_dims, softmax_buf) = model.get_variable::<f32>(SOFTMAX_OUT_NAME)?;
    let mut indices: Vec<_> = (0..softmax_dims[1]).collect();
    indices.sort_unstable_by(|&i, &j| {
                                 softmax_buf[j]
                                     .partial_cmp(&softmax_buf[i])
                                     .unwrap_or(cmp::Ordering::Equal)
                             });
    let categories = load_category_list(args.flag_s)?;
    for &i in &indices[..5] {
        println!("{} {} {}", i, softmax_buf[i], categories[i]);
    }

    Ok(())
}
