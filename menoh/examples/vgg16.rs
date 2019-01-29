use docopt::Docopt;
use image::{DynamicImage, FilterType, GenericImageView};
use serde_derive::Deserialize;
use std::cmp::Ordering;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};

const USAGE: &'static str = r#"
VGG16 example

Usage: vgg16 [options]

Options:
  -i --image PATH         input image path [default: Light_sussex_hen.jpg]
  -m --model PATH         onnx model path [default: VGG16.onnx]
  -s --synset-words PATH  synset words path [default: synset_words.txt]
"#;

#[derive(Debug, Deserialize)]
struct Args {
    flag_i: PathBuf,
    flag_m: PathBuf,
    flag_s: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Args = Docopt::new(USAGE)
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());

    const INSIZE: usize = 224;
    const CONV1_1_IN_NAME: &'static str = "140326425860192";
    const FC6_OUT_NAME: &'static str = "140326200777584";
    const SOFTMAX_OUT_NAME: &'static str = "140326200803680";

    let mut model = menoh::Builder::from_onnx(args.flag_m)?
        .add_input::<f32>(CONV1_1_IN_NAME, &[1, 3, INSIZE, INSIZE])?
        .add_output(FC6_OUT_NAME)?
        .add_output(SOFTMAX_OUT_NAME)?
        .build("mkldnn", "")?;

    let img = image::open(args.flag_i)?;
    let (_, conv1_1_buf) = model.get_variable_mut::<f32>(CONV1_1_IN_NAME)?;
    set_image(conv1_1_buf, &img, INSIZE);

    model.run()?;

    let (_, fc6_buf) = model.get_variable::<f32>(FC6_OUT_NAME)?;
    println!("{:?}", &fc6_buf[..10]);

    let (softmax_dims, softmax_buf) = model.get_variable::<f32>(SOFTMAX_OUT_NAME)?;
    let mut indices: Vec<_> = (0..softmax_dims[1]).collect();
    indices.sort_unstable_by(|&i, &j| {
        softmax_buf[j]
            .partial_cmp(&softmax_buf[i])
            .unwrap_or(Ordering::Equal)
    });
    let categories = load_category_list(args.flag_s)?;
    for &i in &indices[..5] {
        println!("{} {} {}", i, softmax_buf[i], categories[i]);
    }

    Ok(())
}

fn set_image(buf: &mut [f32], img: &DynamicImage, size: usize) {
    assert!(buf.len() <= 3 * size * size);
    let img = img.resize_exact(size as _, size as _, FilterType::Nearest);

    const MEAN: [f32; 3] = [103.939, 116.779, 123.68];
    for c in 0..3 {
        for y in 0..size {
            for x in 0..size {
                // 3 - (c + 1): RGB -> BGR
                buf[(c * size + y) * size + x] =
                    img.get_pixel(x as _, y as _).data[3 - (c + 1)] as f32 - MEAN[c];
            }
        }
    }
}

fn load_category_list<P>(path: P) -> io::Result<Vec<String>>
where
    P: AsRef<Path>,
{
    let mut categories = Vec::new();
    for line in BufReader::new(File::open(path)?).lines() {
        categories.push(line?);
    }
    Ok(categories)
}
