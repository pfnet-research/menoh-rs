# menoh-rs

[![crates.io](https://img.shields.io/crates/v/menoh.svg)](https://crates.io/crates/menoh)
[![docs.rs](https://docs.rs/menoh/badge.svg)](https://docs.rs/menoh)
[![Travis CI](https://travis-ci.org/Hakuyume/menoh-rs.svg?branch=master)](https://travis-ci.org/pfnet-research/menoh-rs)
[![AppVeyor](https://ci.appveyor.com/api/projects/status/3btlg4uqa5pm6mfb/branch/master?svg=true)](https://ci.appveyor.com/project/Hakuyume/menoh-rs/branch/master)

Rust binding for [Menoh](https://github.com/pfnet-research/menoh)  
[Documentation](https://docs.rs/menoh)

## Requirements
- Rust 1.27+
- [Menoh](https://github.com/pfnet-research/menoh) 1.0+
  (please make sure that `pkg-config` can find `menoh`)

## Demo

```
$ git clone https://github.com/pfnet-research/menoh-rs.git
$ cd menoh-rs/menoh

$ curl -L https://www.dropbox.com/s/bjfn9kehukpbmcm/VGG16.onnx?dl=1 -o VGG16.onnx
$ curl -LO https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt
$ curl -LO https://upload.wikimedia.org/wikipedia/commons/5/54/Light_sussex_hen.jpg

$ cargo run --example vgg16  # use Light_sussex_hen.jpg
$ cargo run --example vgg16 -- --image <image>  # use your image
```

## Example

```rust
extern crate menoh;

fn main() -> Result<(), menoh::Error> {
    let mut model = menoh::Builder::from_onnx("MLP.onnx")?
        .add_input::<f32>("input", &[2, 3])?
        .add_output::<f32>("fc2")?
        .build("mkldnn", "")?;

    {
        let (in_dims, in_buf) = model.get_variable_mut::<f32>("input")?;
        in_buf.copy_from_slice(&[0., 1., 2., 3., 4., 5.]);
        println!("in:");
        println!("    dims: {:?}", in_dims);
        println!("    buf: {:?}", in_buf);
    }

    model.run()?;

    let (out_dims, out_buf) = model.get_variable::<f32>("fc2")?;
    println!("out:");
    println!("    dims: {:?}", out_dims);
    println!("    buf: {:?}", out_buf);
    Ok(())
}
```
