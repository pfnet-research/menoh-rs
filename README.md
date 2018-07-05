# menoh-rs

[![Build Status](https://travis-ci.org/Hakuyume/menoh-rs.svg?branch=master)](https://travis-ci.org/Hakuyume/menoh-rs)  
[Documentation](https://hakuyume.github.io/menoh-rs/menoh/)

A Rust binding for [Menoh](https://github.com/pfnet-research/menoh)

## Requirements
- Rust 1.27
- Cargo
- pkg-config (for [pkg-config](https://crates.io/crates/pkg-config))
- libclang (for [bindgen](https://crates.io/crates/bindgen))
- [Menoh](https://github.com/pfnet-research/menoh) 1.0+

## Demo

```
$ git clone https://github.com/Hakuyume/menoh-rs.git
$ cd menoh-rs/menoh

$ curl -L https://www.dropbox.com/s/bjfn9kehukpbmcm/VGG16.onnx?dl=1 -o VGG16.onnx
$ curl -LO https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt
$ curl -LO https://upload.wikimedia.org/wikipedia/commons/5/54/Light_sussex_hen.jpg

$ cargo run --example vgg16  # use Light_sussex_hen.jpg
$ cargo run --example vgg16 -- --image <image>  # use your image
```
