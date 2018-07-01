# menoh-rs

A rust wrapper for [Menoh](https://github.com/pfnet-research/menoh)

## Requirements
- Rust 1.26+
- Cargo
- pkg-config (for [pkg-config](https://crates.io/crates/pkg-config))
- libclang (for [bindgen](https://crates.io/crates/bindgen))
- [Menoh](https://github.com/pfnet-research/menoh) 1.0+

## Demo

```
$ git clone https://github.com/Hakuyume/menoh-rs.git
$ cd menoh-rs/menoh
$ curl -L https://www.dropbox.com/s/bjfn9kehukpbmcm/VGG16.onnx?dl=1 -o VGG16.onnx
$ cargo run --example vgg16
```
