//! Rust binding for [Menoh](https://github.com/pfnet-research/menoh)
//!
//! ## Example
//!
//! ```
//! fn main() -> Result<(), menoh::Error> {
//!     let mut model = menoh::Builder::from_onnx("MLP.onnx")?
//!         .add_input::<f32>("input", &[2, 3])?
//!         .add_output("fc2")?
//!         .build("mkldnn", "")?;
//!
//!     {
//!         let (in_dims, in_buf) = model.get_variable_mut::<f32>("input")?;
//!         in_buf.copy_from_slice(&[0., 1., 2., 3., 4., 5.]);
//!         println!("in:");
//!         # assert_eq!(in_dims, &[2, 3]);
//!         println!("    dims: {:?}", in_dims);
//!         println!("    buf: {:?}", in_buf);
//!     }
//!
//!     model.run()?;
//!
//!     let (out_dims, out_buf) = model.get_variable::<f32>("fc2")?;
//!     println!("out:");
//!     # assert_eq!(out_dims, &[2, 5]);
//!     println!("    dims: {:?}", out_dims);
//!     println!("    buf: {:?}", out_buf);
//!     # let expected = &[0., 0., 15., 96., 177., 0., 0., 51., 312., 573.];
//!     # for i in 0..expected.len() {
//!     #     assert!((out_buf[i] - expected[i]).abs() < 1e-6);
//!     # }
//!     Ok(())
//! }
//! ```
//!
//! ## Usage
//!
//! ### 1. Build a `Model`.
//!
//! ```
//! # fn main() -> Result<(), menoh::Error> {
//! let mut model = menoh::Builder::from_onnx("MLP.onnx")?
//! // register `"input"` as input
//! // and specify its type (`f32`) and shape (`&[2, 3]`).
//!     .add_input::<f32>("input", &[2, 3])?
//! // register `"fc2"` as output.
//!     .add_output("fc2")?
//! // specify backend (`"mkldnn"`) and its configuration (`""`).
//!     .build("mkldnn", "")?;
//! # Ok(())
//! # }
//! ```
//! Instead of `Builder`, we can use a combination of some low-level APIs.
//! ```
//! # fn main() -> Result<(), menoh::Error> {
//! let mut model_data = menoh::ModelData::from_onnx("MLP.onnx")?;
//!
//! let mut vpt_builder = menoh::VariableProfileTableBuilder::new()?;
//! vpt_builder.add_input::<f32>("input", &[2, 3])?;
//! vpt_builder.add_output("fc2")?;
//! let vpt = vpt_builder.build(&model_data)?;
//!
//! model_data.optimize(&vpt)?;
//! let model_builder = menoh::ModelBuilder::new(&vpt)?;
//! let mut model = model_builder.build(model_data, "mkldnn", "")?;
//! # Ok(())
//! # }
//! ```
//!
//! ### 2. Set data to input variable(s).
//!
//! ```
//! # fn main() -> Result<(), menoh::Error> {
//! # let mut model = menoh::Builder::from_onnx("MLP.onnx")?
//! #     .add_input::<f32>("input", &[2, 3])?
//! #     .add_output("fc2")?
//! #     .build("mkldnn", "")?;
//! // fetch a read/write view of a variable.
//! let (in_dims, in_buf) = model.get_variable_mut::<f32>("input")?;
//! // set data to the variable.
//! in_buf.copy_from_slice(&[0., 1., 2., 3., 4., 5.]);
//! # Ok(())
//! # }
//! ```
//! Note: The lifetime of views has to end before executing `Model::run`.
//! Blocks will be required to limit the lifetime.
//! ```compile_fail
//! // NG: `in_buf` lives after `model.run()`.
//! # fn main() -> Result<(), menoh::Error> {
//! # let mut model = menoh::Builder::from_onnx("MLP.onnx")?
//! #     .add_input::<f32>("input", &[2, 3])?
//! #     .add_output("fc2")?
//! #     .build("mkldnn", "")?;
//! let (in_dims, in_buf) = model.get_variable_mut::<f32>("input")?;
//! in_buf.copy_from_slice(&[0., 1., 2., 3., 4., 5.]);
//! model.run()?;
//! # Ok(())
//! # }
//! ```
//! ```
//! // OK: the lifetime of `in_buf` is limited by a block.
//! # fn main() -> Result<(), menoh::Error> {
//! # let mut model = menoh::Builder::from_onnx("MLP.onnx")?
//! #     .add_input::<f32>("input", &[2, 3])?
//! #     .add_output("fc2")?
//! #     .build("mkldnn", "")?;
//! {
//!     let (in_dims, in_buf) = model.get_variable_mut::<f32>("input")?;
//!     in_buf.copy_from_slice(&[0., 1., 2., 3., 4., 5.]);
//! }
//! model.run()?;
//! # Ok(())
//! # }
//! ```
//!
//! ### 3. Execute computation.
//!
//! ```
//! # fn main() -> Result<(), menoh::Error> {
//! # let mut model = menoh::Builder::from_onnx("MLP.onnx")?
//! #     .add_input::<f32>("input", &[2, 3])?
//! #     .add_output("fc2")?
//! #     .build("mkldnn", "")?;
//! model.run()?;
//! # Ok(())
//! # }
//! ```
//!
//! ### 4. Fetch the result(s).
//!
//! ```
//! # fn main() -> Result<(), menoh::Error> {
//! # let mut model = menoh::Builder::from_onnx("MLP.onnx")?
//! #     .add_input::<f32>("input", &[2, 3])?
//! #     .add_output("fc2")?
//! #     .build("mkldnn", "")?;
//! // fetch a read-only view of a variable.
//! let (out_dims, out_buf) = model.get_variable::<f32>("fc2")?;
//! // use the data (e.g. print them).
//! println!("{:?}", out_buf);
//! # Ok(())
//! # }
//! ```

mod builder;
mod dtype;
mod error;
mod handler;
mod model;
mod model_builder;
mod model_data;
mod variable_profile_table;
mod variable_profile_table_builder;

pub use builder::Builder;
pub use dtype::Dtype;
pub use error::Error;
pub use model::Model;
pub use model_builder::ModelBuilder;
pub use model_data::ModelData;
pub use variable_profile_table::VariableProfileTable;
pub use variable_profile_table_builder::VariableProfileTableBuilder;
