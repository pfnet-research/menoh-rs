//! A Rust binding for [Menoh](https://github.com/pfnet-research/menoh)
//!
//! ```
//! # fn main() -> Result<(), menoh::Error> {
//! let in_name = "139830916504208";
//! let out_name = "139830916504880";
//!
//! let mut model = menoh::Builder::from_onnx("test.onnx")?
//!                     .add_input::<f32>(in_name, &[2, 3])?
//!                     .add_output::<f32>(out_name)?
//!                     .build("mkldnn", "")?;
//!
//! // This block limits the lifetime of `in_buf`.
//! {
//!     let (in_dims, in_buf) = model.get_variable_mut::<f32>(in_name)?;
//!     in_buf.copy_from_slice(&[0., 1., 2., 3., 4., 5.]);
//!     println!("in:");
//!     # assert_eq!(in_dims, &[2, 3]);
//!     println!("    dims: {:?}", in_dims);
//!     println!("    buf: {:?}", in_buf);
//! }
//!
//! model.run()?;
//!
//! let (out_dims, out_buf) = model.get_variable::<f32>(out_name)?;
//! println!("out:");
//! # assert_eq!(out_dims, &[2, 5]);
//! println!("    dims: {:?}", out_dims);
//! println!("    buf: {:?}", out_buf);
//! # Ok(())
//! # }
//! ```

extern crate menoh_sys;

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
