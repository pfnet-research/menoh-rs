use std::path;

use Dtype;
use Error;
use Model;
use ModelBuilder;
use ModelData;
use VariableProfileTableBuilder;

/// Helper to build `Model`.
pub struct Builder {
    model_data: ModelData,
    vpt_builder: VariableProfileTableBuilder,
}

impl Builder {
    /// Create a builder from a ONNX file.
    ///
    /// ```
    /// # use menoh::*;
    /// # fn main() -> Result<(), Error> {
    /// let builder = Builder::from_onnx("MLP.onnx")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_onnx<P>(path: P) -> Result<Self, Error>
    where
        P: AsRef<path::Path>,
    {
        Ok(Self {
            model_data: ModelData::from_onnx(path)?,
            vpt_builder: VariableProfileTableBuilder::new()?,
        })
    }

    /// Create a builder from a ONNX data.
    ///
    /// ```
    /// # use menoh::*;
    /// # fn main() -> Result<(), Error> {
    /// # let onnx_data = include_bytes!("../../MLP.onnx");
    /// let builder = Builder::from_onnx_bytes(onnx_data)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_onnx_bytes(data: &[u8]) -> Result<Self, Error> {
        Ok(Self {
            model_data: ModelData::from_onnx_bytes(data)?,
            vpt_builder: VariableProfileTableBuilder::new()?,
        })
    }

    /// Register a variable as input.
    ///
    /// ```
    /// # use menoh::*;
    /// # fn main() -> Result<(), Error> {
    /// # let builder = Builder::from_onnx("MLP.onnx")?;
    /// let builder = builder.add_input::<f32>("input", &[2, 3])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_input<T>(mut self, name: &str, dims: &[usize]) -> Result<Self, Error>
    where
        T: Dtype,
    {
        self.vpt_builder.add_input::<T>(name, dims)?;
        Ok(self)
    }

    /// Register a variable as output.
    ///
    /// ```
    /// # use menoh::*;
    /// # fn main() -> Result<(), Error> {
    /// # let builder = Builder::from_onnx("MLP.onnx")?;
    /// let builder = builder.add_output("fc2")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_output(mut self, name: &str) -> Result<Self, Error> {
        self.vpt_builder.add_output(name)?;
        Ok(self)
    }

    /// Build a `Model`.
    ///
    /// ```
    /// # use menoh::*;
    /// # fn main() -> Result<(), Error> {
    /// # let builder = Builder::from_onnx("MLP.onnx")?
    /// #     .add_input::<f32>("input", &[2, 3])?
    /// #     .add_output("fc2")?;
    /// let model = builder.build("mkldnn", "")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(mut self, backend: &str, backend_config: &str) -> Result<Model, Error> {
        let vpt = self.vpt_builder.build(&self.model_data)?;
        self.model_data.optimize(&vpt)?;
        let model_builder = ModelBuilder::new(&vpt)?;
        Ok(model_builder.build(self.model_data, backend, backend_config)?)
    }
}
