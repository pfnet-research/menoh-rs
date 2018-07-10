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
    /// # use menoh::Builder;
    /// # fn main() -> Result<(), menoh::Error> {
    /// let builder = Builder::from_onnx("test.onnx")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_onnx<P>(path: P) -> Result<Self, Error>
        where P: AsRef<path::Path>
    {
        Ok(Self {
               model_data: ModelData::from_onnx(path)?,
               vpt_builder: VariableProfileTableBuilder::new()?,
           })
    }

    /// Register a variable as input.
    ///
    /// ```
    /// # use menoh::Builder;
    /// # fn main() -> Result<(), menoh::Error> {
    /// # let builder = Builder::from_onnx("test.onnx")?;
    /// let builder = builder.add_input::<f32>("139830916504208", &[2, 3])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_input<T>(mut self, name: &str, dims: &[usize]) -> Result<Self, Error>
        where T: Dtype
    {
        self.vpt_builder.add_input::<T>(name, dims)?;
        Ok(self)
    }

    /// Register a variable as output.
    ///
    /// ```
    /// # use menoh::Builder;
    /// # fn main() -> Result<(), menoh::Error> {
    /// # let builder = Builder::from_onnx("test.onnx")?;
    /// let builder = builder.add_output::<f32>("139830916504880")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_output<T>(mut self, name: &str) -> Result<Self, Error>
        where T: Dtype
    {
        self.vpt_builder.add_output::<T>(name)?;
        Ok(self)
    }

    /// Build a `Model`.
    ///
    /// ```
    /// # use menoh::Builder;
    /// # fn main() -> Result<(), menoh::Error> {
    /// # let builder = Builder::from_onnx("test.onnx")?
    /// #                   .add_input::<f32>("139830916504208", &[2, 3])?
    /// #                   .add_output::<f32>("139830916504880")?;
    /// let model = builder.build("mkldnn", "")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(mut self, backend: &str, backend_config: &str) -> Result<Model, Error> {
        let vpt = self.vpt_builder.build(&self.model_data)?;
        self.model_data.optimize(&vpt)?;
        let model_builder = ModelBuilder::new(&vpt)?;
        Ok(model_builder
               .build(self.model_data, backend, backend_config)?)
    }
}
