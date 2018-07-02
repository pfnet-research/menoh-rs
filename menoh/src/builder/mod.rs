use std::path;

use dtype::Dtype;
use Error;
use Model;
use ModelBuilder;
use ModelData;
use VariableProfileTableBuilder;

pub struct Builder {
    model_data: ModelData,
    vpt_builder: VariableProfileTableBuilder,
}

impl Builder {
    pub fn new<P>(path: P) -> Result<Self, Error>
        where P: AsRef<path::Path>
    {
        Ok(Self {
               model_data: ModelData::from_onnx(path)?,
               vpt_builder: VariableProfileTableBuilder::new()?,
           })
    }

    pub fn add_input<T>(mut self, name: &str, dims: &[usize]) -> Result<Self, Error>
        where T: Dtype
    {
        self.vpt_builder.add_input::<T>(name, dims)?;
        Ok(self)
    }

    pub fn add_output<T>(mut self, name: &str) -> Result<Self, Error>
        where T: Dtype
    {
        self.vpt_builder.add_output::<T>(name)?;
        Ok(self)
    }

    pub fn build(mut self, backend: &str, backend_config: &str) -> Result<Model, Error> {
        let vpt = self.vpt_builder.build(&self.model_data)?;
        self.model_data.optimize(&vpt)?;
        let model_builder = ModelBuilder::new(&vpt)?;
        Ok(model_builder
               .build(&self.model_data, backend, backend_config)?)
    }
}
