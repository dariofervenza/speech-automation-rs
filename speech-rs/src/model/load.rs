use crate::model::canary::CanaryModel;

use std::env::var;
use ort::session::{ builder::GraphOptimizationLevel, Session };
use ort::execution_providers::CUDAExecutionProvider;
use ort::{ Result, Error };


fn load_model_part(model_path: String) -> Result<Session, Error> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .with_intra_threads(4)?
        .commit_from_file(model_path);
    model
}


pub fn load_model(debug: bool) -> Result<CanaryModel, Error> {
    let init = ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;
    let encoder_model_path = var("ONNX_ENCODER_PATH").expect("Error loading encoder path");
    let decoder_model_path = var("ONNX_DECODER_PATH").expect("Error loading decoder path");
    if debug {
        println!("Cuda execution {}", init);
        println!("Encoder path {}", encoder_model_path);
        println!("Decoder path {}", decoder_model_path);
    }
    let encoder = load_model_part(encoder_model_path);
    let decoder = load_model_part(decoder_model_path);
    CanaryModel::from_models(encoder, decoder)
}
