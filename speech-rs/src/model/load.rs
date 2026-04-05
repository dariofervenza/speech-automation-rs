use crate::model::canary::CanaryModel;
use crate::config::types::GlobalConfig;

// use std::env::var;
use ort::session::{ builder::GraphOptimizationLevel, Session };
use ort::execution_providers::CUDAExecutionProvider;
use ort::{ Result, Error };
use log::debug;


fn load_model_part(model_path: String) -> Result<Session, Error> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .with_intra_threads(4)?
        .commit_from_file(model_path);
    model
}


pub fn load_model(app_cfg: &GlobalConfig) -> Result<CanaryModel, Error> {
    let init = ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;
    debug!("Cuda execution {}", init);
    // let encoder_model_path = var("ONNX_ENCODER_PATH").expect("Error loading encoder path");
    // let decoder_model_path = var("ONNX_DECODER_PATH").expect("Error loading decoder path");
    let model_paths = &app_cfg.models.canary.model_path;
    let encoder_model_path = &model_paths.encoder;
    let decoder_model_path = &model_paths.decoder;
    debug!("Encoder path {}", encoder_model_path);
    debug!("Decoder path {}", decoder_model_path);
    let encoder = load_model_part(encoder_model_path.clone());
    let decoder = load_model_part(decoder_model_path.clone());
    CanaryModel::from_models(encoder, decoder)
}
