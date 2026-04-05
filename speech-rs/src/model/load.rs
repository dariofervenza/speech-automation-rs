use crate::model::canary::CanaryModel;
use crate::config::types::GlobalConfig;

// use std::env::var;
use ort::session::{ Session };
use ort::execution_providers::{ CUDAExecutionProvider, ExecutionProvider };
// use ort::execution_providers::cuda::preload_dylibs;
use ort::{ Result, Error };
use log::{ debug, info };
// use std::path::Path;


fn load_model_part(model_path: String) -> Result<Session, Error> {
    let model = Session::builder()?
        .with_execution_providers(
            [CUDAExecutionProvider::default().with_device_id(0).build().error_on_failure()]
        )?
        .commit_from_file(model_path);
    model
}


pub fn load_model(app_cfg: &GlobalConfig) -> Result<CanaryModel, Error> {
    //let cuda = Path::new("/usr/local/cuda/targets/x86_64-linux/lib/");
    //let cdnn = Path::new("/usr/lib/x86_64-linux-gnu");
    //let res = preload_dylibs(
    //    Some(cuda),
    //    Some(cdnn),
    //).expect("error preloading cuda libs");
    //info!("Load libs res: {:?}", res);
    let init = ort::init()
        .commit()?;
    let exec_prov = CUDAExecutionProvider::default().with_device_id(0);
    info!("IS CUDA AVAILABLE: {:?}", exec_prov.is_available());
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
