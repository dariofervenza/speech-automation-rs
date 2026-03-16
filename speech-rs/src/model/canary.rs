use ort::session::Session;
use ort::{ Result, Error };


pub struct CanaryModel {
    encoder: Session,
    decoder: Session,
}


impl CanaryModel {
    pub fn from_models(encoder: Result<Session, Error>, decoder: Result<Session, Error>) -> Result<CanaryModel, Error> {
        let model = CanaryModel {
            encoder: encoder?,
            decoder: decoder?
        };
        Ok(model)
    }
}

