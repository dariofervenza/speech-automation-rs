use ort::session::Session;
use ort::{ Result, Error, inputs };
use ort::value::TensorRef;
use ndarray::{ Axis, Array2, Array, Dim, s };


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

    pub fn encode(&mut self, mono_audio: Vec<f32>) {
        let tensor = TensorRef::from_array_view(
            ([1, 1, mono_audio.len()], &*mono_audio)
        ).expect("error creatign tensor");
        println!("ENCODER INPUTS: {:?}", self.encoder.inputs());
        println!("ENCODER OUTPUTS: {:?}", self.encoder.outputs());
        let outputs = self.encoder.run(inputs![tensor]).expect("error in encoder model");
        println!("ENCODER OUTPUTS HAS LEN {:?}", outputs.len());
        let keys: Vec<&str> = outputs.keys().collect();
        println!("ENCODER OUTPUTS HAS keys {:?}", keys);
        // println!("ENCODER OUTPUTS HAS values {:?}", outputs.values());
    }
}

