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

        // https://www.mdpi.com/2076-3417/14/24/11583

        // https://docs.rs/knf-rs/0.3.2/knf_rs/fn.compute_fbank.html

        // https://www.kaggle.com/code/ahmedabdoamin/preprocessing-speech-mfcc-vs-filter-banks

        // https://apxml.com/courses?page=3&level=3

        // https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/canary/canary.py

        // https://github.com/NVIDIA-NeMo/NeMo/tree/main/nemo/collections/asr/parts


        // https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/audio/data/audio_to_audio.py#L592
    }
}

