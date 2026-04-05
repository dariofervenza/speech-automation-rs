use crate::model::fbank::FbankProcessor;
use crate::model::encoder::EncoderOutput;
use crate::model::decoder::{ DecoderInput, DecoderOutput };
use crate::model::tensor::TensorProcessor;
use crate::config::types::{ Canary, AudioFile };
use ort::session::{ Session, SessionOutputs };
use ort::{ Result, Error, inputs };
use ort::value::{ TensorRef, TensorValueType, Value };
use ort::tensor::PrimitiveTensorElementType;
use ndarray::{ Axis, ArrayD, s };
use log::{ info, debug };
use crate::model::tokenizer::Tokenizer;
use std::path::Path;


fn init_tokens(model_cfg: &Canary, audio: &AudioFile, tokenizer: &Tokenizer) -> ArrayD<i64> {
    info!("Audio file: {}", audio.original_file);
    let mut vec_init: Vec<i64> = model_cfg.init_prompt.clone();
    let from_lang = tokenizer.tokenize(&audio.source_lang);
    let to_lang = tokenizer.tokenize(&audio.target_lang);
    vec_init[4] = from_lang as i64;
    vec_init[5] = to_lang as i64;
    let array_init = ArrayD::from_shape_vec(
        vec![1, vec_init.len()], vec_init.to_vec()
    ).expect("Error creating array init");
    array_init
}


fn init_decoder_mems(model_cfg: &Canary) -> Value<TensorValueType<f32>> {
    let init_decoder_mems = &model_cfg.init_decoder_mems;
    let layers = init_decoder_mems.layers;
    let batch = init_decoder_mems.batch;
    let mems_len = init_decoder_mems.mems_len; // The first pass has NO history
    let hidden_size = init_decoder_mems.hidden_size;
    let mems_shape = vec![layers, batch, mems_len, hidden_size];
    Value::from_array(
        ArrayD::<f32>::zeros(mems_shape)
    ).expect("Failed to create ORT Value")
}


pub struct CanaryModel {
    encoder: Session,
    decoder: Session,
}

impl FbankProcessor for CanaryModel {}
impl TensorProcessor for CanaryModel {}


impl CanaryModel {
    pub fn from_models(encoder: Result<Session, Error>, decoder: Result<Session, Error>) -> Result<CanaryModel, Error> {
        let model = CanaryModel {
            encoder: encoder?,
            decoder: decoder?
        };
        
        Ok(model)
    }

    fn print_out_shape<T>(out: &SessionOutputs, key_name: &str)
    where 
        T: PrimitiveTensorElementType
    {
        let tensor = Self::infered_tensor::<T>(out, key_name);
        info!("{} shape is: {:?}", key_name, &tensor.0);
    }

    pub fn pipe(&mut self, input_vec: Vec<f32>, model_cfg: &Canary, audio: &AudioFile) {
        for i in 0..15 {
            info!("Pass no: {}", i);
            let encoder_out = self.encode(input_vec.clone(), model_cfg);
            self.decode(encoder_out, model_cfg, audio);
        }
    }

    fn encode(&mut self, mono_audio: Vec<f32>, model_cfg: &Canary) -> EncoderOutput {
        debug!("Mono pre fbank shape is {}", mono_audio.len());
        let fbank = self.preprocess_fbank(
            mono_audio,
            model_cfg.resampling_frequency,
            &model_cfg.fbank_cfg
        );
        let fbank = fbank.insert_axis(Axis(0));
        let fbank = fbank.permuted_axes([0, 2, 1]);
        // let fbank = fbank.to_owned();
        let fbank_mean = fbank.mean();
        debug!("Fbank mean: {:?}", fbank_mean);
        let shape = fbank.shape().to_vec();
        debug!("Permuted fbank shape is {:?}", shape);
        debug!("First elements fbank are {:?}", fbank.slice(s![0, 0..3, 0..30]));
        let tensor = Self::array_to_tensor::<f32>(fbank.into_dyn());
        debug!("ENCODER INPUTS: {:?}", self.encoder.inputs);
        debug!("ENCODER OUTPUTS: {:?}", self.encoder.outputs);
        let num_frames = shape[2] as i64;
        let lengths = vec![num_frames];
        let length_tensor = TensorRef::from_array_view(
            (vec![1], &lengths[..])
        ).expect("Error creating lengths tensor");
        let outputs = self.encoder.run(
            inputs![
                "audio_signal" => tensor,
                "length" => length_tensor,
            ]
        ).expect("error in encoder model"); 
        let keys: Vec<&str> = outputs.keys().collect();
        debug!("ENCODER OUTPUTS HAS LEN {:?}", outputs.len());
        debug!("ENCODER OUTPUTS HAS keys {:?}", keys);
        // info!("ENCODER OUTPUTS HAS values {:?}", outputs.values());
        // https://www.mdpi.com/2076-3417/14/24/11583
        // https://docs.rs/knf-rs/0.3.2/knf_rs/fn.compute_fbank.html
        // https://www.kaggle.com/code/ahmedabdoamin/preprocessing-speech-mfcc-vs-filter-banks
        // https://apxml.com/courses?page=3&level=3
        // https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/canary/canary.py
        // https://github.com/NVIDIA-NeMo/NeMo/tree/main/nemo/collections/asr/parts
        // https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/audio/data/audio_to_audio.py#L592
        Self::print_out_shape::<f32>(&outputs, "encoder_embeddings");
        Self::print_out_shape::<i64>(&outputs, "encoder_mask");
        EncoderOutput::new(outputs)
    }

    fn _decode(&mut self, decoder_input: DecoderInput) -> DecoderOutput {
        let outputs = self.decoder.run(
            inputs![
                "input_ids" => decoder_input.input_ids,
                "decoder_mems" => decoder_input.decoder_mems,
                "encoder_embeddings" => decoder_input.encoder_embeddings,
                "encoder_mask" => decoder_input.encoder_mask,
            ]
        ).expect("error in decoder model"); 
        let keys: Vec<&str> = outputs.keys().collect();
        debug!("decoder OUTPUTS HAS LEN {:?}", outputs.len());
        debug!("decoder OUTPUTS HAS keys {:?}", keys);
        // let mems = Self::infered_tensor::<f32>(&outputs, "decoder_hidden_states");
        DecoderOutput::new(outputs)
    }

    fn decode(&mut self, encoder_out: EncoderOutput, model_cfg: &Canary, audio: &AudioFile) {
        let vocab_path = Path::new("src/config/specs/vocab.txt");
        let tokenizer = Tokenizer::load_vocab(&vocab_path);
        let n_steps = model_cfg.max_tokens;
        let prev_ids = init_tokens(model_cfg, audio, &tokenizer);
        let prev_ids_tensor = Self::array_to_tensor(prev_ids.clone());
        info!("Input ids for init decoding: {:?}", prev_ids_tensor.try_extract_tensor::<i64>().unwrap());
        let decoder_mems = init_decoder_mems(model_cfg);
        // TODO: use prev_ids as Ref in to_out_input to avodi expensive clone()
        let decoder_input = encoder_out.to_out_input(prev_ids_tensor, decoder_mems);
        info!("Init pred");
        debug!("Decoder inputs are: {:?}", self.decoder.inputs);
        let mut predicted_tokens = Vec::<usize>::with_capacity(n_steps);
        let mut init_prediction = self._decode(decoder_input);
        init_prediction.store_pred(&mut predicted_tokens);
        for i in 1..=n_steps {
            if i % model_cfg.log_every_steps == 0 {
                info!("Decoding cycle no. {}", i);
            }
            let new_tokens = init_prediction.next_tokens();
            let last_token = new_tokens.last().expect("There arent new tokens");
            let new_ids = ArrayD::from_shape_vec(
                vec![1, 1], vec![*last_token as i64]
            ).expect("Error creatign new array id in decoder");
            let prev_ids_tensor = Self::array_to_tensor(new_ids);
            let decoder_mems = Self::array_to_tensor(
                init_prediction.decoder_hidden_states.clone()
            );
            debug!("Hidden: {:?}", init_prediction.decoder_hidden_states.clone().shape());
            let decoder_input = encoder_out.to_out_input(prev_ids_tensor, decoder_mems);
            init_prediction = self._decode(decoder_input);
            if let Some(token) = init_prediction.next_tokens().last() {
                if *token == 3 {
                    break
                }
            }
            init_prediction.store_pred(&mut predicted_tokens);
        }
        //println!("DECODER INPUTS: {:?}", self.decoder.inputs);
        //println!("DECODER OUTPUTS: {:?}", self.decoder.outputs);
        debug!("Finish predicted tokens: {:?}", predicted_tokens);
        let mut final_str = String::new();
        for token in predicted_tokens {
            // use if let?
            final_str += &tokenizer.detokenize(&token);
        }
        info!("Final string:\n{}", final_str)
    }
}

