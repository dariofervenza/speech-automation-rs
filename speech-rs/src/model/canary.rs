use ort::session::{ Session, SessionOutputs };
use ort::{ Result, Error, inputs };
use ort::value::{ TensorRef, TensorValueType, Value };
use ort::tensor::PrimitiveTensorElementType;
use ndarray::{ Axis, Array2, ArrayD, };
use kaldi_native_fbank::fbank::{ FbankComputer, FbankOptions };
use kaldi_native_fbank::online::{ OnlineFeature, FeatureComputer };
use std::fmt::Debug;

fn init_tokens() -> Value<TensorValueType<i64>> {
    let vec_init: Vec<i64> = vec![16053, 7, 4, 16, 64, 64, 5, 9, 11, 13];
    let array_init = ArrayD::from_shape_vec(
        vec![1, vec_init.len()], vec_init
    ).expect("Error creating array init");
    Value::from_array(array_init).expect("Error creating init tensor")
}


fn init_decoder_mems() -> Value<TensorValueType<f32>> {
    let layers = 10;
    let batch = 1;
    let mems_len = 0; // The first pass has NO history
    let hidden_size = 1024;
    let mems_shape = vec![layers, batch, mems_len, hidden_size];
    let empty_data: Vec<f32> = Vec::new();
    Value::from_array(
        ArrayD::from_shape_vec(mems_shape, empty_data)
            .expect("Failed to create empty mems shape")
    ).expect("Failed to create ORT Value")
}


pub struct CanaryModel {
    encoder: Session,
    decoder: Session,
}


struct DecoderInput {
    pub input_ids: Value<TensorValueType<i64>>,
    pub decoder_mems: Value<TensorValueType<f32>>,
    pub encoder_embeddings: Value<TensorValueType<f32>>,
    pub encoder_mask: Value<TensorValueType<i64>>,
}


struct EncoderOutput {
    embeddings: ArrayD<f32>,
    mask: ArrayD<i64>,
}

impl EncoderOutput {
    pub fn new(out: SessionOutputs<'_>) -> Self {
        let embeddings = &out["encoder_embeddings"];
        let mask = &out["encoder_mask"];
        let embeddings = embeddings.try_extract_tensor::<f32>().expect("Error extracting embeddings").to_owned();
        let mask = mask.try_extract_tensor::<i64>().expect("Error extracting mask").to_owned();
        let embeddings_shape: Vec<usize> = (embeddings.0).iter().map(
            |&x| x as usize
        ).collect();
        let mask_shape: Vec<usize> = mask.0.iter().map(
            |&x| x as usize
        ).collect();
        let embeddings = ArrayD::from_shape_vec(
            embeddings_shape, embeddings.1.to_vec()
        ).expect("Error extracting embedings array");
        let mask = ArrayD::from_shape_vec(
            mask_shape, mask.1.to_vec()
        ).expect("Error extracting embedings array");
        println!("Embeddings shape: {:?}", embeddings.shape());
        println!("Mask shape: {:?}", mask.shape());
        EncoderOutput {
            embeddings, 
            mask,
        }
    }

    fn array_to_tensor<T>(array_in: ArrayD<T>) -> Value<TensorValueType<T>>
    where
        T: PrimitiveTensorElementType + Debug + Clone + 'static
    {
        Value::from_array(array_in).expect("Error creating out in tensorref")
    }

    pub fn to_out_input(&self, prev_ids: Value<TensorValueType<i64>>, decoder_mems: Value<TensorValueType<f32>>) -> DecoderInput {
        DecoderInput {
            input_ids: prev_ids,
            decoder_mems: decoder_mems,
            encoder_embeddings: EncoderOutput::array_to_tensor::<f32>(self.embeddings.clone()),
            encoder_mask: EncoderOutput::array_to_tensor::<i64>(self.mask.clone()),
        }

    }
}


impl CanaryModel {
    pub fn from_models(encoder: Result<Session, Error>, decoder: Result<Session, Error>) -> Result<CanaryModel, Error> {
        let model = CanaryModel {
            encoder: encoder?,
            decoder: decoder?
        };
        
        Ok(model)
    }

    fn preprocess_fbank(&self, mono_audio: Vec<f32>, debug: bool) -> Array2<f32> {
        let mut opts = FbankOptions::default();
        let num_bins = 128;
        opts.mel_opts.num_bins = num_bins;
        opts.frame_opts.samp_freq = 16000.0;
        opts.frame_opts.dither = 0.0;
        let computer = FbankComputer::new(opts.clone()).expect("Error creating the computer");
        let mut online_fbank = OnlineFeature::new(FeatureComputer::Fbank(computer));
        online_fbank.accept_waveform(16000.0, &mono_audio);
        online_fbank.input_finished();
        let num_frames = online_fbank.num_frames_ready();
        let mut all_features = Vec::with_capacity(num_frames * num_bins);
        for i in 0..num_frames {
            let frame = online_fbank.get_frame(i);
            match frame {
                Some(value) => all_features.extend_from_slice(value),
                None => println!("[SKIP] Got none in the fbanc frame"),
            }
        }
        let total_elements = all_features.len();
        let actual_num_frames = total_elements / num_bins;
        let truncated_len = actual_num_frames * num_bins;
        if all_features.len() > truncated_len {
            all_features.truncate(truncated_len);
        }
        let fbank = Array2::from_shape_vec(
            (actual_num_frames, num_bins), all_features
        ).expect("Error creating array2 fbank");
        if debug {
            println!("fbank shape is: {:?}", fbank.shape());
        }
        fbank
    }

    fn print_out_shape<T>(out: &SessionOutputs, key_name: &str)
    where 
        T: PrimitiveTensorElementType
    {
        let values = &out[key_name];
        let tensor = values.try_extract_tensor::<T>().expect("Error extracting out tensor");
        println!("{} shape is: {:?}", key_name, &tensor.0);
    }

    pub fn pipe(&mut self, input_vec: Vec<f32>, debug: bool) {
        for i in 0..1 {
            println!("Pass no: {}", i);
            let encoder_out = self.encode(input_vec.clone(), debug);
            self.decode(encoder_out);
        }

    }

    fn encode(&mut self, mono_audio: Vec<f32>, debug: bool) -> EncoderOutput {
        let fbank = self.preprocess_fbank(mono_audio, debug);
        let fbank = fbank.insert_axis(Axis(0));
        let fbank = fbank.permuted_axes([0, 2, 1]);
        // let fbank = fbank.to_owned();
        let shape = fbank.shape().to_vec();
        // println!("Permuted shape is {:?}", shape);
        // println!("First elements fbank are {:?}", fbank.slice(s![0, 0..1, 0..10]));
        let tensor = Value::from_array(
            fbank
        ).expect("error creating tensor");
        // println!("ENCODER INPUTS: {:?}", self.encoder.inputs());
        // println!("ENCODER OUTPUTS: {:?}", self.encoder.outputs());
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
        if true {
            println!("ENCODER OUTPUTS HAS LEN {:?}", outputs.len());
            println!("ENCODER OUTPUTS HAS keys {:?}", keys);
        }
        // println!("ENCODER OUTPUTS HAS values {:?}", outputs.values());
        // https://www.mdpi.com/2076-3417/14/24/11583
        // https://docs.rs/knf-rs/0.3.2/knf_rs/fn.compute_fbank.html
        // https://www.kaggle.com/code/ahmedabdoamin/preprocessing-speech-mfcc-vs-filter-banks
        // https://apxml.com/courses?page=3&level=3
        // https://github.com/Blaizzy/mlx-audio/blob/main/mlx_audio/stt/models/canary/canary.py
        // https://github.com/NVIDIA-NeMo/NeMo/tree/main/nemo/collections/asr/parts
        // https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/audio/data/audio_to_audio.py#L592
        CanaryModel::print_out_shape::<f32>(&outputs, "encoder_embeddings");
        CanaryModel::print_out_shape::<i64>(&outputs, "encoder_mask");
        EncoderOutput::new(outputs)
    }

    fn decode(&mut self, encoder_out: EncoderOutput) {
        let prev_ids = init_tokens();
        let decoder_mems = init_decoder_mems();
        let decoder_input = encoder_out.to_out_input(prev_ids, decoder_mems);
        let outputs = self.decoder.run(
            inputs![
                "input_ids" => decoder_input.input_ids,
                "decoder_mems" => decoder_input.decoder_mems,
                "encoder_embeddings" => decoder_input.encoder_embeddings,
                "encoder_mask" => decoder_input.encoder_mask,
            ]
        ).expect("error in encoder model"); 
        let keys: Vec<&str> = outputs.keys().collect();
        println!("decoder OUTPUTS HAS LEN {:?}", outputs.len());
        println!("decoder OUTPUTS HAS keys {:?}", keys);
        println!("ENCODER OUTPUTS: {:?}", self.encoder.outputs);
        let mems =  &outputs["decoder_hidden_states"];
        let mems = mems.try_extract_tensor::<f32>().expect("error extractign mems");
        println!("decoder mems {:?}", mems.0);
        let logits = &outputs["logits"];
        let logits = logits.try_extract_tensor::<f32>().expect("error extractign logints");
        let logits_val = logits.1;
        let logits_arr = ArrayD::from_shape_vec(
            vec![1 as usize, logits_val.len()], logits_val.to_vec()
        ).expect("ererro creating logits pred array");
        println!("logits with shape {:?}", logits.0);
        let argmax_per_row: Vec<usize> = logits_arr
            .axis_iter(Axis(0)) // iterate over rows
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()
            })
            .collect();

        println!("nEXT TOKEN IS  {:?}", argmax_per_row);

        //println!("DECODER INPUTS: {:?}", self.decoder.inputs);
        //println!("DECODER OUTPUTS: {:?}", self.decoder.outputs);


    }
}

