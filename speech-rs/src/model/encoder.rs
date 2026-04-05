use crate::model::decoder::DecoderInput;
use crate::model::tensor::TensorProcessor;
use ort::session::{ SessionOutputs };
use ort::value::{ TensorValueType, Value };
use ndarray::{ ArrayD };
use log::{ info };


pub struct EncoderOutput {
    embeddings: ArrayD<f32>,
    mask: ArrayD<i64>,
}

impl TensorProcessor for EncoderOutput {}

impl EncoderOutput {
    pub fn new(out: SessionOutputs<'_>) -> Self {
        let embeddings = Self::try_tensor::<f32>(&out, "encoder_embeddings");
        let mask = Self::try_tensor::<i64>(&out, "encoder_mask");
        info!("Embeddings shape: {:?}", embeddings.shape());
        info!("Mask shape: {:?}", mask.shape());
        EncoderOutput {
            embeddings, 
            mask,
        }
    }

    pub fn to_out_input(&self, prev_ids: Value<TensorValueType<i64>>, decoder_mems: Value<TensorValueType<f32>>) -> DecoderInput {
        DecoderInput {
            input_ids: prev_ids,
            decoder_mems: decoder_mems,
            encoder_embeddings: Self::array_to_tensor::<f32>(self.embeddings.clone()),
            encoder_mask: Self::array_to_tensor::<i64>(self.mask.clone()),
        }

    }
}

