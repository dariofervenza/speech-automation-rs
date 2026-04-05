use crate::model::tensor::TensorProcessor;
use ort::value::{ TensorValueType, Value };
use ort::session::{ SessionOutputs };
use ndarray::{ ArrayD, Axis, s };
use log::{ debug };
use ndarray_stats::QuantileExt;


pub struct DecoderInput {
    pub input_ids: Value<TensorValueType<i64>>,
    pub decoder_mems: Value<TensorValueType<f32>>,
    pub encoder_embeddings: Value<TensorValueType<f32>>,
    pub encoder_mask: Value<TensorValueType<i64>>,
}


pub struct DecoderOutput {
    pub logits: ArrayD<f32>,
    pub decoder_hidden_states: ArrayD<f32>,
}

impl TensorProcessor for DecoderOutput {}

impl DecoderOutput {
    pub fn new(out: SessionOutputs<'_>) -> Self {
        let logits = Self::try_tensor::<f32>(&out, "logits");
        let decoder_hidden_states = Self::try_tensor::<f32>(
            &out, "decoder_hidden_states"
        );
        debug!("Logits shape: {:?}", logits.shape());
        debug!("Decoder hidden shape: {:?}", decoder_hidden_states.shape());
        DecoderOutput {
            logits,
            decoder_hidden_states
        }
    }

    pub fn next_tokens(&self) -> Vec<usize> {
        let logits_arr = self.logits.clone();
        // TODO change this by ndarray_stats??
        let argmax_per_row: Vec<usize> = logits_arr
            .clone()
            .remove_axis(Axis(0))
            .axis_iter(Axis(0)) // iterate over rows
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap()
            })
            .collect();
        debug!("NEXT TOKEN IS  {:?}", argmax_per_row.last());
        let max_logit = logits_arr.slice(s![0, 0, ..]).to_owned();
        let max_logit = max_logit.max();
        debug!("MAX LOGIT: {:?}", max_logit);

        argmax_per_row
    }

    pub fn store_pred(&self, out_vec: &mut Vec<usize>) {
        let new_tokens = self.next_tokens();
        let last_token = new_tokens.last().expect("There arent new tokens");
        out_vec.push(*last_token);
    }
}