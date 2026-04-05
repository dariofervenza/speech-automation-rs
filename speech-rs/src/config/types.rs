use serde::Deserialize;


#[derive(Deserialize, Debug)]
pub struct ResampleConfig {
    pub chunk_size: usize,
    pub sub_chunks: usize,
    pub nbr_channels: usize,
}


#[derive(Deserialize, Debug)]
pub struct AudioFile {
    pub original_file: String,
    pub resampled_file: String,
    pub mono_file: String,
    pub save_resampled: bool,
    pub save_mono: bool,
    pub source_lang: String,
    pub target_lang: String,
}


#[derive(Deserialize, Debug)]
pub struct AudioWav {
    pub resample: ResampleConfig,
    pub testfiles: Vec<AudioFile>
}


#[derive(Deserialize, Debug)]
pub struct ModelPath {
    pub encoder: String,
    pub decoder: String,
}


#[derive(Deserialize, Debug)]
pub struct InitDecoderMems {
    pub layers: usize,
    pub batch: usize,
    pub mems_len: usize,
    pub hidden_size: usize
}

#[derive(Deserialize, Debug)]
pub struct FbankConfig {
    pub num_mel_features: usize,
    pub dither: f32,
    pub frame_shift_ms: f32,
    pub frame_length_ms: f32,
    pub window_type: String,
    pub round_to_power_of_two: bool,
    pub snip_edges: bool,
    pub preemph_coeff: f32,
    pub use_energy: bool,
    pub raw_energy: bool,
    pub use_log_fbank: bool,
}


#[derive(Deserialize, Debug)]
pub struct Canary {
    pub resampling_frequency: f32,
    pub max_tokens: usize,
    pub log_every_steps: usize,
    pub init_prompt: Vec<i64>,
    pub model_path: ModelPath,
    pub init_decoder_mems: InitDecoderMems,
    pub fbank_cfg: FbankConfig,
}


#[derive(Deserialize, Debug)]
pub struct Models {
    pub canary: Canary,
}


#[derive(Deserialize, Debug)]
pub struct Logging {
    pub level: String,
    pub logs_folder: String,
    _remove_after: String,
}


#[derive(Deserialize, Debug)]
pub struct GlobalConfig {
    pub audio_wav: AudioWav,
    pub models: Models,
    pub logging: Logging
}
