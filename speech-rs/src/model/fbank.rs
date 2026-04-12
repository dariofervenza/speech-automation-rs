use crate::config::types::FbankConfig;
use ndarray::{ Array2, Axis, s };
use kaldi_native_fbank::fbank::{ FbankComputer, FbankOptions };
use kaldi_native_fbank::online::{ OnlineFeature, FeatureComputer };
use log::{ debug, info };


pub trait FbankProcessor {
    fn preprocess_fbank(&self, mono_audio: Vec<f32>, sampling_freq: f32, fbank_cfg: &FbankConfig) -> Array2<f32> {
        let mut opts = FbankOptions::default();
        let num_bins = 128;
        // https://kaldi-asr.org/doc/structkaldi_1_1FbankOptions.html
        // https://kaldi-asr.org/doc/structkaldi_1_1FrameExtractionOptions.html
        // https://kaldi-asr.org/doc/feature-window_8h_source.html
        opts.mel_opts.num_bins = fbank_cfg.num_mel_features;
        opts.frame_opts.samp_freq = sampling_freq;
        opts.frame_opts.dither = fbank_cfg.dither;
        opts.frame_opts.frame_shift_ms = fbank_cfg.frame_shift_ms;
        opts.frame_opts.frame_length_ms = fbank_cfg.frame_length_ms;
        opts.frame_opts.window_type = fbank_cfg.window_type.clone();
        // round will do 16000 * 0.001 * 25 frame lenth = 400 --> round to 512
        opts.frame_opts.round_to_power_of_two = fbank_cfg.round_to_power_of_two;  // not important
        opts.frame_opts.snip_edges = fbank_cfg.snip_edges;  // important
        opts.frame_opts.preemph_coeff = fbank_cfg.preemph_coeff;
        // opts.frame_opts.remove_dc_offset = false; // not important
        // opts.energy_floor = 1.0;
        // opts.mel_opts.htk_mode = false; // not important
        opts.use_energy = fbank_cfg.use_energy;  // important
        opts.raw_energy = fbank_cfg.raw_energy;  // important
        opts.use_log_fbank = fbank_cfg.use_log_fbank;   // important
        // opts.use_power = false;
        let computer = FbankComputer::new(opts.clone()).expect("Error creating the computer");
        let mut online_fbank = OnlineFeature::new(FeatureComputer::Fbank(computer));
        online_fbank.accept_waveform(sampling_freq, &mono_audio);
        online_fbank.input_finished();
        let num_frames = online_fbank.num_frames_ready();
        let mut all_features = Vec::with_capacity(num_frames * num_bins);
        for i in 0..num_frames {
            let frame = online_fbank.get_frame(i);
            match frame {
                Some(value) => all_features.extend_from_slice(value),
                None => debug!("[SKIP] Got none in the fbanc frame"),
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
        debug!("INITITAL fbank shape is: {:?}", fbank.shape());
        Self::normalize(fbank)
    }

    fn normalize(fbank: Array2<f32>) -> Array2<f32> {
        // apply before permutting
        debug!("Normalize original {:?}", fbank.slice(s![0, 0..15]));
        let mean_axis = fbank
            .mean_axis(Axis(0))
            .expect("Error computing mean fbank")
            .insert_axis(Axis(0));
        let std_axis = fbank
            .std_axis(Axis(0), 1 as f32)
            .insert_axis(Axis(0));
        debug!("Normalize org shape: {:?}", fbank.shape());
        debug!("Normalize mean shape: {:?}", mean_axis.shape());
        debug!("Normalize std shape: {:?}", std_axis.shape());
        let fbank: Array2<f32> = (fbank - mean_axis) / (std_axis + 1.0e-6);
        debug!("Normalize standard fbank {:?}", fbank.slice(s![0, 0..15]));
        fbank
        }
}