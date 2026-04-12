use crate::config::types::{ GlobalConfig, Canary, AudioFile, ResampleConfig };
use std::path::Path;
use wavers::{ Wav, AsNdarray, Samples, write };
use ndarray::{ Array2, s };
use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{ Resampler, Fft, FixedSync };
use audioutils::{ interleaved_samples_to_stereos, stereos_to_mono_channel, };
use log::{ debug, info };


pub struct AudioLoader {
    mono_audio: Vec<f32>
}

impl AudioLoader {
    pub fn from_path(audio_file: &AudioFile, app_cfg: &GlobalConfig) -> Self {
        let canary_cfg = &app_cfg.models.canary;
        let resample_cfg = &app_cfg.audio_wav.resample;
        AudioLoader {
            mono_audio: Self::load_audio_file(&audio_file, canary_cfg, resample_cfg)
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.mono_audio.clone()
    }

    pub fn from_samples(samples: Vec<f32>, sample_rate: u32, app_cfg: &GlobalConfig)  -> Self {
        let resample_cfg = &app_cfg.audio_wav.resample;
        let audio_cfg = &app_cfg.models.canary;
        let resample_freq = audio_cfg.resampling_frequency as usize;
        let resampled = Self::resample_audio_vec(
            &samples,
            sample_rate as usize,
            resample_freq,
            resample_cfg
        );
        let stereos = interleaved_samples_to_stereos(&resampled).expect("error converting to stereos");
        let mono_audio = stereos_to_mono_channel(&stereos);
        AudioLoader {
            mono_audio
        }
    }

    fn resample_audio_vec(
        input_data: &Vec<f32>,
        input_rate: usize,
        output_rate: usize,
        resample_cfg: &ResampleConfig
    ) -> Vec<f32> {
        let mut resampler_obj = Fft::<f32>::new(
            input_rate, 
            output_rate, 
            resample_cfg.chunk_size,
            resample_cfg.sub_chunks,
            resample_cfg.nbr_channels,
            FixedSync::Both
        ).expect("Error creating the resampler");
        let input_n_frames = input_data.len() / 2;
        let needed_len = resampler_obj.process_all_needed_output_len(input_n_frames);
        debug!("Resampling needed len {}", needed_len);
        let input_adapter = InterleavedSlice::new(
            input_data, resample_cfg.nbr_channels, input_n_frames
        ).expect("error creaating input adapter");
        let mut output_data = vec![0.0; resample_cfg.nbr_channels * needed_len];
        let output_n_frames = output_data.len() / resample_cfg.nbr_channels;
        let mut output_adapter = InterleavedSlice::new_mut(
            &mut output_data, resample_cfg.nbr_channels, output_n_frames
        ).expect("error creaating output adapter");
        let resampling_result = resampler_obj.process_all_into_buffer(
            &input_adapter, &mut output_adapter, input_n_frames, None
        ).expect("Error resampling");
        debug!("Resampling res: {:?}", resampling_result);
        debug!("Output resampled vect len is {}", output_data.len());
        output_data
    }

    fn load_audio_file(audio_file: &AudioFile, audio_cfg: &Canary, resample_cfg: &ResampleConfig) -> Vec<f32> {
        let resample_freq = audio_cfg.resampling_frequency as usize;
        let mut wav: Wav<f32> = Wav::from_path(
            audio_file.original_file.as_str()
        ).expect("Error loading audio file");
        let wav_spec = wav.wav_spec();
        let sample_rate: usize = wav.sample_rate() as usize;
        debug!("Wav spec: {:?}", wav_spec);
        debug!("Original sample rate: {}", sample_rate);
        let samples = wav.read().expect("error getting samples"); 
        let audio_vec = samples.to_vec();
        let (array_f32, _): (Array2<f32>, i32) = wav.as_ndarray().unwrap();
        // interleaved as vector
        debug!("Interleaved Vec has len {}", audio_vec.len());
        debug!("Original wav shape: {:?}", array_f32.shape());
        // Array shape: [809508, 2]
        debug!("First elements array channel 1 {:?}", array_f32.slice(s![..10, 0]));
        debug!("First elements array channel 2 {:?}", array_f32.slice(s![..10, 1]));
        let resampled = Self::resample_audio_vec(
            &audio_vec,
            sample_rate as usize,
            resample_freq,
            resample_cfg
            
        );
        debug!("First elements of resampled: {:?}", &resampled[..20]);
        let samples_resample: Samples<f32> = Samples::from(resampled);
        if audio_file.save_resampled {
            let out_path = &Path::new(audio_file.resampled_file.as_str());
            write(
                out_path,
                &samples_resample,
                resample_freq as i32,
                resample_cfg.nbr_channels as u16
            ).expect("Error writing output resampled wav");
        }
        let out = samples_resample.to_vec();
        let stereos = interleaved_samples_to_stereos(&out).expect("error converting to stereos");
        let mono = stereos_to_mono_channel(&stereos);
        debug!("First elements of mono audio: {:?}", &mono[0..30]);
        debug!("Mono is {:?}", mono.len());
        if audio_file.save_mono {
            let samples_mono: Samples<f32> = Samples::from(mono.clone());
            let out_path = &Path::new(audio_file.mono_file.as_str());
            write(
                out_path,
                &samples_mono,
                resample_freq as i32,
                1
            ).expect("Error writing output mono wav");
        }
        mono
    }
}
