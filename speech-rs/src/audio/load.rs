use std::path::Path;
use wavers::{ Wav, AsNdarray, Samples, write };
use ndarray::{ Array2, s };
use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{ Resampler, Fft, FixedSync };
use audioutils::{ interleaved_samples_to_stereos, stereos_to_mono_channel, };

pub struct AudioLoader {
    mono_audio: Vec<f32>
}


impl AudioLoader {
    pub fn from_path(path: &str, debug: bool) -> Self {
        AudioLoader {
            mono_audio: AudioLoader::load_audio_file(path, debug),
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.mono_audio.clone()
    }

    fn resample_audio_vec(
        input_data: &Vec<f32>, input_rate: usize, output_rate: usize, debug: bool
    ) -> Vec<f32> {
        let mut resampler_obj = Fft::<f32>::new(
            input_rate, 
            output_rate, 
            1024,
            2,
            2,
            FixedSync::Both
        ).expect("Error creating the resampler");
        let input_n_frames = input_data.len() / 2;
        let needed_len = resampler_obj.process_all_needed_output_len(input_n_frames);
        if debug {
            println!("Needed len {}", needed_len);
        }
        let input_adapter = InterleavedSlice::new(
            input_data, 2, input_n_frames
        ).expect("error creaating input adapter");
        let mut output_data = vec![0.0; 2 * needed_len];
        let output_n_frames = output_data.len() / 2;
        let mut output_adapter = InterleavedSlice::new_mut(
            &mut output_data, 2, output_n_frames
        ).expect("error creaating output adapter");
        let resampling_result = resampler_obj.process_all_into_buffer(
            &input_adapter, &mut output_adapter, input_n_frames, None
        ).expect("Error resampling");
        if debug {
            println!("Resampling res: {:?}", resampling_result);
            println!("Output resampled vect len is {}", output_data.len());
        }

        output_data

    }

    fn load_audio_file(path: &str, debug: bool) -> Vec<f32> {
        let mut wav: Wav<f32> = Wav::from_path(path).expect("Error loading audio file");
        let wav_spec = wav.wav_spec();
        let sample_rate: usize = wav.sample_rate() as usize;
        if debug {
            println!("Wav spec: {:?}", wav_spec);
            println!("Original sample rate: {}", sample_rate);
        }
        let samples = wav.read().expect("error getting samples"); 
        let audio_vec = samples.to_vec();
        let (array_f32, _): (Array2<f32>, i32) = wav.as_ndarray().unwrap();
        if debug {
            // interleaved as vector
            println!("Vec has len {}", audio_vec.len());
            // Array shape: [809508, 2]
            println!("First elements array channel 1 {:?}", array_f32.slice(s![..10, 0]));
            println!("First elements array channel 2 {:?}", array_f32.slice(s![..10, 1]));
        }
        let resampled = AudioLoader::resample_audio_vec(
            &audio_vec, sample_rate as usize, 16000, debug
        );
        if debug {
            println!("First elements of resampled: {:?}", &resampled[..20]);
        }
        let samples_resample: Samples<f32> = Samples::from(resampled);
        let out_path = &Path::new("/home/dario/speech-automation-rs/harvard_resampled.wav");
        write(out_path, &samples_resample, 16000, 2).unwrap();
        let out = samples_resample.to_vec();
        let stereos = interleaved_samples_to_stereos(&out).expect("error converting to stereos");
        let mono = stereos_to_mono_channel(&stereos);
        if debug {
            println!("Mono is {:?}", mono.len());
        }
        mono
    }
}

