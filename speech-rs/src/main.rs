mod model;
mod audio;
mod config;
use model::load::load_model;
use model::canary::CanaryModel;
use audio::load::AudioLoader;
use audio::capture::{ stream_audio, take_audio_chunks };
use config::types::{ GlobalConfig, AudioFile };
use log::{ debug, info };
use std::io::Write;
use std::sync::mpsc;

fn normalize_audio(data: &mut Vec<f32>) {
    let max_value = data.iter().fold(0.0, |max, &s| s.abs().max(max));
    if max_value > 0.0 {
        for sample in data.iter_mut() {
            *sample /= max_value;
        }
    }
}

fn apply_gain(data: &mut Vec<f32>, factor: f32) {
    for sample in data.iter_mut() {
        // factor 2.0 = +6dB, factor 4.0 = +12dB
        *sample = (*sample * factor).clamp(-1.0, 1.0);
    }
}


fn ai_processing(
    rx: &mpsc::Receiver<f32>,
    sample_rate: u32,
    app_cfg: &GlobalConfig
) -> Result<(), anyhow::Error> {
    let app_cfg_thread = app_cfg.clone();
    let dummy_audio = AudioFile {
        original_file: String::from("Live_capture.wav"),
        resampled_file: String::from("Live_capture_reampled.wav"),
        mono_file: String::from("Live_capture_mono.wav"),
        save_resampled: false,
        save_mono: false,
        source_lang: String::from("<|es|>"),
        target_lang: String::from("<|es|>"),
    };
    let (tx_model, rx_model) = mpsc::channel::<Vec<f32>>();
    std::thread::spawn(move || {
        let mut model = load_model(&app_cfg_thread)?; 
        info!("Model loaded");
        let mut transcript = String::new(); 
        let mut n_seconds = 0 as u64;
        let mut n_save = 1 as usize;
        loop {
            let mut loaded_audio = rx_model.recv()?;
            let rms = (
                loaded_audio.iter().map(|&s| s * s).sum::<f32>() / loaded_audio.len() as f32
            ).sqrt();
            if rms > 0.0070 {
                println!("{rms}\n\n");
                normalize_audio(&mut loaded_audio);
                apply_gain(&mut loaded_audio, 1.5);
                debug!("Audio reached thread with rms > 0.01");
                let last_transcript = model.pipe(
                    loaded_audio.to_vec(),
                    &app_cfg_thread.models.canary,
                    &dummy_audio
                );
                transcript += format!("\n{}", last_transcript).as_str();
                n_seconds += 40;
                let n_minutes = n_seconds / 60;
                if n_minutes > 20 {
                    let file_name = format!("files/transcript_no_{n_save}.txt");
                    let mut file = std::fs::File::create(file_name)?;
                    file.write_all(transcript.as_bytes())?;
                    transcript = String::from(last_transcript);
                    n_seconds = 0;
                    n_save += 1;
                } 
            }

        }
        Ok::<(), anyhow::Error>(())
    });
    loop {
        let audio_buffer = take_audio_chunks(rx, sample_rate, 40);
        let loaded_audio = AudioLoader::from_samples(audio_buffer, sample_rate, app_cfg);
        debug!("\n\nGot audio chunk\n\n\n");
        let _ = tx_model.send(loaded_audio.to_vec())?;
        std::thread::sleep(std::time::Duration::from_secs(1));   
    }

    Ok(())
}

fn main() -> Result<(), anyhow::Error> {
    // LD_LIBRARY_PATH=/home/dario/dario/Downloads/Rust/speech-automation-rs/speech-rs/onnxruntime-linux-x64-gpu-1.22.0/lib cargo run
    let app_cfg = GlobalConfig::from_file("src/config/specs/config.toml");
    app_cfg.set_log_level();
    
    let (stream, rx, sample_rate) = stream_audio()?;
    let _ = ai_processing(&rx, sample_rate, &app_cfg)?;
    drop(stream);
    Ok(())
}


fn old_main() -> Result<(), anyhow::Error> {
    // LD_LIBRARY_PATH=/home/dario/dario/Downloads/Rust/speech-automation-rs/speech-rs/onnxruntime-linux-x64-gpu-1.22.0/lib cargo run
    let app_cfg = GlobalConfig::from_file("src/config/specs/config.toml");
    app_cfg.set_log_level();
    let mut model = load_model(&app_cfg).expect("error loading model"); 
    let test_audio_file = &app_cfg.audio_wav.testfiles
        .first()
        .expect("No AUDIO FILE PROVIDED");
    let loaded_audio = AudioLoader::from_path(
        &test_audio_file,
        &app_cfg,
    );
    debug!("First elements:\n{:?}", &loaded_audio.to_vec()[..10]);
    model.pipe(loaded_audio.to_vec(), &app_cfg.models.canary, test_audio_file);
    let (stream, rx, sample_rate) = stream_audio()?;
    ai_processing(&rx, sample_rate, &app_cfg);
    Ok(())
}
