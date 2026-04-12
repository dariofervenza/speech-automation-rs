use cpal::traits::{ HostTrait,  DeviceTrait, StreamTrait};
use cpal::{ FromSample, Sample, Stream };
use log::info;
use std::any;
use std::sync::{Arc, Mutex};
use std::io::BufWriter;
use std::fs::File;
use std::sync::mpsc;


fn sample_format(format: cpal::SampleFormat) -> hound::SampleFormat {
    if format.is_dsd() {
        panic!("DSD formats cannot be written to WAV files");
    } else if format.is_float() {
        hound::SampleFormat::Float
    } else {
        hound::SampleFormat::Int
    }
}


fn wav_spec_from_config(config: &cpal::SupportedStreamConfig) -> hound::WavSpec {
    hound::WavSpec {
        channels: config.channels() as _,
        sample_rate: config.sample_rate() as _,
        bits_per_sample: (config.sample_format().sample_size() * 8) as _,
        sample_format: sample_format(config.sample_format()),
    }
}


type WavWriterHandle = Arc<Mutex<Option<hound::WavWriter<BufWriter<File>>>>>;


fn write_input_data<T, U>(input: &[T], writer: &WavWriterHandle)
where
    T: Sample,
    U: Sample + hound::Sample + FromSample<T>,
{
    if let Ok(mut guard) = writer.try_lock() {
        if let Some(writer) = guard.as_mut() {
            for &sample in input.iter() {
                let sample: U = U::from_sample(sample);
                writer.write_sample(sample).ok();
            }
        }
    }
}


pub fn capture_audio() -> Result<(), anyhow::Error>{
    let host = cpal::default_host();
    let devices = host.devices()?;
    info!("Listing all available audio devices:");
    for device in devices {
            // .name() returns a Result, so we unwrap it or provide a fallback
            let name = device.description()?;
            println!("Device: {}", name);
    }
    println!("\n\n");
    let devices = host.input_devices()?;
    for device in devices.into_iter() {
        let name = device.description()?;
        // This is the key: does this device actually support recording?
        if let Ok(configs) = device.supported_input_configs() && name.to_string().contains("PulseAudio") {
            let count = configs.count();
            if count > 0 {
                println!("Found MIC: {} (with {} configurations)", name, count);
            }
        }
    }
    let device = host.default_input_device();
    if let Some(dev) = device {
        let config = dev.default_input_config()?;
        const PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/recorded.wav");
        let spec = wav_spec_from_config(&config);
        println!("Spec: {:?}", spec);
        // Spec: WavSpec { channels: 2, sample_rate: 44100, bits_per_sample: 32, sample_format: Float }
        let writer = hound::WavWriter::create(PATH, spec)?;
        let writer = Arc::new(Mutex::new(Some(writer)));
        let writer_2 = writer.clone();

        let err_fn = move |err| {
            eprintln!("an error occurred on stream: {err}");
        };

        let stream = match config.sample_format() {
            cpal::SampleFormat::I8 => dev.build_input_stream(
                &config.into(),
                move |data, _: &_| write_input_data::<i8, i8>(data, &writer_2),
                err_fn,
                None,
            )?,
            cpal::SampleFormat::I16 => dev.build_input_stream(
                &config.into(),
                move |data, _: &_| write_input_data::<i16, i16>(data, &writer_2),
                err_fn,
                None,
            )?,
            cpal::SampleFormat::I32 => dev.build_input_stream(
                &config.into(),
                move |data, _: &_| write_input_data::<i32, i32>(data, &writer_2),
                err_fn,
                None,
            )?,
            cpal::SampleFormat::F32 => dev.build_input_stream(
                &config.into(),
                move |data, _: &_| write_input_data::<f32, f32>(data, &writer_2),
                err_fn,
                None,
            )?,
            sample_format => {
                return Err(anyhow::Error::msg(format!(
                    "Unsupported sample format '{sample_format}'"
                )))
            }
        };
        println!("Recording....");
        stream.play()?;
        std::thread::sleep(std::time::Duration::from_secs(10));
        drop(stream);
        writer.lock().unwrap().take().unwrap().finalize()?;
    }
    Ok(())
}


pub fn stream_audio() -> Result<(Stream, mpsc::Receiver<f32>, u32), anyhow::Error> {
    let host = cpal::default_host();
    let device = host.default_input_device();
    let device = device.ok_or_else(|| anyhow::anyhow!("No device found"))?;
    let config = device.default_input_config()?;
    let err_fn = move |err| {
        eprintln!("an error occurred on stream: {err}");
    };
    let (tx, rx) = mpsc::channel::<f32>();
    let sample_rate = config.sample_rate();
    let stream = match config.sample_format() {
        
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data, _: &_| {
                for &sample in data {
                    let _ = tx.send(sample);
                }
            },
            err_fn,
            None,
        )?,
        sample_format => {
            return Err(anyhow::Error::msg(format!(
                "Unsupported sample format '{sample_format}'"
            )))
        }
    };
    println!("Started stream....");
    stream.play()?;
    Ok((stream, rx, sample_rate))
}


pub fn take_audio_chunks(rx: &mpsc::Receiver<f32>, sample_rate: u32, n_seconds: u32) -> Vec<f32> {
    let nsamples = sample_rate * n_seconds;
    let mut vec_stream = Vec::with_capacity(nsamples as usize);
    while vec_stream.len() < vec_stream.capacity() {
        if let Ok(sample) = rx.recv() {
            vec_stream.push(sample);
        }
    }
    vec_stream
} 
