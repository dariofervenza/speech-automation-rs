use wavers::{ Wav, IntoNdarray };
use ndarray::{ Axis, Array2, Array };
use rubato::{ InterpolationParameters, InterpolationType, Resampler, SincFixedIn, WindowFunction };


fn resample_audio_vec(input_data: &Vec<Vec<f32>>, input_rate: f64, output_rate: f64) -> Vec<Vec<f32>> {
    let params = InterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: InterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let chunk_size = if input_data.len() > 0 {input_data[0].len()} else {0};
    let mut resampler = SincFixedIn::<f32>::new(
        output_rate / input_rate,
        2.0,
        params,
        chunk_size,
        2,
    ).expect("Error creating resampler");
    resampler.process(input_data, None).expect("Error resampling audio")
}


pub fn load_audio_file(path: &str) -> Array2<f32> {
    let wav: Wav<f32> = Wav::from_path(path).expect("Error loading audio file");
    let wav_spec = wav.wav_spec();
    println!("Wav spec: {:?}", wav_spec);
    let (f32_array, sample_rate) = wav
        .into_ndarray()
        .expect("Error converting to ndarray");
    println!("Samples rate is {}", sample_rate);
    println!("Array shape: {:?}", f32_array.shape());
    let vec_audio: Vec<Vec<f32>> = f32_array
        .axis_iter(Axis(1))
        .map(|row| {row.to_vec()})
        .collect(); // ---------------------------> use more modern version of rubato that has audio buffers
    let resampled = resample_audio_vec(
        &vec_audio, sample_rate as f64, 16000 as f64
    );
    let rows = resampled.len();
    let cols = if rows > 0 {resampled[0].len()} else {0};
    let flat: Vec<f32> = resampled.into_iter().flatten().collect();
    let f32_array = Array2::from_shape_vec((rows, cols), flat)
        .expect("Error flatteing audio");
    println!("Array shape: {:?}", f32_array.shape());
    let flat_array = f32_array.mean_axis(Axis(1)).expect("Failed to convert to mono");
    println!("Flat shape: {:?}", flat_array.shape());
    let final_array = flat_array.insert_axis(Axis(0));
    final_array
}