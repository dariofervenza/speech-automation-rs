mod model;
mod audio;
use model::load::load_model;
use audio::load::load_audio_file;
use ndarray::s;


fn main() {
    let mut model = load_model();
    let path = "/home/dario/speech-automation-rs/harvard.wav";
    let audio_arr = load_audio_file(path);
    println!("First elements:\n{:?}", audio_arr.slice(s![.., 0..10]));

}
