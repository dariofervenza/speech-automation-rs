mod model;
mod audio;
use model::load::load_model;
use audio::load::AudioLoader;

fn main() {
    let debug = false;
    let mut model = load_model(debug).expect("error loading model");
    let path = "/home/dario/speech-automation-rs/harvard.wav";
    let loaded_audio = AudioLoader::from_path(path, debug);
    println!("First elements:\n{:?}", &loaded_audio.to_vec()[..10]);
    model.pipe(loaded_audio.to_vec(), debug);

}
