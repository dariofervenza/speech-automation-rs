mod model;
mod audio;
mod config;
use model::load::load_model;
use audio::load::AudioLoader;
use config::types::GlobalConfig;
use log::debug;

fn main() {
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

}
