use crate::config::types::GlobalConfig;
use log::{debug, error, info, trace, warn};
use std::time::SystemTime;
use std::path::{ Path };


impl GlobalConfig {
    pub fn set_log_level(&self) {
        let base_path = Path::new(
            self.logging.logs_folder.as_str()
        );
        let _ = std::fs::create_dir(base_path);
        let log_folder = base_path.join("logs.log");
        let log_folder = log_folder.to_str().expect("Error creating log file str path");
        let level = match self.logging.level.to_lowercase().as_str() {
            "error" => log::LevelFilter::Error,
            "warn" | "warning" => log::LevelFilter::Warn,
            "info" => log::LevelFilter::Info,
            "debug" => log::LevelFilter::Debug,
            _ => log::LevelFilter::Info,
        };
        fern::Dispatch::new()
            .format(|out, message, record| {
                out.finish(format_args!(
                    "[{} {}, {}] {}",
                    humantime::format_rfc3339_seconds(SystemTime::now()),
                    record.level(),
                    record.target(),
                    message
                ));
            })
            .level(level)
            .chain(std::io::stdout())
            .chain(fern::log_file(log_folder).expect("Error creating fern log file"))
            .apply()
            .expect("Error applying fern dispatch");
        info!("Logger startup completed");
    }

}

