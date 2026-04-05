use crate::config::types::{GlobalConfig};
use std::fs;


impl GlobalConfig {
    pub fn from_file(file_path: &str) -> Self {
        let contents = fs::read_to_string(file_path).expect("err read file config");
        let config: GlobalConfig = toml::from_str(&contents).expect("error generating cfg struct");
        config
    }
}