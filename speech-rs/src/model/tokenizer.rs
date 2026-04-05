use std::collections::HashMap;
use std::fs::File;
use std::io::{ BufRead, BufReader };
use std::path::Path;
use log::warn;


pub struct Tokenizer {
    pub inverse_vocab: HashMap<usize, String>,
    pub vocab: HashMap<String, usize>,
}


impl Tokenizer {
    pub fn load_vocab(path: &Path) -> Self {
        let file = File::open(path).expect("eerror creatring vocab path dfile");
        let reader = BufReader::new(file);
        let mut inverse_vocab = HashMap::new();
        let mut vocab = HashMap::new();
        for line in reader.lines() {
            let line = line.expect("Error getting vocab line");
            if line.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                let key = parts[0].to_string();
                // Parse the last element as the ID
                if let Ok(id) = parts[parts.len() - 1].parse::<usize>() {
                    inverse_vocab.insert(id, key.clone());
                    vocab.insert(key, id);
                }
            }
            else {
                warn!("PARTS len is different than 2: {:?}", parts);
            }
        }
        Self { inverse_vocab, vocab}
    }

    pub fn tokenize(&self, token: &String) -> usize {
        *self.vocab.get(token).expect("Error gettign token")
    }

    pub fn detokenize(&self, token: &usize) -> String {
        self.inverse_vocab.get(token).expect("Error gettign token").replace("\u{2581}", " ")
    }
}


