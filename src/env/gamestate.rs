use std::fs::File;
use std::io;
use std::io::{Read, BufReader};
use flate2::read::GzDecoder;

#[derive(Debug)]
pub struct GameState {
    pub buffer: Vec<u8>
}

impl GameState {
    pub fn new(path: &str) -> io::Result<Self> {
        // Open the file
        let file = File::open(path)?;

        // Wrap in a GzDecoder
        let mut gz = GzDecoder::new(BufReader::new(file));

        // Read all bytes
        let mut buffer = Vec::new();
        gz.read_to_end(&mut buffer)?;

        Ok(GameState { buffer })
    }
}