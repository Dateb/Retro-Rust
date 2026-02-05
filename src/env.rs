mod emulator;
mod gamedata;
mod gamestate;
mod frame_stack;
mod controller;
mod movie;

use std::path::{Path, PathBuf};
use image::{ImageBuffer, RgbImage, imageops::resize, imageops::FilterType, Luma};
use crate::env::controller::Controller;
use crate::env::emulator::RustRetroEmulator;
use crate::env::frame_stack::FrameStack;
use crate::env::gamedata::RustRetroGameData;
use crate::env::gamestate::GameState;
use crate::env::movie::RustRetroMovie;

pub struct StepInfo {
    pub observation: Vec<f32>,
    pub reward: f32,
    pub is_done: bool
}

pub struct RetroEnv {
    emu: RustRetroEmulator,
    data: RustRetroGameData,
    movie: RustRetroMovie,
    controller: Controller,
    frame_stack: FrameStack,
    frame_skip: u8,
}

impl RetroEnv {
    pub fn new(game_path: String, save_state_name: &str, frame_skip: u8) -> Self {
        let game_state_path = PathBuf::from(&game_path)
            .join(save_state_name)
            .to_string_lossy()
            .to_string();

        let start_game_state = GameState::new(&game_state_path)
            .expect("Failed to load state");

        let mut emu = RustRetroEmulator::new(start_game_state);
        let rom_path = game_path.clone() + "/rom.md";

        let rom_path = Path::new(&rom_path)
            .canonicalize()
            .expect("ROM path not found");

        if !emu.load_rom(rom_path.to_str().unwrap()) {
            panic!("Failed to load ROM");
        }

        let data = RustRetroGameData::new(game_path);
        emu.configure_data(&data);

        let controller = Controller::new(data.get_button_combos());

        let frame_stack = FrameStack::new(84 * 84);

        let movie = RustRetroMovie::new(
            &mut emu,
            String::from("yellow.bk2"),
            String::from("Airstriker-Genesis")
        );
        RetroEnv { emu, data, movie, controller, frame_stack, frame_skip }
    }


    pub fn reset(&mut self) -> StepInfo {
        self.emu.set_start_state();

        self.movie.close();
        self.movie = RustRetroMovie::new(
            &mut self.emu,
            String::from("yellow.bk2"),
            String::from("Airstriker-Genesis")
        );

        self.emu.step();
        self.movie.step();
        self.data.reset();
        self.data.update_ram();

        self.frame_stack.clear();
        let frame = self.get_screen_buffer();
        self.frame_stack.push(frame);

        StepInfo {
            observation: self.frame_stack.stacked(),
            reward: self.data.current_reward(),
            is_done: self.is_done()
        }
    }

    pub fn step(&mut self, action: usize) -> StepInfo {
        let button_bit_mask = self.controller.get_button_bitmask(action);

        let mut reward = 0.0;
        for _ in 0..self.frame_skip {
            for (idx, value) in button_bit_mask.iter().enumerate() {
                self.movie.set_key(idx, *value == 1);
            }

            self.emu.set_button_mask(button_bit_mask.as_slice(), 0);

            self.movie.step();
            self.emu.step();
            self.data.update_ram();
            reward += self.data.current_reward();
        }

        let frame = self.get_screen_buffer();
        self.frame_stack.push(frame);

        StepInfo {
            observation: self.frame_stack.stacked(),
            reward,
            is_done: self.is_done(),
        }
    }

    fn is_done(&self) -> bool { self.data.is_done() }

    fn get_screen_buffer(&self) -> Vec<f32> {
        let (buffer, w, h) = self
            .emu
            .get_screen()
            .expect("Screen not available");

        RetroEnv::preprocess_screen(buffer, w, h)
    }

    fn preprocess_screen(buffer: Vec<u8>, w: i32, h: i32) -> Vec<f32> {
        // 1. Convert buffer -> ImageBuffer
        let img: RgbImage = ImageBuffer::from_raw(w as u32, h as u32, buffer)
            .expect("Failed to convert screen buffer to image");

        // 2. Resize to smaller dimensions, e.g., 84x84
        let resized = resize(&img, 84, 84, FilterType::Nearest);

        // 3. Optional: convert to grayscale
        let gray: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_fn(resized.width(), resized.height(), |x, y| {
            let pixel = resized.get_pixel(x, y);
            // Standard grayscale: 0.299 R + 0.587 G + 0.114 B
            let gray_val = (0.299 * pixel[0] as f32
                + 0.587 * pixel[1] as f32
                + 0.114 * pixel[2] as f32) as u8;
            Luma([gray_val])
        });

        // 4. Flatten to Vec<f32> and normalize
        gray.pixels()
            .map(|p| p[0] as f32 / 255.0)
            .collect()
    }

    pub fn episode_reward(&self) -> f32 {
        self.data.total_reward()
    }

    pub fn num_actions(&self) -> usize { self.controller.num_actions }
}
