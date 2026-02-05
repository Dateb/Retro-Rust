pub(crate) mod emulator;
mod gamedata;
mod gamestate;
mod frame_stack;
mod controller;

use std::path::{Path, PathBuf};
use image::{imageops::resize, imageops::FilterType, ImageBuffer, Luma, RgbImage};
use crate::environments::image_retro_env::controller::Controller;
use crate::environments::image_retro_env::emulator::RustRetroEmulator;
use crate::environments::image_retro_env::frame_stack::FrameStack;
use crate::environments::image_retro_env::gamedata::RustRetroGameData;
use crate::environments::image_retro_env::gamestate::GameState;
use crate::traits::retro_env::{RetroEnv, StepInfo};

pub struct ImageRetroEnv {
    pub emu: RustRetroEmulator,
    data: RustRetroGameData,
    controller: Controller,
    frame_stack: FrameStack,
    pub frame_skip: u8,
}

impl ImageRetroEnv {
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

        ImageRetroEnv { emu, data, controller, frame_stack, frame_skip }
    }

    pub fn skipped_frame_step(&self, button_bit_mask: &Vec<u8>) -> f32 {
        self.emu.set_button_mask(button_bit_mask.as_slice(), 0);
        self.emu.step();
        self.data.update_ram();

        self.data.current_reward()
    }

    pub fn step_current_frame(&mut self, reward: f32) -> StepInfo {
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

        ImageRetroEnv::preprocess_screen(buffer, w, h)
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

    pub fn get_button_bitmask(&self, action: usize) -> &Vec<u8> {
        self.controller.get_button_bitmask(action)
    }
}

impl RetroEnv for ImageRetroEnv {
    fn step(&mut self, action: usize) -> StepInfo {
        let button_bit_mask = self.get_button_bitmask(action);

        let mut reward = 0.0;
        for _ in 0..self.frame_skip {
            reward += self.skipped_frame_step(button_bit_mask)
        }

        self.step_current_frame(reward)
    }

    fn reset(&mut self) -> StepInfo {
        self.emu.set_start_state();

        self.emu.step();
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

    fn num_actions(&self) -> usize { self.controller.num_actions }
}
