pub mod emulator;
mod gamedata;
mod gamestate;
mod frame_stack;
mod controller;
pub mod platform;

use std::path::PathBuf;
use image::{imageops::resize, imageops::FilterType, ImageBuffer, Luma, RgbImage};
use crate::environments::image_retro_env::controller::Controller;
use crate::environments::image_retro_env::emulator::RustRetroEmulator;
use crate::environments::image_retro_env::frame_stack::FrameStack;
use crate::environments::image_retro_env::gamedata::RustRetroGameData;
use crate::environments::image_retro_env::gamestate::GameState;
use crate::environments::image_retro_env::platform::Platform;
use crate::traits::retro_env::{RetroEnv, StepInfo};

pub struct ImageRetroEnv {
    pub game_name: String,
    pub emu: RustRetroEmulator,
    data: RustRetroGameData,
    controller: Controller,
    frame_stack: FrameStack,
    pub frame_skip: u8,
}

impl ImageRetroEnv {
    pub fn new(game_name: &str, platform: Platform, save_state_name: String) -> Self {
        let mut game_path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("games");

        let platform_name = platform.as_str();
        let game_dir = format!("{game_name}-{platform_name}");
        game_path.push(game_dir);

        println!("Starting environment setup...");
        println!("{}", "-".repeat(30));
        let start_save_state = Self::create_save_state(&game_path, save_state_name);
        println!("✔ Save state verified");

        let emu = RustRetroEmulator::new(&platform, start_save_state);
        println!("✔ Emulator verified");

        let mut rom_path = game_path.clone();
        rom_path.push(platform.rom_name());

        let rom_path = rom_path
            .to_string_lossy()
            .to_string();

        if !emu.load_rom(&rom_path) {
            panic!("Failed to load ROM");
        }
        println!("✔ Rom verified");

        let data = RustRetroGameData::new(game_path.to_string_lossy().to_string());
        emu.configure_data(&data);

        let controller = Controller::new(data.get_button_combos());
        let frame_stack = FrameStack::new(84 * 84);

        println!("{}", "-".repeat(30));
        println!("Environment is ready to run!");
        ImageRetroEnv {
            game_name: game_name.to_string(),
            emu,
            data,
            controller,
            frame_stack,
            frame_skip: 4
        }
    }

    fn create_save_state(game_path: &PathBuf, save_state_name: String) -> GameState {
        let game_state_path = game_path
            .join(save_state_name)
            .to_string_lossy()
            .to_string();

        GameState::new(&game_state_path).expect("Failed to load state")
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
