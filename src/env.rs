mod emulator;
mod gamedata;
mod gamestate;
mod frame_stack;
mod controller;

use std::path::Path;
use burn::prelude::Backend;
use image::{ImageBuffer, RgbImage, imageops::resize, imageops::FilterType, Luma};
use crate::env::controller::Controller;
use crate::env::frame_stack::FrameStack;

pub struct RetroEnv {
    emu: emulator::RustRetroEmulator,
    data: gamedata::RustRetroGameData,
    controller: Controller,
    frame_stack: FrameStack,
    frame_skip: usize,
}

impl RetroEnv {
    pub fn new(game_path: String) -> Self {
        let game_state_path = game_path.clone() + "/Level1.state";
        let start_game_state = gamestate::GameState::new(&game_state_path).expect("Failed to load state");

        let emu = emulator::RustRetroEmulator::new(start_game_state);
        let rom_path = game_path.clone() + "/rom.md";

        let rom_path = Path::new(&rom_path)
            .canonicalize()
            .expect("ROM path not found");

        if !emu.load_rom(rom_path.to_str().unwrap()) {
            panic!("Failed to load ROM");
        }

        let data = gamedata::RustRetroGameData::new(game_path);
        emu.configure_data(&data);

        let controller = Controller::new(data.get_button_combos());

        let frame_stack = FrameStack::new(84 * 84);

        RetroEnv { emu, data, controller, frame_stack, frame_skip: 4 }
    }


    pub fn reset(&mut self) -> Vec<f32> {
        self.emu.set_start_state();
        self.data.reset();
        self.data.update_ram();
        self.emu.step();

        self.frame_stack.clear();
        let frame = self.get_screen_buffer();
        self.frame_stack.push(frame);

        self.frame_stack.stacked()
    }

    pub fn step(&mut self, action: usize) -> (Vec<f32>, f32, bool) {
        self.emu.set_button_mask(self.controller.get_button_bitmask(action).as_slice(), 0);

        let mut reward = 0.0;
        for _ in 0..self.frame_skip {
            self.emu.step();
            self.data.update_ram();
            reward += self.data.current_reward();
        }

        let frame = self.get_screen_buffer();
        self.frame_stack.push(frame);

        (self.frame_stack.stacked(), reward, self.is_done())
    }

    pub fn is_done(&self) -> bool {
        self.data.is_done()
    }

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

    pub fn print_screen(&self) {
        if let Some((pixels, width, height)) = self.emu.get_screen() {
            println!("Got frame {}x{}", width, height);

            for b in &pixels {
                print!("{:02X} ", b);
            }
            println!();
        } else {
            println!("No frame available");
        }
    }
}
