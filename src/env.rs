mod emulator;
mod gamedata;
mod gamestate;

use std::path::Path;
use burn::tensor::{Float, Tensor, TensorData};
use burn::backend::Wgpu;

type Backend = Wgpu;

pub struct RetroEnv {
    emu: emulator::RustRetroEmulator,
    data: gamedata::RustRetroGameData,
    valid_action_keys: Vec<i32>,
    frame_skip: usize
}

impl RetroEnv {
    pub fn new(game_path: String) -> Self {
        let game_state_path = game_path.clone() + "/1Player.Level1.state";
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

        let valid_action_keys = data.get_valid_action_keys();

        RetroEnv { emu, data, valid_action_keys, frame_skip: 4 }
    }


    pub fn reset(&self) -> Tensor<Wgpu, 3> {
        let episode_reward = self.data.total_reward();
        self.emu.set_start_state();
        self.data.reset();
        self.data.update_ram();
        self.emu.step();

        self.screen_to_tensor()
    }

    pub fn valid_action_keys(&self) -> Vec<i32> {
        self.data.get_valid_action_keys()
    }

    pub fn step(&self, action_index: usize) -> (Tensor<Wgpu, 3>, f32, bool) {
        let mut action = self.valid_action_keys[action_index] as usize;
        let button_mask = RetroEnv::action_to_button_mask(action, 12);
        self.emu.set_button_mask(button_mask.as_slice(), 0);
        for _ in 0..self.frame_skip {
            self.emu.step();
            self.data.update_ram();
        }

        (self.screen_to_tensor(), self.data.current_reward(), self.is_done())
    }

    pub fn action_to_button_mask(mut action: usize, n: usize) -> Vec<u8> {
        assert!(action < (1usize << n), "value out of range for {} bits", n);

        let mut button_mask = Vec::with_capacity(n);

        for _ in 0..n {
            button_mask.push((action & 1) as u8);
            action >>= 1;
        }

        button_mask
    }

    pub fn is_done(&self) -> bool {
        self.data.is_done()
    }

    fn screen_to_tensor(&self) -> Tensor<Wgpu, 3> {
        let (buffer, w, h) = self
            .emu
            .get_screen()
            .expect("Screen not available");

        let device = Default::default();

        // Convert u8 -> f32 and normalize
        let buffer_f64: Vec<f64> = buffer.iter().map(|&x| x as f64 / 255.0).collect();

        let tensor_data = TensorData::new(buffer_f64, [w as usize, h as usize, 3]);
        let tensor: Tensor<Backend, 3, Float> = Tensor::<Backend, 3, Float>::from_data(tensor_data, &device);

        tensor
    }

    pub fn episode_reward(&self) -> f32 {
        self.data.total_reward()
    }

    pub fn num_actions(&self) -> usize {
        self.valid_action_keys.len()
    }

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