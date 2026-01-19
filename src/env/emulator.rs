use std::ffi::CString;
use std::fs;
use crate::env::gamedata::RetroGameData;
use crate::env::gamedata::RustRetroGameData;
use crate::env::gamestate::GameState;

#[repr(C)]
pub struct RetroEmulator;

unsafe extern "C" {
    fn emulator_new() -> *mut RetroEmulator;
    fn emulator_configure_data(
        emulator: *mut RetroEmulator,
        data: *mut RetroGameData,
    );
    fn emulator_load_rom(emulator: *mut RetroEmulator, rom_path: *const std::os::raw::c_char) -> bool;
    fn emulator_run(emulator: *mut RetroEmulator);
    fn emulator_set_state(emulator: *mut RetroEmulator, state_data: *const u8, size: usize) -> bool;
    fn emulator_get_screen_width(emulator: *mut RetroEmulator) -> i32;
    fn emulator_get_screen_height(emulator: *mut RetroEmulator) -> i32;
    fn emulator_get_screen(
        emulator: *mut RetroEmulator,
        width: i32,
        height: i32,
        out_rgb: *mut u8,
    ) -> bool;
    fn emulator_set_button_mask(
        emulator: *mut RetroEmulator,
        mask: *const u8,
        num_buttons: usize,
        player: u32,
    );
    fn load_core_info(json: *const std::os::raw::c_char) -> bool;
}

#[derive(Debug)]
pub struct RustRetroEmulator {
    retro_emulator: *mut RetroEmulator,
    start_game_state: GameState
}

impl RustRetroEmulator {
    pub fn new(start_game_state: GameState) -> Self {
        let json_str = fs::read_to_string("cores/genesis.json").unwrap();
        let json_str_c = CString::new(json_str).expect("CString::new failed");
        unsafe {
            load_core_info(json_str_c.as_ptr());
            let retro_emulator = emulator_new();
            RustRetroEmulator { retro_emulator, start_game_state }
        }
    }
    pub fn configure_data(&self, data: &RustRetroGameData) {
        unsafe {
            emulator_configure_data(self.retro_emulator, data.retro_data);
        }
    }
    pub fn step(&self) {
        unsafe {
            emulator_run(self.retro_emulator)
        }
    }
    pub fn get_screen(&self) -> Option<(Vec<u8>, i32, i32)> {
        unsafe {
            let mut w = emulator_get_screen_width(self.retro_emulator);
            let mut h = emulator_get_screen_height(self.retro_emulator);

            let mut buffer = vec![0u8; (w * h * 3) as usize];

            let ok = emulator_get_screen(self.retro_emulator, w, h, buffer.as_mut_ptr());
            ok.then(|| (buffer, w, h))
        }
    }
    pub fn set_button_mask(&self, mask: &[u8], player: u32) {
        unsafe {
            emulator_set_button_mask(
                self.retro_emulator,
                mask.as_ptr(),
                mask.len(),
                player,
            );
        }
    }
    pub fn load_rom(&self, rom_path: &str) -> bool {
        let c_path = CString::new(rom_path).expect("CString::new failed");
        unsafe {
            emulator_load_rom(self.retro_emulator, c_path.as_ptr())
        }
    }
    pub fn set_start_state(&self) -> bool {
        unsafe {
            emulator_set_state(
                self.retro_emulator,
                self.start_game_state.buffer.as_ptr(),
                self.start_game_state.buffer.len()
            )
        }
    }
}