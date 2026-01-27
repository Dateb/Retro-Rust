use std::ffi::CString;
use std::path::Path;

#[repr(C)]
pub struct RetroGameData;

#[repr(C)]
struct RetroActionSet {
    key: i32,
    values: *mut i32,
    num_values: usize,
}

unsafe extern "C" {
    fn gamedata_new() -> *mut RetroGameData;
    fn gamedata_load(
        gamedata: *mut RetroGameData,
        data_path: *const std::os::raw::c_char,
        scenario_path: *const std::os::raw::c_char
    ) -> bool;
    fn gamedata_reset(gamedata: *mut RetroGameData);
    fn gamedata_valid_actions(
        gamedata: *mut RetroGameData,
        num_entries: *mut usize
    ) -> *mut RetroActionSet;
    fn gamedata_free_valid_actions(
        actions: *mut RetroActionSet,
        num_entries: usize,
    );
    fn gamedata_update_ram(gamedata: *mut RetroGameData);
    fn gamedata_current_reward(gamedata: *mut RetroGameData) -> f32;
    fn gamedata_total_reward(gamedata: *mut RetroGameData) -> f32;
    fn gamedata_is_done(gamedata: *mut RetroGameData) -> bool;
}

#[derive(Debug)]
pub struct RustRetroGameData {
    pub retro_data: *mut RetroGameData
}

impl RustRetroGameData {
    pub fn new(game_path: String) -> Self {
        unsafe {
            let retro_data = gamedata_new();
            let data_path = game_path.clone() + "/data.json";
            let data_path = Path::new(&data_path)
                .canonicalize()
                .expect("Data path not found");

            let data_path = CString::new(data_path.to_str().unwrap()).expect("CString::new failed");

            let scenario_path = game_path.clone() + "/scenario.json";
            let scenario_path = Path::new(&scenario_path)
                .canonicalize()
                .expect("Scenario path not found");

            let scenario_path = CString::new(scenario_path.to_str().unwrap()).expect("CString::new failed");

            if !gamedata_load(retro_data, data_path.as_ptr(), scenario_path.as_ptr()) {
                panic!("Failed to load game data");
            }

            RustRetroGameData {
                retro_data
            }
        }
    }

    pub fn get_button_combos(&self) -> Vec<Vec<u64>> {
        unsafe {
            let mut n: usize = 0;
            let ptr = gamedata_valid_actions(self.retro_data, &mut n);

            let mut result = Vec::with_capacity(n);

            for i in 0..n {
                let entry = &*ptr.add(i);

                // Copy values from raw pointer into Rust Vec
                let values: Vec<u64> = std::slice::from_raw_parts(
                    entry.values,
                    entry.num_values,
                )
                    .iter()
                    .map(|&v| v as u64)
                    .collect();

                result.push(values);
            }

            gamedata_free_valid_actions(ptr, n);
            result
        }
    }

    pub fn reset(&self) {
        unsafe {
            gamedata_reset(self.retro_data);
        }
    }

    pub fn update_ram(&self) {
        unsafe {
            gamedata_update_ram(self.retro_data);
        }
    }

    pub fn current_reward(&self) -> f32 {
        unsafe {
            gamedata_current_reward(self.retro_data)
        }
    }

    pub fn total_reward(&self) -> f32 {
        unsafe {
            gamedata_total_reward(self.retro_data)
        }
    }

    pub fn is_done(&self) -> bool {
        unsafe {
            gamedata_is_done(self.retro_data)
        }
    }
}