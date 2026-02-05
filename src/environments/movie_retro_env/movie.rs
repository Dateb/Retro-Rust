use std::ffi::CString;
use std::os::raw::c_char;
use crate::environments::image_retro_env::emulator::{RetroEmulator, RustRetroEmulator};

#[repr(C)]
pub struct RetroMovie {
    _unused: [u8; 0]
}

unsafe extern "C" {
    fn movie_new(name: *const c_char) -> *mut RetroMovie;
    fn movie_close(movie: *mut RetroMovie);
    fn movie_step(movie: *mut RetroMovie) -> bool;
    fn movie_set_key(movie: *mut RetroMovie, key: usize, set: bool);
    fn movie_configure(movie: *mut RetroMovie, emulator: *mut RetroEmulator, name: *const c_char);
    fn movie_set_state(movie: *mut RetroMovie, data: *const u8, size: usize);
}

#[derive(Debug)]
pub struct RustRetroMovie {
    pub retro_emulator: *mut RetroEmulator,
    pub retro_movie: *mut RetroMovie,
}

impl RustRetroMovie {
    pub fn new(emulator: *mut RustRetroEmulator, movie_name: String, game_name: String) -> Self {
        unsafe {
            let movie_name = CString::new(movie_name).expect("CString::new failed");
            let retro_movie = movie_new(movie_name.as_ptr());

            let game_name = CString::new(game_name).expect("CString::new failed");
            movie_configure(retro_movie, (*emulator).retro_emulator, game_name.as_ptr());
            movie_set_state(retro_movie, (*emulator).start_game_state.buffer.as_ptr(), (*emulator).start_game_state.buffer.len());

            RustRetroMovie {
                retro_emulator: (*emulator).retro_emulator,
                retro_movie
            }
        }
    }

    pub fn set_key(&self, key: usize, set: bool) {
        unsafe {
            movie_set_key(self.retro_movie, key, set);
        }
    }

    pub fn close(&self) {
        unsafe {
            movie_close(self.retro_movie)
        }
    }

    pub fn step(&self) {
        unsafe {
            movie_step(self.retro_movie);
        }
    }
}