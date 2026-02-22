use std::fs;
use std::path::Path;
use crate::environments::image_retro_env::ImageRetroEnv;
use crate::environments::movie_retro_env::movie::RustRetroMovie;
use crate::traits::retro_env::{RetroEnv, StepInfo};

pub mod movie;

pub struct MovieRetroEnv {
    image_env: ImageRetroEnv,
    movie: RustRetroMovie,
    movies_dir: &'static str,
    movie_counter: u16
}

impl MovieRetroEnv {
    pub fn new(mut image_env: ImageRetroEnv) -> Self {
        let movies_dir = "movies";
        let movies_path = Path::new(movies_dir);
        if !movies_path.exists() { fs::create_dir(movies_path).expect("Cannot create movies dir") }

        let movie_counter = 1u16;

        let movie_path = format!("{}/movie_{}.bk2", movies_dir, movie_counter);
        
        unsafe {
            let movie = RustRetroMovie::new(
                &mut image_env.emu,
                movie_path,
                format!("{}-Genesis", image_env.game_name.clone())
            );

            Self { image_env, movie, movies_dir, movie_counter }
        }
    }

    fn next_movie_path(&mut self) -> String {
        self.movie_counter += 1;
        format!("{}/movie_{}.bk2", self.movies_dir, self.movie_counter)
    }
}

impl RetroEnv for MovieRetroEnv {
    fn step(&mut self, action: usize) -> StepInfo {
        let button_bit_mask = self.image_env.get_button_bitmask(action);

        let mut reward = 0.0;
        for _ in 0..self.image_env.frame_skip {
            for (idx, value) in button_bit_mask.iter().enumerate() {
                self.movie.set_key(idx, *value == 1);
            }
            self.movie.step();
            reward += self.image_env.skipped_frame_step(button_bit_mask)
        }

        self.image_env.step_current_frame(reward)
    }

    fn reset(&mut self) -> StepInfo {
        let step_info = self.image_env.reset();

        self.movie.close();
        
        unsafe {
            self.movie = RustRetroMovie::new(
                &mut self.image_env.emu,
                self.next_movie_path(),
                format!("{}-Genesis", self.image_env.game_name.clone())
            );
        }
        
        self.movie.step();

        step_info
    }

    fn num_actions(&self) -> usize {
        self.image_env.num_actions()
    }
}