use std::process::{Child, Command, Stdio};
use std::io::{BufRead, BufReader, Write};
use retro_rust::environments::image_retro_env::ImageRetroEnv;
use retro_rust::environments::image_retro_env::platform::Platform;
use retro_rust::environments::movie_retro_env::MovieRetroEnv;
use retro_rust::environments::vector_retro_env::VectorRetroEnv;
use retro_rust::traits::retro_env::{RetroEnv, StepInfo};

// #[test]
// fn create_image_env() {
//     let game_name = "Airstriker";
//     let platform = Platform::Genesis;
//     let save_state_name = String::from("Level1.state");
//
//     let mut image_env = ImageRetroEnv::new(game_name, platform, save_state_name);
//
//     image_env.reset();
//
//     let action = 0;
//     image_env.step(action);
//
//     image_env.reset();
// }

// #[test]
// fn create_movie_env() {
//     let game_name = "Airstriker";
//     let platform = Platform::Genesis;
//     let save_state_name = String::from("Level1.state");
//
//     let image_env = ImageRetroEnv::new(game_name, platform, save_state_name);
//     let mut movie_env = MovieRetroEnv::new(image_env);
//
//     movie_env.reset();
//
//     let action = 0;
//     movie_env.step(action);
//
//     movie_env.reset();
// }

#[test]
fn run_vector_env() {
    let mut vector_env = VectorRetroEnv::new(2);
    
    let actions = vec![0, 1];
    vector_env.step(actions);
}
