use std::process::{Child, Command, Stdio};
use std::io::{BufRead, BufReader, Write};
use rand::Rng;
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
//     let mut any_non_zero_frame = false;
//     let mut any_non_zero_reward = false;
// 
//     for _ in 0..500 {
//         // generate a random action between 0 and 125
//         let action = rand::thread_rng().gen_range(0..126);
// 
//         let step_infos = image_env.step(action);
// 
//         // check first frame for non-zero pixel
//         let frame = &step_infos.observation;
//         if frame.iter().any(|&pixel| pixel != 0.0) {
//             any_non_zero_frame = true;
//         }
// 
//         // check first reward for non-zero
//         let reward = step_infos.reward;
//         if reward != 0.0 {
//             any_non_zero_reward = true;
//         }
// 
//         // stop early if both conditions satisfied
//         if any_non_zero_frame && any_non_zero_reward {
//             break;
//         }
//     }
// 
//     assert!(any_non_zero_frame, "All of the first 500 frames were completely zero");
//     assert!(any_non_zero_reward, "All of the first 500 rewards were zero");
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

    let mut any_non_zero_frame = false;
    let mut any_non_zero_reward = false;

    for _ in 0..500 {
        // generate a random action between 0 and 125
        let action = rand::thread_rng().gen_range(0..126);
        let actions = vec![action, action];

        let step_infos = vector_env.step(&actions);

        // check first frame for non-zero pixel
        let frame = &step_infos[0].observation;
        if frame.iter().any(|&pixel| pixel != 0.0) {
            any_non_zero_frame = true;
        }

        // check first reward for non-zero
        let reward = step_infos[0].reward;
        if reward != 0.0 {
            any_non_zero_reward = true;
        }

        // stop early if both conditions satisfied
        if any_non_zero_frame && any_non_zero_reward {
            break;
        }
    }

    assert!(any_non_zero_frame, "All of the first 500 frames were completely zero");
    assert!(any_non_zero_reward, "All of the first 500 rewards were zero");
}
