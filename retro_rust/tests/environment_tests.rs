use std::process::{Child, Command, Stdio};
use std::io::{BufRead, BufReader, Write};
use rand::Rng;
use retro_rust::environments::image_retro_env::ImageRetroEnv;
use retro_rust::environments::image_retro_env::platform::Platform;
use retro_rust::environments::movie_retro_env::MovieRetroEnv;
use retro_rust::environments::retro_env::{build_env, RetroEnvScenario};
use retro_rust::environments::vector_retro_env::VectorRetroEnv;
use retro_rust::traits::retro_env::{RetroEnv, StepInfo};

// #[test]
// fn create_image_env() {
//     let scenario = RetroEnvScenario::new(
//         "Airstriker",
//         Platform::Genesis,
//         "Level1.state",
//         false
//     );
//
//     let mut image_env = build_env(scenario);
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
//     let scenario = RetroEnvScenario::new(
//         "Airstriker",
//         Platform::Genesis,
//         "Level1.state",
//         true
//     );
//
//     let mut movie_env = build_env(scenario);
//
//     let action = 0;
//     movie_env.step(action);
//
//     movie_env.reset();
// }

#[test]
fn run_vector_env() {
    let movie_scenario = RetroEnvScenario::new(
        "Airstriker",
        Platform::Genesis,
        "Level1.state",
        true
    );

    let image_scenario = RetroEnvScenario::new(
        "Airstriker",
        Platform::Genesis,
        "Level1.state",
        false
    );

    let mut scenarios = Vec::with_capacity(5);

    scenarios.push(movie_scenario);
    scenarios.extend(std::iter::repeat(image_scenario).take(4));

    let num_envs = scenarios.len();

    let mut vector_env = VectorRetroEnv::new(scenarios);

    let mut any_non_zero_frame = false;
    let mut any_non_zero_reward = false;

    for _ in 0..500 {
        // generate a random action between 0 and 125
        let action = rand::thread_rng().gen_range(0..126);
        let actions = vec![action; num_envs];

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
