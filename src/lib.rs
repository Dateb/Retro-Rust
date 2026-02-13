use crate::environments::image_retro_env::ImageRetroEnv;
use crate::environments::image_retro_env::platform::Platform;
use crate::traits::retro_env::RetroEnv;

pub mod environments;
pub mod traits;

pub fn main() {
    let game_name = "Airstriker";
    let platform = Platform::Genesis;
    let save_state_name = String::from("Level1.state");

    let mut env = ImageRetroEnv::new(game_name, platform, save_state_name);

    // Initialise your policy
    let policy = |obs: Vec<f32>| -> usize { 0 };

    let num_episodes = 100;
    for _ in 1..num_episodes {
        let mut step_info = env.reset();
        let mut next_image = step_info.observation;
        let mut next_action = policy(next_image);

        while !step_info.is_done {
            step_info = env.step(next_action);

            next_image = step_info.observation;
            let reward = step_info.reward;
            let done = step_info.is_done;

            // Use environment feedback to make a new decision
            next_action = policy(next_image);
        }
    }
}