use crate::environments::image_retro_env::ImageRetroEnv;
use crate::environments::image_retro_env::platform::Platform;
use crate::traits::retro_env::RetroEnv;

pub mod environments;
pub mod traits;

pub fn main() {
    // A training scenario is defined by a (game, platform, save_state) triple
    let game_name = "Airstriker";
    let platform = Platform::Genesis;
    let save_state_name = String::from("Level1.state");

    let mut env = ImageRetroEnv::new(game_name, platform, save_state_name);

    // Use this function signature for your policy
    let policy = |_obs: Vec<f32>| -> usize { 0 };

    let num_episodes = 100;
    for _ in 1..num_episodes {
        let mut step_info = env.reset();
        let mut next_image = step_info.observation;
        let mut next_action = policy(next_image);

        let mut episode_reward = step_info.reward;

        // Action-Feedback loop for one episode
        while !step_info.is_done {
            step_info = env.step(next_action);

            next_image = step_info.observation;
            episode_reward += step_info.reward;

            next_action = policy(next_image);
        }
        println!("Episode reward: {}", episode_reward);
    }
}