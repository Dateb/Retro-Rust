use retro_rust::environments::image_retro_env::ImageRetroEnv;
use retro_rust::environments::image_retro_env::platform::Platform;

#[test]
fn create_image_env() {
    let game_name = "Airstriker";
    let platform = Platform::Genesis;
    let save_state_name = String::from("Level1.state");

    ImageRetroEnv::new(game_name, platform, save_state_name);
}