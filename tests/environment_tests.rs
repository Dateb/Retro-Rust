use retro_rust::environments::image_retro_env::ImageRetroEnv;
use retro_rust::environments::image_retro_env::platform::Platform;
use retro_rust::environments::movie_retro_env::MovieRetroEnv;
use retro_rust::traits::retro_env::RetroEnv;

#[test]
fn create_env() {
    let game_name = "Airstriker";
    let platform = Platform::Genesis;
    let save_state_name = String::from("Level1.state");

    let image_env = ImageRetroEnv::new(game_name, platform, save_state_name);
    let mut movie_env = MovieRetroEnv::new(image_env);

    movie_env.reset();

    let action = 0;
    movie_env.step(action);

    movie_env.reset();
}