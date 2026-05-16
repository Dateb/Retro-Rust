use crate::environments::image_retro_env::ImageRetroEnv;
use crate::traits::retro_env::RetroEnv;
use crate::environments::image_retro_env::platform::Platform;
use crate::environments::movie_retro_env::MovieRetroEnv;

#[derive(Clone)]
pub struct RetroEnvScenario<'a> {
    pub(crate) game_name: &'a str,
    pub(crate) platform: Platform,
    pub(crate) save_state_name: &'a str,
    pub(crate) record_movie: bool,
}

impl<'a> RetroEnvScenario<'a> {
    pub fn new(game_name: &'a str, platform: Platform, save_state_name: &'a str, record_movie: bool) -> Self {
        RetroEnvScenario {
            game_name,
            platform,
            save_state_name,
            record_movie,
        }
    }
}

pub fn build_env(scenario: RetroEnvScenario) -> Box<dyn RetroEnv> {
    let mut env: Box<dyn RetroEnv> = match scenario.record_movie {
        false => {
            Box::new(
                ImageRetroEnv::new(
                    scenario.game_name,
                    scenario.platform,
                    scenario.save_state_name,
                )
            )
        },
        true => {
            let env: Box<ImageRetroEnv> = Box::new(
                ImageRetroEnv::new(
                    scenario.game_name,
                    scenario.platform,
                    scenario.save_state_name,
                )
            );

            Box::new(MovieRetroEnv::new(env))
        }
    };

    env.reset();

    env
}

