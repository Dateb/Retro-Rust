use std::{env, io};
use std::io::{BufRead, Stdout, Write};
use retro_rust::environments::image_retro_env::{emulator, ImageRetroEnv};
use retro_rust::environments::image_retro_env::platform::Platform;
use retro_rust::environments::retro_env::{build_env, RetroEnvScenario};
use retro_rust::traits::retro_env::RetroEnv;

fn process_action_command(cmd: &str, env: &mut Box<dyn RetroEnv>, out: &mut Stdout) {
    let action = cmd.parse().expect("Invalid action");
    let mut step_info = env.step(action);

    if step_info.is_done { 
        let step_info_reset = env.reset();
        step_info.observation = step_info_reset.observation;
    }

    // ---- Write observation raw bytes ----
    let obs_bytes = bytemuck::cast_slice::<f32, u8>(&step_info.observation);
    out.write_all(obs_bytes).unwrap();

    // ---- Write reward ----
    out.write_all(&step_info.reward.to_le_bytes()).unwrap();

    // ---- Write done flag ----
    out.write_all(&[step_info.is_done as u8]).unwrap();

    out.flush().unwrap();
}

pub fn run_worker(game: &str, platform: &str, save_state: &str) {
    let scenario = RetroEnvScenario::new(
        "Airstriker",
        Platform::Genesis,
        "Level1.state",
        false
    );

    let mut env = build_env(scenario);

    let stdin = io::stdin();
    let mut out = io::stdout();
    let lines = stdin.lock().lines();

    for line in lines {
        let line = line.expect("Failed to read line");
        let cmd = line.trim();

        match cmd {
            "CLOSE" => break,
            _ => process_action_command(cmd, &mut env, &mut out)
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        eprintln!("Usage: worker <game> <platform> <save_state>");
        std::process::exit(1);
    }

    let game = &args[1];
    let platform = &args[2];
    let save_state = &args[3];

    run_worker(game, platform, save_state);
}
