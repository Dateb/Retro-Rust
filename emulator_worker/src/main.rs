use std::{env, io};
use std::io::{BufRead, Stdout, Write};
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

    let obs_bytes = bytemuck::cast_slice::<f32, u8>(&step_info.observation);
    let reward_bytes = step_info.reward.to_le_bytes();
    let done_byte = [step_info.is_done as u8];

    let mut buf = Vec::with_capacity(obs_bytes.len() + 4 + 1);
    buf.extend_from_slice(obs_bytes);
    buf.extend_from_slice(&reward_bytes);
    buf.extend_from_slice(&done_byte);

    out.write_all(&buf).unwrap();

    out.flush().unwrap();
}

pub fn run_worker(scenario: RetroEnvScenario) {
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

    if args.len() < 5 {
        eprintln!("Usage: worker <game> <platform> <save_state> <record_movie>");
        std::process::exit(1);
    }

    let scenario = RetroEnvScenario::new(
        &args[1],
        Platform::Genesis,
        &args[3],
        args[4].parse::<bool>().unwrap()
    );

    run_worker(scenario);
}
