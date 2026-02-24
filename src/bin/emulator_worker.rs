use std::{env, io};
use std::io::{BufRead, Stdout, Write};
use retro_rust::environments::image_retro_env::{emulator, ImageRetroEnv};
use retro_rust::environments::image_retro_env::platform::Platform;
use retro_rust::traits::retro_env::RetroEnv;

fn process_action_command(cmd: &str, env: &mut ImageRetroEnv, out: &mut Stdout) {
    let action = cmd.parse().expect("Invalid action");
    let step_info = env.step(action);
    
    let response = serde_json::to_string(&step_info).unwrap();
    writeln!(out, "{}", response).unwrap();
    out.flush().unwrap();

    if step_info.is_done { env.reset(); }
}

pub fn run_worker(game: &str, platform: &str, save_state: &str) {
    let game_name = "Airstriker";
    let platform = Platform::Genesis;
    let save_state_name = String::from("Level1.state");

    let mut env = ImageRetroEnv::new(game_name, platform, save_state_name);
    
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
