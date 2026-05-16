use std::io::{BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::io::Read;
use crate::environments::retro_env::RetroEnvScenario;
use crate::traits::retro_env::StepInfo;

pub struct VectorRetroEnv {
    workers: Vec<Child>,
    stdins: Vec<ChildStdin>,
    readers: Vec<BufReader<std::process::ChildStdout>>,
    step_buffers: Vec<Vec<u8>>,
    num_envs: usize,
}

impl VectorRetroEnv {
    pub fn new(scenarios: Vec<RetroEnvScenario>) -> Self {
        let num_envs = scenarios.len();

        let mut workers = Vec::with_capacity(num_envs);
        let mut stdins = Vec::with_capacity(num_envs);
        let mut readers = Vec::with_capacity(num_envs);

        for scenario in &scenarios {
            let mut worker = Self::create_worker(scenario);

            let stdin = worker.stdin.take().expect("Worker stdin missing");
            let stdout = worker.stdout.take().expect("Worker stdout missing");

            stdins.push(stdin);
            readers.push(BufReader::new(stdout));
            workers.push(worker);
        }

        let obs_len = 84*84*4*4;

        let mut step_buffers = Vec::with_capacity(num_envs);
        for _ in 0..num_envs {
            step_buffers.push(vec![0u8; obs_len + 5]);
        }

        Self {
            workers,
            stdins,
            readers,
            step_buffers,
            num_envs,
        }
    }

    pub fn step(&mut self, actions: &[usize]) -> Vec<StepInfo> {
        for (i, action) in actions.iter().enumerate() {
            self.send_action(i, *action);
        }
        
        let mut results = Vec::with_capacity(self.num_envs);

        for i in 0..self.num_envs {
            let step_info = self.read_step(i);
            results.push(step_info);
        }

        results
    }
    
    fn create_worker(retro_env_scenario: &RetroEnvScenario) -> Child {
        let worker_path = Self::resolve_worker_path();
        println!("Spawning worker at: {:?}", worker_path);

        Command::new(Self::resolve_worker_path())
            .arg(retro_env_scenario.game_name)
            .arg(retro_env_scenario.platform.as_str())
            .arg(retro_env_scenario.save_state_name)
            .arg(retro_env_scenario.record_movie.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("failed to execute emulator worker")
    }

    fn resolve_worker_path() -> std::path::PathBuf {
        if let Ok(path) = std::env::var("CARGO_BIN_EXE_emulator_worker") {
            return path.into();
        }

        // Fallback for normal runs
        let mut path = std::env::current_exe().unwrap();
        path.pop();

        if path.ends_with("deps") {
            path.pop();
        }

        path.join("emulator_worker")
    }


    fn send_action(&mut self, env_idx: usize, action: usize) {
        let stdin = &mut self.stdins[env_idx];
        writeln!(stdin, "{}", action).unwrap();
    }

    fn read_step(&mut self, env_idx: usize) -> StepInfo {
        let reader = &mut self.readers[env_idx];

        let obs_len = 84*84*4*4;

        reader.read_exact(&mut self.step_buffers[env_idx]).unwrap();

        let obs_bytes = &self.step_buffers[env_idx][..obs_len];
        let reward_bytes = &self.step_buffers[env_idx][obs_len..obs_len+4];
        let done_byte = self.step_buffers[env_idx][obs_len+4];

        let observation: Vec<f32> = bytemuck::cast_slice(obs_bytes).to_vec();
        let reward = f32::from_le_bytes(reward_bytes.try_into().unwrap());
        let is_done = done_byte != 0;

        StepInfo {
            observation,
            reward,
            is_done,
        }
    }

    pub fn num_envs(&self) -> usize {
        self.num_envs
    }
}

impl Drop for VectorRetroEnv {
    fn drop(&mut self) {
        // Gracefully shutdown workers
        for stdin in &mut self.stdins {
            let _ = writeln!(stdin, "CLOSE");
            let _ = stdin.flush();
        }

        for worker in &mut self.workers {
            let _ = worker.wait();
        }
    }
}
