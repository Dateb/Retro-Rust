use std::io::{BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::io::Read;

use crate::traits::retro_env::StepInfo;

pub struct VectorRetroEnv {
    workers: Vec<Child>,
    stdins: Vec<ChildStdin>,
    readers: Vec<BufReader<std::process::ChildStdout>>,
    num_envs: usize,
}

impl VectorRetroEnv {
    pub fn new(num_envs: usize) -> Self {
        let mut workers = Vec::with_capacity(num_envs);
        let mut stdins = Vec::with_capacity(num_envs);
        let mut readers = Vec::with_capacity(num_envs);

        for _ in 0..num_envs {
            let mut worker = Self::create_worker();

            let stdin = worker.stdin.take().unwrap();
            let stdout = worker.stdout.take().unwrap();

            stdins.push(stdin);
            readers.push(BufReader::new(stdout));
            workers.push(worker);
        }

        Self {
            workers,
            stdins,
            readers,
            num_envs,
        }
    }

    pub fn step(&mut self, actions: &Vec<usize>) -> Vec<StepInfo> {
        for i in 0..self.num_envs {
            self.send_action(i, actions[i]);
        }
        
        let mut results = Vec::with_capacity(self.num_envs);

        for i in 0..self.num_envs {
            let step_info = self.read_step(i);
            results.push(step_info);
        }

        results
    }
    
    fn create_worker() -> Child {
        let worker_path = Self::resolve_worker_path();
        println!("Spawning worker at: {:?}", worker_path);

        Command::new(Self::resolve_worker_path())
            .arg("Airstriker")
            .arg("Genesis")
            .arg("Level1.state")
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

        let obs_len = 84*84*4;

        // ---- Read observation ----
        let mut obs = vec![0f32; obs_len];
        let obs_bytes = bytemuck::cast_slice_mut::<f32, u8>(&mut obs);
        reader.read_exact(obs_bytes).unwrap();

        // ---- Read reward ----
        let mut reward_buf = [0u8; 4];
        reader.read_exact(&mut reward_buf).unwrap();
        let reward = f32::from_le_bytes(reward_buf);

        // ---- Read done ----
        let mut done_buf = [0u8; 1];
        reader.read_exact(&mut done_buf).unwrap();
        let is_done = done_buf[0] != 0;

        StepInfo {
            observation: obs,
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
