use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};

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

    pub fn step(&mut self, actions: Vec<usize>) -> Vec<StepInfo> {
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
        Command::new(Self::worker_path())
            .arg("Airstriker")
            .arg("Genesis")
            .arg("Level1.state")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("failed to execute emulator worker")
    }

    fn worker_path() -> std::path::PathBuf {
        let mut path = std::env::current_exe().unwrap();

        // current exe is something like:
        // target/debug/deps/your_test_binary

        path.pop(); // remove test binary name

        // If inside deps/, go up one more level
        if path.ends_with("deps") {
            path.pop();
        }

        path.join("emulator_worker")
    }

    fn send_action(&mut self, env_idx: usize, action: usize) {
        let stdin = &mut self.stdins[env_idx];
        writeln!(stdin, "{}", action).unwrap();
        stdin.flush().unwrap();
    }

    fn read_step(&mut self, env_idx: usize) -> StepInfo {
        let reader = &mut self.readers[env_idx];
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap()
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
