use std::fs::File;
use std::io;
use std::io::Write;
use burn::backend::{Autodiff, Wgpu};
use std::time::Instant;
use crate::env::RetroEnv;
use crate::q_learning::QLearner;
use crate::q_learning::policy::Policy;

type Backend = Autodiff<Wgpu>;

pub struct LearnTimer {
    learner: QLearner<Backend>,
    env: RetroEnv,
    time_measurements: Vec<(u64, f32)>
}

impl LearnTimer {
    pub fn new(env: RetroEnv) -> Self {
        let device = &Default::default();
        let learner: QLearner<Backend> = QLearner::new(device, env.num_actions());
        let mut time_measurements = Vec::new();
        LearnTimer { learner, env, time_measurements }
    }

    pub fn run_and_write(&mut self, max_episodes: usize) {
        let device = &Default::default();
        let mut policy: Policy<Backend> = Policy::new(device, self.env.num_actions());
        let start_time = Instant::now();

        for _ in 0..max_episodes {
            policy = self.learner.learn_episode(&mut self.env, policy);

            let time_measurement = (start_time.elapsed().as_secs(), self.learner.rewards.average().unwrap());

            self.time_measurements.push(time_measurement);

            self.save_measurements().expect("Saving measurements has a problem.");
        }
    }

    fn save_measurements(&self) -> Result<(), io::Error> {
        let json = serde_json::to_string_pretty(&self.time_measurements)?;
        let mut file = File::create("time_measurements.json")?;
        file.write_all(json.as_bytes())?;

        Ok(())
    }
}
