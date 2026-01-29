use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::Write;
use burn::backend::{Autodiff, Wgpu};
use std::time::Instant;
use crate::env::RetroEnv;
use crate::q_learning::model::Model;
use crate::q_learning::QLearner;

type Backend = Autodiff<Wgpu>;

pub struct LearnTimer {
    learner: QLearner<Backend>,
    env: RetroEnv,
    time_measurements: Vec<(u64, f32)>
}

impl LearnTimer {
    pub fn new(env: RetroEnv) -> Self {
        let learner: QLearner<Backend> = QLearner::new(env.num_actions());
        let mut time_measurements = Vec::new();
        LearnTimer { learner, env, time_measurements }
    }

    pub fn run_and_write(&mut self, max_episodes: usize) {
        let mut policy_network: Model<Backend> = self.learner.new_policy_network();
        let start_time = Instant::now();

        for _ in 0..max_episodes {
            policy_network = self.learner.learn_episode(&mut self.env, policy_network);

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
