mod network;
mod network_config;
mod replay_buffer;
mod model;

use burn::module::AutodiffModule;
use rand::{rng, Rng, TryRngCore};
use crate::q_learning::replay_buffer::ReplayBuffer;
use burn::prelude::{Backend, ToElement};
use burn::tensor::backend::AutodiffBackend;
use burn::train::TrainStep;
use crate::env::RetroEnv;
use crate::q_learning::model::Model;

const TARGET_UPDATE_INTERVAL: i32 = 1000;

pub struct QLearner<B: AutodiffBackend> {
    model: Model<B>,
    num_actions: usize,
    pub replay_buffer: ReplayBuffer<B>
}

impl<B: AutodiffBackend> QLearner<B> {
    pub fn new(num_actions: usize) -> Self {
        let model: Model<B> = Model::new(num_actions);
        let replay_buffer = ReplayBuffer::new(1000);

        QLearner { model, replay_buffer, num_actions }
    }

    pub fn learn(&mut self, mut env: RetroEnv) {
        let batch_size = 32;

        let mut rng = rng();
        let mut image = env.reset();
        let mut next_action_index = 0usize;
        for i in 1..1000000 {
            let step_info = env.step(next_action_index);

            let next_image = step_info.0;
            let reward = step_info.1;
            let done = step_info.2;

            self.replay_buffer.store_transition(
                &image,
                next_action_index as i32,
                reward,
                &next_image,
                done
            );

            image = next_image.clone();

            if self.replay_buffer.len >= batch_size {
                let retro_batch = self.replay_buffer.sample(32);

                TrainStep::step(&self.model, retro_batch);

                next_action_index = match rng.random_range(0..100) < 5 {
                    true => rng.random_range(0..self.num_actions),
                    false => self.model.predict_action(next_image)
                };
            } else {
                next_action_index = rng.random_range(0..self.num_actions);
            }

            if done {
                dbg!(env.episode_reward());
                dbg!(i);
                env.reset();
            }

            if (i + 1) % TARGET_UPDATE_INTERVAL == 0 {
                self.model.update_target_network();
            }
        }
    }
}
