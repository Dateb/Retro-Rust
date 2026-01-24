mod network;
mod network_config;
mod replay_buffer;
mod model;
mod utils;

use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use rand::{rng, Rng, TryRngCore};
use crate::q_learning::replay_buffer::ReplayBuffer;
use burn::prelude::{Backend, ToElement};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Device;
use burn::train::TrainStep;
use crate::env::RetroEnv;
use crate::q_learning::model::{train_step, Model};
use crate::timeit;

const TARGET_UPDATE_INTERVAL: i32 = 10000;

pub struct QLearner<B: AutodiffBackend> {
    device: Device<B>,
    num_actions: usize,
    pub replay_buffer: ReplayBuffer<B>
}

impl<B: AutodiffBackend> QLearner<B> {
    pub fn new(num_actions: usize) -> Self {
        let device = Default::default();
        let replay_buffer = ReplayBuffer::new(100_000);

        QLearner { device, replay_buffer, num_actions }
    }

    pub fn learn(&mut self, mut env: RetroEnv) {
        let mut model: Model<B> = Model::new(&self.device, self.num_actions);
        let batch_size = 32;
        let mut optimizer = AdamConfig::new().init();

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
                let retro_batch = timeit!("sample", {self.replay_buffer.sample(32, &self.device)});
                // let retro_batch = self.replay_buffer.sample(32, &self.device);

                let loss = timeit!("train", {train_step(&model, retro_batch)});
                // let loss = train_step(&model, retro_batch);

                let gradients = loss.backward();
                let gradient_params = GradientsParams::from_grads(gradients, &model);
                model = optimizer.step(1e-4, model, gradient_params);

                next_action_index = match rng.random_range(0..100) < 5 {
                    true => rng.random_range(0..self.num_actions),
                    // false => model.predict_action(&self.device, next_image)
                    false => rng.random_range(0..self.num_actions)
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
                model.update_target_network();
            }
        }
    }
}
