mod model;
mod model_config;
mod replay_buffer;

use std::time::Instant;
use rand::{rng, Rng, TryRngCore};
use crate::q_learning::model::Model;
use crate::q_learning::model_config::ModelConfig;
use crate::q_learning::replay_buffer::ReplayBuffer;
use burn::prelude::{Backend, TensorData, ToElement};
use burn::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn::train::TrainStep;
use crate::env::RetroEnv;

pub struct QLearner<B: Backend> {
    model: Model<B>,
    num_actions: usize,
    pub replay_buffer: ReplayBuffer<B>
}

impl<B: Backend + AutodiffBackend> QLearner<B> {
    pub fn new(num_actions: usize) -> Self {
        let device = Default::default();
        let model = ModelConfig::new(num_actions, 512).init::<B>(&device);
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

                let train_output
                    = TrainStep::step(&self.model, retro_batch);

                if rng.random_range(0..100) < 5 {
                    next_action_index = rng.random_range(0..self.num_actions);
                } else {
                    // Todo: There should be a way to avoid this forward pass (predict function)
                    let device = Default::default();

                    let next_image_tensor: Tensor<B, 4> = Tensor::from_data(
                        TensorData::new(
                            next_image, // Vec<f32>, already flattened
                            [1, 4, 84, 84],
                        ),
                        &device,
                    );

                    let q_values = self.model.forward(next_image_tensor);
                    next_action_index = q_values.argmax(1).into_scalar().to_i32() as usize;
                }
            } else {
                next_action_index = rng.random_range(0..self.num_actions);
            }

            if done {
                dbg!(env.episode_reward());
                env.reset();
            }
        }
    }
}