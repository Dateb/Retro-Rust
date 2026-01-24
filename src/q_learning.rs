mod network_config;
mod replay_buffer;
mod model;
mod utils;

use burn::module::AutodiffModule;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use rand::{rng, Rng, TryRngCore};
use crate::q_learning::replay_buffer::{ReplayBuffer, RetroBatch};
use burn::prelude::{Backend, Float, Int, TensorData, ToElement};
use burn::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Device;
use burn::train::TrainStep;
use crate::env::RetroEnv;
use crate::q_learning::model::Model;
use crate::q_learning::network_config::NetworkConfig;
use crate::timeit;

const TARGET_UPDATE_INTERVAL: i32 = 1000;

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
        let mut target_network: Model<B> = NetworkConfig::new(self.num_actions, 512).init(&self.device);
        target_network.valid();
        let mut behavior_network: Model<B> = NetworkConfig::new(self.num_actions, 512).init(&self.device);
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
                // let retro_batch = timeit!("sample", {self.replay_buffer.sample(32, &self.device)});
                let retro_batch = self.replay_buffer.sample(32, &self.device);

                let loss = self.forward_regression(
                    retro_batch.images,
                    retro_batch.actions,
                    retro_batch.rewards,
                    retro_batch.next_images,
                    retro_batch.dones,
                    &behavior_network,
                    &target_network,
                    0.99,
                );

                let gradients = loss.backward();
                let gradient_params = GradientsParams::from_grads(gradients, &behavior_network);

                behavior_network.valid();
                next_action_index = match rng.random_range(0..100) < 5 {
                    true => rng.random_range(0..self.num_actions),
                    false => self.predict_action(next_image, &behavior_network)
                };
                behavior_network = optimizer.step(1e-4, behavior_network, gradient_params);
            } else {
                next_action_index = rng.random_range(0..self.num_actions);
            }

            if done {
                dbg!(env.episode_reward());
                dbg!(i);
                image = env.reset();
            }

            if (i + 1) % TARGET_UPDATE_INTERVAL == 0 {
                target_network = behavior_network.clone();
                target_network.valid();
            }
        }
    }

    pub fn forward_regression(
        &self,
        images: Tensor<B, 4>,
        actions: Tensor<B, 1, Int>,
        rewards: Tensor<B, 1, Float>,
        next_images: Tensor<B, 4>,
        dones: Tensor<B, 1, Float>,
        behavior_network: &Model<B>,
        target_network: &Model<B>,
        gamma: f64,
    ) -> Tensor<B, 1> {
        // Q(s, a)
        let q_values_all: Tensor<B, 2> = behavior_network.forward(images);
        let q_values: Tensor<B, 1> = q_values_all.gather(1, actions.unsqueeze_dim(1)).squeeze();

        // max_a' Q_target(s', a')
        let next_q_values_all = target_network.forward(next_images);
        let next_q_values: Tensor<B, 1> = next_q_values_all.max_dim(1).squeeze();

        let target_q = rewards + gamma * next_q_values.mul(1.0 - dones);

        // MSE loss
        MseLoss::new().forward(q_values, target_q, Reduction::Mean)
    }

    pub fn predict_action(&self, image: Vec<f32>, behavior_network: &Model<B>) -> usize {
        let next_image_tensor: Tensor<B, 4> = Tensor::from_data(
            TensorData::new(image, [1, 4, 84, 84]),
            &self.device,
        );

        let q_values = behavior_network.forward(next_image_tensor);
        let action = q_values.argmax(1).into_scalar().to_i32() as usize;

        action
    }
}
