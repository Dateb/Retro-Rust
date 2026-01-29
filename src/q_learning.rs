mod network_config;
mod replay_buffer;
pub(crate) mod model;
mod utils;

use burn::module::AutodiffModule;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::optim::adaptor::OptimizerAdaptor;
use rand::{rng, Rng};
use crate::q_learning::replay_buffer::ReplayBuffer;
use burn::prelude::{Float, Int, TensorData, ToElement};
use burn::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Device;
use crate::env::RetroEnv;
use crate::q_learning::model::Model;
use crate::q_learning::network_config::NetworkConfig;
use crate::q_learning::utils::RollingAverage;

const TARGET_UPDATE_INTERVAL: usize = 1000;

pub struct QLearner<B: AutodiffBackend> {
    device: Device<B>,
    target_network: Model<B>,
    pub replay_buffer: ReplayBuffer<B>,
    optimizer: OptimizerAdaptor<Adam, Model<B>, B>,
    num_actions: usize,
    batch_size: usize,
    pub rewards: RollingAverage,
    iteration_count: usize
}

impl<B: AutodiffBackend> QLearner<B> {
    pub fn new(num_actions: usize) -> Self {
        let device = Default::default();
        let mut target_network: Model<B> = NetworkConfig::new(num_actions, 512).init(&device);
        let batch_size = 32;
        let replay_buffer = ReplayBuffer::new(10_000);
        let mut optimizer = AdamConfig::new().init();
        let rewards = RollingAverage::new();
        let mut iteration_count = 0;

        QLearner {
            device,
            target_network,
            replay_buffer,
            optimizer,
            num_actions,
            batch_size,
            rewards,
            iteration_count
        }
    }

    pub fn new_policy_network(&self) -> Model<B> {
        NetworkConfig::new(self.num_actions, 512).init(&self.device)
    }

    pub fn learn(&mut self, env: &mut RetroEnv, num_episodes: usize) {
        let mut policy_network: Model<B> = self.new_policy_network();
        for _ in 1..num_episodes {
            policy_network = self.learn_episode(env, policy_network);

            dbg!(self.rewards.average().expect("No elements to average over"));
            dbg!(self.iteration_count);
        }
    }

    pub fn learn_episode(&mut self, env: &mut RetroEnv, mut policy_network: Model<B>) -> Model<B> {
        let mut image = env.reset();
        let mut next_action = self.get_next_action(&policy_network, image.clone());

        while !env.is_done() {
            let step_info = env.step(next_action);

            let next_image = step_info.0;
            let reward = step_info.1;
            let done = step_info.2;

            self.replay_buffer.store_transition(
                &image,
                next_action as i32,
                reward,
                &next_image,
                done
            );

            image = next_image.clone();

            next_action = match self.replay_buffer.len >= self.batch_size {
                true => {
                    policy_network = self.train(policy_network);
                    self.get_next_action(&policy_network, next_image.clone())
                },
                false => rng().random_range(0..self.num_actions)
            };

            if (self.iteration_count + 1) % TARGET_UPDATE_INTERVAL == 0 {
                self.target_network = policy_network.clone();
            }
            self.iteration_count += 1;
        }

        self.rewards.push(env.episode_reward());

        policy_network
    }

    fn train(&mut self, policy_network: Model<B>) -> Model<B> {
        let retro_batch = self.replay_buffer.sample(32, &self.device);

        let loss = self.forward_regression(
            retro_batch.images,
            retro_batch.actions,
            retro_batch.rewards,
            retro_batch.next_images,
            retro_batch.dones,
            &policy_network,
            &self.target_network,
            0.99,
        );

        let gradients = loss.backward();
        let gradient_params = GradientsParams::from_grads(gradients, &policy_network);

        self.optimizer.step(1e-4, policy_network, gradient_params)
    }

    fn get_next_action(&self, policy_network: &Model<B>, image: Vec<f32>) -> usize {
        match rng().random_range(0..100) < 5 {
            true => rng().random_range(0..self.num_actions),
            false => self.predict_action(image, &policy_network)
        }
    }

    fn predict_action(&self, image: Vec<f32>, behavior_network: &Model<B>) -> usize {
        let next_image_tensor: Tensor<B, 4> = Tensor::from_data(
            TensorData::new(image, [1, 4, 84, 84]),
            &self.device,
        );

        let q_values = behavior_network.forward(next_image_tensor).detach();
        let action = q_values.argmax(1).into_scalar().to_i32() as usize;

        action
    }

    fn forward_regression(
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

        MseLoss::new().forward(q_values, target_q.detach(), Reduction::Mean)
    }
}
