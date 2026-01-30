mod network_config;
mod replay_buffer;
pub mod model;
mod utils;
pub mod policy;

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
use crate::q_learning::policy::Policy;
use crate::q_learning::utils::RollingAverage;

const TARGET_UPDATE_INTERVAL: usize = 1000;

pub struct QLearner<B: AutodiffBackend> {
    device: Device<B>,
    target_network: Model<B>,
    pub replay_buffer: ReplayBuffer<B>,
    optimizer: OptimizerAdaptor<Adam, Model<B>, B>,
    num_actions: usize,
    pub rewards: RollingAverage,
    iteration_count: usize
}

impl<B: AutodiffBackend> QLearner<B> {
    pub fn new(device: &Device<B>, num_actions: usize) -> Self {
        let mut target_network: Model<B> = NetworkConfig::new(num_actions, 512).init(device);
        let capacity = 10_000;
        let replay_buffer = ReplayBuffer::new(32, capacity, capacity);
        let mut optimizer = AdamConfig::new().init();
        let rewards = RollingAverage::new();
        let mut iteration_count = 0;

        QLearner {
            device: device.clone(),
            target_network,
            replay_buffer,
            optimizer,
            num_actions,
            rewards,
            iteration_count
        }
    }

    pub fn learn(&mut self, env: &mut RetroEnv, num_episodes: usize) {
        let mut policy: Policy<B> = Policy::new(&self.device, env.num_actions());
        for _ in 1..num_episodes {
            policy = self.learn_episode(env, policy);

            dbg!(self.rewards.average().expect("No elements to average over"));
            dbg!(self.iteration_count);
        }
    }

    pub fn learn_episode(&mut self, env: &mut RetroEnv, mut policy: Policy<B>) -> Policy<B> {
        let mut image = env.reset();
        let mut next_action = policy.get_next_action(image.clone(), env.num_actions(), &self.device);

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

            next_action = match self.replay_buffer.learning_is_ready() {
                true => {
                    policy = self.train(policy);
                    policy.get_next_action(next_image.clone(), env.num_actions(), &self.device)
                },
                false => rng().random_range(0..self.num_actions)
            };

            if (self.iteration_count + 1) % TARGET_UPDATE_INTERVAL == 0 {
                self.target_network = policy.network.clone();
            }
            self.iteration_count += 1;
        }

        self.rewards.push(env.episode_reward());

        policy
    }

    fn train(&mut self, policy: Policy<B>) -> Policy<B> {
        let retro_batch = self.replay_buffer.sample(32, &self.device);

        let loss = self.forward_regression(
            retro_batch.images,
            retro_batch.actions,
            retro_batch.rewards,
            retro_batch.next_images,
            retro_batch.dones,
            &policy.network,
            &self.target_network,
            0.99,
        );

        policy.update(loss, &mut self.optimizer)
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
