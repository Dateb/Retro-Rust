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

pub struct QLearner<B: AutodiffBackend> {
    device: Device<B>,
    target_network: Model<B>,
    pub replay_buffer: ReplayBuffer<B>,
    optimizer: OptimizerAdaptor<Adam, Model<B>, B>,
    num_actions: usize,
    train_frequency: usize,
    target_update_interval: usize,
    discount_factor: f64,
    total_time_steps: usize,
    exploration_fraction: f64,
    exploration_final_epsilon: f64,
    iteration_count: usize,
    pub rewards: RollingAverage,
}

impl<B: AutodiffBackend> QLearner<B> {
    pub fn new(
        device: &Device<B>,
        num_actions: usize,
        replay_buffer_capacity: usize,
        min_samples: usize,
        batch_size: usize,
        train_frequency: usize,
        target_update_interval: usize,
        discount_factor: f64,
        total_time_steps: usize,
        exploration_fraction: f64,
        exploration_final_epsilon: f64
    ) -> Self {
        let mut target_network: Model<B> = NetworkConfig::new(num_actions, 512).init(device);
        let replay_buffer = ReplayBuffer::new(
            batch_size,
            replay_buffer_capacity,
            min_samples
        );
        let mut optimizer = AdamConfig::new().init();
        let rewards = RollingAverage::new();
        let mut iteration_count = 0;

        QLearner {
            device: device.clone(),
            target_network,
            replay_buffer,
            optimizer,
            num_actions,
            train_frequency,
            target_update_interval,
            discount_factor,
            total_time_steps,
            exploration_fraction,
            exploration_final_epsilon,
            iteration_count,
            rewards,
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

            next_action = match self.replay_buffer.learning_is_ready()
                && self.iteration_count % self.train_frequency == 0 {
                true => {
                    policy = self.train(policy);
                    policy.get_next_action(next_image.clone(), env.num_actions(), &self.device)
                },
                false => rng().random_range(0..self.num_actions)
            };

            if (self.iteration_count + 1) % self.target_update_interval == 0 {
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
            self.discount_factor,
        );

        policy.update(
            loss,
            &mut self.optimizer,
            self.iteration_count as f64,
            self.total_time_steps as f64,
            self.exploration_fraction,
            self.exploration_final_epsilon
        )
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
        discount_factor: f64,
    ) -> Tensor<B, 1> {
        // Q(s, a)
        let q_values_all: Tensor<B, 2> = behavior_network.forward(images);
        let q_values: Tensor<B, 1> = q_values_all.gather(1, actions.unsqueeze_dim(1)).squeeze();

        // max_a' Q_target(s', a')
        let next_q_values_all = target_network.forward(next_images);
        let next_q_values: Tensor<B, 1> = next_q_values_all.max_dim(1).squeeze();

        let target_q = rewards + discount_factor * next_q_values.mul(1.0 - dones);

        MseLoss::new().forward(q_values, target_q.detach(), Reduction::Mean)
    }
}
