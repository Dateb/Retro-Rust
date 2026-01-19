mod model;
mod model_config;
use rand::{rng, Rng, TryRngCore};
use crate::q_learning::model::Model;
use crate::q_learning::model_config::ModelConfig;
use burn::prelude::Backend;
use burn::backend::Wgpu;
use burn::Tensor;
use crate::env::RetroEnv;

pub struct RetroBatch<B: Backend> {
    image: Tensor<B, 3>,
    action: usize,
    next_image: Tensor<B, 3>,
    reward: f32,
    is_done: bool
}

pub struct QLearner {
    model: Model<Wgpu>,
    num_actions: usize
}

impl QLearner {
    pub fn new(num_actions: usize) -> Self {
        let device = Default::default();

        QLearner { model: ModelConfig::new(num_actions, 512).init::<Wgpu>(&device), num_actions }
    }

    pub fn learn(&self, env: RetroEnv) {
        let mut rng = rng();
        let mut image = env.reset();
        let mut next_action_index = 0usize;
        for _ in 1..1000000 {
            let step_info = env.step(next_action_index);

            let next_image = step_info.0;
            let current_reward = step_info.1;
            let is_done = step_info.2;

            let train_output =
                self.model.forward_regression(
                    &image,
                    next_action_index,
                    &next_image,
                    current_reward,
                    is_done
                );

            image = next_image;

            let is_done = step_info.2;
            if is_done {
                dbg!(env.episode_reward());
                env.reset();
            } else {
                if rng.random_range(0..100) < 5 {
                    next_action_index = rng.random_range(0..self.num_actions);
                } else {
                    let q_values = train_output.output;
                    next_action_index = q_values
                        .argmax(1)
                        .flatten::<1>(0, 1)
                        .into_scalar() as usize;
                }
            }
        }
    }
}