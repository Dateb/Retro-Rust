use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{GradientsParams, LearningRate, Optimizer};
use burn::prelude::{Backend, Bool, Float, Int, TensorData, ToElement};
use burn::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Device;
use burn::train::{RegressionOutput, TrainOutput, TrainStep};
use crate::q_learning::network::Network;
use crate::q_learning::network_config::NetworkConfig;
use crate::q_learning::replay_buffer::RetroBatch;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    behavior_network: Network<B>,
    target_network: Network<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &Device<B>, num_actions: usize) -> Self {
        let behavior_network: Network<B> = NetworkConfig::new(num_actions, 512).init::<B>(device);
        let target_network = NetworkConfig::new(num_actions, 512).init::<B>(device);

        Model { behavior_network, target_network }
    }

    pub fn forward_regression(
        &self,
        images: Tensor<B, 4>,
        actions: Tensor<B, 1, Int>,
        rewards: Tensor<B, 1, Float>,
        next_images: Tensor<B, 4>,
        dones: Tensor<B, 1, Float>,
        gamma: f64,
    ) -> Tensor<B, 1> {
        // Q(s, a)
        let q_values_all: Tensor<B, 2> = self.behavior_network.forward(images);
        let q_values: Tensor<B, 1> = q_values_all.gather(1, actions.unsqueeze_dim(1)).squeeze();

        // max_a' Q_target(s', a')
        let next_q_values_all = self.target_network.forward(next_images);
        let next_q_values: Tensor<B, 1> = next_q_values_all.max_dim(1).squeeze();

        let target_q = rewards + gamma * next_q_values.mul(1.0 - dones);

        // MSE loss
        MseLoss::new().forward(q_values, target_q, Reduction::Mean)
    }

    pub fn predict_action(&self, device: &Device<B>, image: Vec<f32>) -> usize {
        let next_image_tensor: Tensor<B, 4> = Tensor::from_data(
            TensorData::new(image, [1, 4, 84, 84]),
            device,
        );

        let q_values = self.behavior_network.forward(next_image_tensor);
        let action = q_values.argmax(1).into_scalar().to_i32() as usize;

        action
    }

    pub fn update_target_network(&mut self) {
        self.target_network = self.behavior_network.clone();
    }
}

pub fn train_step<B: Backend>(model: &Model<B>, batch: RetroBatch<B>) -> Tensor<B, 1> {
    // Forward + loss
    let loss = model.forward_regression(
        batch.images,
        batch.actions,
        batch.rewards,
        batch.next_images,
        batch.dones,
        0.99,
    );

    loss
}