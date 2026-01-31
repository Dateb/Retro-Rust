use std::cmp;
use burn::optim::{Adam, GradientsParams, Optimizer};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::prelude::{Backend, Device, TensorData, ToElement};
use burn::Tensor;
use burn::tensor::backend::AutodiffBackend;
use rand::{rng, Rng};
use crate::q_learning::model::Model;
use crate::q_learning::network_config::NetworkConfig;

pub struct Policy<B: Backend> {
    pub network: Model<B>,
    exploration_rate: f64,
    lr: f64
}

impl<B: Backend + AutodiffBackend> Policy<B> {
    pub fn new(device: &Device<B>, num_actions: usize) -> Self {
        let network = NetworkConfig::new(num_actions, 512).init(device);
        Policy { network, exploration_rate: 1.0, lr: 1e-4 }
    }

    pub fn update(
        self,
        loss: Tensor<B, 1>,
        optimizer: &mut OptimizerAdaptor<Adam, Model<B>, B>,
        current_time_step: f64,
        total_time_steps: f64,
        exploration_fraction: f64,
        exploration_final_epsilon: f64
    ) -> Self {
        let gradients = loss.backward();
        let gradient_params = GradientsParams::from_grads(gradients, &self.network);

        let policy_network = optimizer.step(self.lr, self.network, gradient_params);

        let progress = current_time_step / (exploration_fraction * total_time_steps);
        let progress = progress.clamp(0.0, 1.0);

        let exploration_rate = 1.0 + progress * (exploration_final_epsilon - 1.0);

        Policy { network: policy_network, exploration_rate, lr: 1e-4 }
    }

    pub fn get_next_action(
        &self, image: Vec<f32>,
        num_actions: usize,
        device: &Device<B>
    ) -> usize {
        match rng().random_range(0.0..1.0) < self.exploration_rate {
            true => rng().random_range(0..num_actions),
            false => self.predict_action(image, device)
        }
    }

    fn predict_action(&self, image: Vec<f32>, device: &Device<B>) -> usize {
        let next_image_tensor: Tensor<B, 4> = Tensor::from_data(
            TensorData::new(image, [1, 4, 84, 84]),
            device,
        );

        let q_values = self.network.forward(next_image_tensor).detach();
        let action = q_values.argmax(1).into_scalar().to_i32() as usize;

        action
    }
}
