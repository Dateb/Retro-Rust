use burn::optim::{Adam, GradientsParams, Optimizer};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::prelude::{Backend, Device, TensorData, ToElement};
use burn::Tensor;
use burn::tensor::backend::AutodiffBackend;
use rand::{rng, Rng};
use crate::q_learning::model::Model;
use crate::q_learning::network_config::NetworkConfig;

pub struct Policy<B: Backend> {
    pub network: Model<B>
}

impl<B: Backend + AutodiffBackend> Policy<B> {
    pub fn new(device: &Device<B>, num_actions: usize) -> Self {
        let network = NetworkConfig::new(num_actions, 512).init(device);
        Policy { network }
    }

    pub fn update(self, loss: Tensor<B, 1>, optimizer: &mut OptimizerAdaptor<Adam, Model<B>, B>) -> Self {
        let gradients = loss.backward();
        let gradient_params = GradientsParams::from_grads(gradients, &self.network);

        let policy_network = optimizer.step(1e-4, self.network, gradient_params);

        Policy { network: policy_network }
    }

    pub fn get_next_action(&self, image: Vec<f32>, num_actions: usize, device: &Device<B>) -> usize {
        match rng().random_range(0..100) < 98 {
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
