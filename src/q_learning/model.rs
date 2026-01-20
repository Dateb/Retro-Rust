use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::prelude::{Backend, Bool, Float, Int, TensorData, ToElement};
use burn::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{RegressionOutput, TrainOutput, TrainStep};
use crate::q_learning::network::Network;
use crate::q_learning::network_config::NetworkConfig;
use crate::q_learning::replay_buffer::RetroBatch;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    behavior_network: Network<B>,
    target_network: Network<B>
}

impl<B: Backend> Model<B> {
    pub fn new(num_actions: usize) -> Self {
        let device = Default::default();
        let behavior_network = NetworkConfig::new(num_actions, 512).init::<B>(&device);
        let target_network = NetworkConfig::new(num_actions, 512).init::<B>(&device);

        Model { behavior_network, target_network }
    }

    pub fn forward_regression(
        &self,
        images: Tensor<B, 4>,
        actions: Tensor<B, 1, Int>,
        rewards: Tensor<B, 1, Float>,
        next_images: Tensor<B, 4>,
        dones: Tensor<B, 1, Bool>
    ) -> RegressionOutput<B> {

        let gamma = 0.99;

        let q_values_all = self.behavior_network.forward(images);
        let q_values = q_values_all.gather(1, actions.unsqueeze_dim(1));

        let next_q_values_all = self.target_network.forward(next_images);
        let next_q_values = next_q_values_all.max_dim(1).detach();

        let dones_f: Tensor<B, 1, Float> = dones.clone().float();
        let target_q: Tensor<B, 2> = rewards.unsqueeze_dim(1) + gamma * next_q_values.clone() * (1 - dones_f.unsqueeze_dim(1));
        let loss: Tensor<B, 1> = MseLoss::new()
            .forward(q_values.clone(), target_q.clone(), Reduction::Mean);

        RegressionOutput::new(loss, next_q_values.clone(), target_q)
    }

    pub fn predict_action(&self, image: Vec<f32>) -> usize {
        let device = Default::default();

        let next_image_tensor: Tensor<B, 4> = Tensor::from_data(
            TensorData::new(
                image,
                [1, 4, 84, 84],
            ),
            &device,
        );

        let q_values = self.behavior_network.forward(next_image_tensor);
        let action = q_values.argmax(1).into_scalar().to_i32() as usize;

        action
    }

    pub fn update_target_network(&mut self) {
        self.target_network = self.behavior_network.clone();
    }
}

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = RetroBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: RetroBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self
            .forward_regression(
                batch.images,
                batch.actions,
                batch.rewards,
                batch.next_images,
                batch.dones
            );

        TrainOutput::new(self, item.loss.backward(), item)
    }
}