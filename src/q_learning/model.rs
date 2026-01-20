use burn::backend::Wgpu;
use burn::module::Module;
use burn::nn::conv::Conv2d;
use burn::nn::{Dropout, Linear, Relu};
use burn::nn::loss::{CrossEntropyLossConfig, MseLoss, Reduction};
use burn::nn::pool::AdaptiveAvgPool2d;
use burn::prelude::{Backend, Bool, Float, TensorData};
use burn::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Int;
use burn::train::{ClassificationOutput, RegressionOutput, TrainOutput, TrainStep};
use crate::q_learning::replay_buffer::{ReplayBuffer, RetroBatch};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub conv1: Conv2d<B>,
    pub conv2: Conv2d<B>,
    pub pool: AdaptiveAvgPool2d,
    pub dropout: Dropout,
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub activation: Relu,
}

impl<B: Backend> Model<B> {
    /// # Shapes
    ///   - Images [width, height, channel]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, frames, height, width] = images.dims();

        // Permute dimensions to: [batch_size, channel, height, width]
        let x = images.clone().reshape([batch_size, frames, height, width]);

        let x = self.conv1.forward(x); // [batch_size, 8, _, _]
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x); // [batch_size, 16, _, _]
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x) // [batch_size, num_classes]
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

        let q_values_all = self.forward(images);
        let q_values = q_values_all.gather(1, actions.unsqueeze_dim(1));

        let next_q_values_all = self.forward(next_images);
        let next_q_values = next_q_values_all.max_dim(1);

        let dones_f: Tensor<B, 1, Float> = dones.clone().float();
        let target_q: Tensor<B, 2> = rewards.unsqueeze_dim(1) + gamma * next_q_values.clone() * (1 - dones_f.unsqueeze_dim(1));
        let loss: Tensor<B, 1> = MseLoss::new()
            .forward(q_values.clone(), target_q.clone(), Reduction::Mean);

        RegressionOutput::new(loss, next_q_values.clone(), target_q)
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
