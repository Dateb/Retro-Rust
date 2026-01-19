use burn::module::Module;
use burn::nn::conv::Conv2d;
use burn::nn::{Dropout, Linear, Relu};
use burn::nn::loss::{CrossEntropyLossConfig, MseLoss, Reduction};
use burn::nn::pool::AdaptiveAvgPool2d;
use burn::prelude::Backend;
use burn::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Int;
use burn::train::{ClassificationOutput, RegressionOutput, TrainOutput, TrainStep};
use crate::q_learning::RetroBatch;

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
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [width, height, channel] = images.dims();

        // Permute dimensions to: [batch_size, channel, height, width]
        let x = images.reshape([1, channel, height, width]);

        let x = self.conv1.forward(x); // [batch_size, 8, _, _]
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x); // [batch_size, 16, _, _]
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
        let x = x.reshape([1, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x) // [batch_size, num_classes]
    }

    pub fn forward_regression(
        &self,
        image: Tensor<B, 3>,
        reward: f32,
        is_done: bool
    ) -> RegressionOutput<B> {
        let gamma = 0.99;

        let next_q_values = self.forward(image);
        let target_q: Tensor<B, 2> = reward + gamma * next_q_values.clone() * (1 - is_done as usize) as f32;
        let loss: Tensor<B, 1> = MseLoss::new()
            .forward(next_q_values.clone(), target_q.clone(), Reduction::Mean);

        RegressionOutput::new(loss, next_q_values.clone(), target_q)
    }
}

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = RetroBatch<B>;
    type Output = RegressionOutput<B>;

    fn step(&self, batch: RetroBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self
            .forward_regression(batch.image, batch.reward, batch.is_done);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}