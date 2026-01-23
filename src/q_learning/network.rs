use burn::backend::Wgpu;
use burn::module::{AutodiffModule, Module};
use burn::nn::conv::Conv2d;
use burn::nn::{Dropout, Linear, Relu};
use burn::nn::loss::{CrossEntropyLossConfig, MseLoss, Reduction};
use burn::nn::pool::AdaptiveAvgPool2d;
use burn::prelude::{Backend, Bool, Float, TensorData, ToElement};
use burn::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Int;
use burn::train::{ClassificationOutput, RegressionOutput, TrainOutput, TrainStep};
use crate::q_learning::replay_buffer::{ReplayBuffer, RetroBatch};

#[derive(Module, Debug)]
pub struct Network<B: Backend> {
    pub conv1: Conv2d<B>,
    pub conv2: Conv2d<B>,
    pub conv3: Conv2d<B>,
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub activation: Relu,
}

impl<B: Backend> Network<B> {
    /// # Shapes
    ///   - Images [width, height, channel]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, frames, height, width] = images.dims();

        // Permute dimensions to: [batch_size, channel, height, width]
        let x = images.clone().reshape([batch_size, frames, height, width]);

        let x = self.conv1.forward(x); // [batch_size, 8, _, _]
        let x = self.activation.forward(x);
        let x = self.conv2.forward(x); // [batch_size, 16, _, _]
        let x = self.activation.forward(x);
        let x = self.conv3.forward(x); // [batch_size, 16, _, _]
        let x = self.activation.forward(x);

        let x = x.reshape([batch_size, 64 * 7 * 7]);

        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x) // [batch_size, num_classes]
    }
}
