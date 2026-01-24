use burn::config::Config;
use burn::module::AutodiffModule;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::AdaptiveAvgPool2dConfig;
use burn::nn::{DropoutConfig, LinearConfig, Relu};
use burn::prelude::Backend;
use crate::q_learning::network::Network;

#[derive(Config, Debug)]
pub struct NetworkConfig {
    num_classes: usize,
    hidden_size: usize,
}

impl NetworkConfig {
    /// Returns the initialized network.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Network<B> {
        Network {
            conv1: Conv2dConfig::new([4, 32], [8, 8])
                .with_stride([4, 4])
                .init(device),
            conv2: Conv2dConfig::new([32, 64], [4, 4])
                .with_stride([2, 2])
                .init(device),
            conv3: Conv2dConfig::new([64, 64], [3, 3])
                .with_stride([1, 1])
                .init(device),
            linear1: LinearConfig::new(64 * 7 * 7, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            activation: Relu::new()
        }
    }
}
