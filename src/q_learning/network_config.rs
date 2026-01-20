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
    #[config(default = "0.5")]
    dropout: f64,
}

impl NetworkConfig {
    /// Returns the initialized network.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Network<B> {
        Network {
            conv1: Conv2dConfig::new([4, 8], [3, 3]).init(device),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}
