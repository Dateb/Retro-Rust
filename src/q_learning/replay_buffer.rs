use burn::Tensor;
use rand::seq::index::sample;
use arraydeque::{ArrayDeque, Wrapping};
use burn::prelude::{Backend, Bool, Float, Int, TensorData};

pub struct RetroBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub actions: Tensor<B, 1, Int>,
    pub rewards: Tensor<B, 1, Float>,
    pub next_images: Tensor<B, 4>,
    pub dones: Tensor<B, 1, Bool>
}

#[derive(Clone, Debug)]
pub struct ReplayBuffer<B: Backend> {
    images: ArrayDeque<Vec<f32>, 1000, Wrapping>,
    actions: ArrayDeque<i32, 1000, Wrapping>,
    rewards: ArrayDeque<f32, 1000, Wrapping>,
    next_images: ArrayDeque<Vec<f32>, 1000, Wrapping>,
    dones: ArrayDeque<bool, 1000, Wrapping>,
    _marker: std::marker::PhantomData<B>
}

impl<B: Backend> ReplayBuffer<B> {
    pub fn new() -> Self {
        ReplayBuffer {
            images: ArrayDeque::new(),
            actions: ArrayDeque::new(),
            rewards: ArrayDeque::new(),
            next_images: ArrayDeque::new(),
            dones: ArrayDeque::new(),
            _marker: std::marker::PhantomData
        }
    }

    pub fn store_transition(
        &mut self,
        image: Vec<f32>,
        action: i32,
        reward: f32,
        next_image: Vec<f32>,
        done: bool
    ) {
        self.images.push_front(image);
        self.actions.push_front(action);
        self.rewards.push_front(reward);
        self.next_images.push_front(next_image);
        self.dones.push_front(done);
    }

    pub fn sample(&self, batch_size: usize) -> RetroBatch<B> {
        let buffer_size = self.len();
        assert!(batch_size <= buffer_size);

        let device = Default::default();
        let height = 320;
        let width = 320;
        let channels = 3;

        let mut rng = rand::rng();
        let indices = sample(&mut rng, buffer_size, batch_size);

        let actions = Tensor::<B, 1, Int>::from_ints(
            indices.iter().map(|i| self.actions[i]).collect::<Vec<i32>>().as_slice(),
            &device,
        );

        let batch_images_cpu: Vec<&Vec<f32>> = indices
            .iter()
            .map(|i| &self.images[i])
            .collect();

        let flattened_cpu_images: Vec<f32> = batch_images_cpu
            .iter()
            .flat_map(|frame| frame.iter()) // concatenate all frames
            .copied()
            .collect();

        let images: Tensor<B, 4> = Tensor::from_data(
            TensorData::new(flattened_cpu_images, [batch_size, height, width, channels]),
            &device
        );

        let rewards = Tensor::<B, 1, Float>::from_floats(
            indices.iter().map(|i| self.rewards[i]).collect::<Vec<f32>>().as_slice(),
            &device,
        );

        let batch_images_cpu: Vec<&Vec<f32>> = indices
            .iter()
            .map(|i| &self.next_images[i])
            .collect();

        let flattened_cpu_images: Vec<f32> = batch_images_cpu
            .iter()
            .flat_map(|frame| frame.iter()) // concatenate all frames
            .copied()
            .collect();

        let next_images: Tensor<B, 4> = Tensor::from_data(
            TensorData::new(flattened_cpu_images, [batch_size, height, width, channels]),
            &device
        );

        let dones = Tensor::<B, 1, Bool>::from_bool(
            indices.iter().map(|i| self.dones[i]).collect::<Vec<bool>>().as_slice().into(),
            &device,
        );

        RetroBatch { images, actions, rewards, next_images, dones }
    }

    pub fn len(&self) -> usize {
        self.images.len()
    }
}
