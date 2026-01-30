use std::cmp;
use burn::Tensor;
use rand::seq::index::sample;
use burn::prelude::{Backend, Float, Int, TensorData};
use burn::tensor::Device;

pub struct RetroBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub actions: Tensor<B, 1, Int>,
    pub rewards: Tensor<B, 1, Float>,
    pub next_images: Tensor<B, 4>,
    pub dones: Tensor<B, 1, Float>
}

const IMAGE_SIZE: usize = 4 * 84 * 84;

#[derive(Clone, Debug)]
pub struct ReplayBuffer<B: Backend> {
    images: Vec<f32>,
    actions: Vec<i32>,
    rewards: Vec<f32>,
    next_images: Vec<f32>,
    dones: Vec<f32>,

    batch_size: usize,
    capacity: usize,
    min_samples: usize,
    pub len: usize,
    pos: usize,

    _marker: std::marker::PhantomData<B>
}

// actions: Vec<Option<Tensor<B, 1, Int>>>
impl<B: Backend> ReplayBuffer<B> {
    pub fn new(batch_size: usize, capacity: usize, min_samples: usize) -> Self {
        Self {
            images: vec![0.0; capacity * IMAGE_SIZE],
            actions: vec![0; capacity],
            rewards: vec![0.0; capacity],
            next_images: vec![0.0; capacity * IMAGE_SIZE],
            dones: vec![0.0; capacity],

            batch_size,
            capacity,
            min_samples: cmp::max(batch_size, min_samples),
            len: 0,
            pos: 0,

            _marker: std::marker::PhantomData
        }
    }

    pub fn store_transition(
        &mut self,
        image: &[f32],
        action: i32,
        reward: f32,
        next_image: &[f32],
        done: bool,
    ) {
        let idx = self.pos;

        let start = idx * IMAGE_SIZE;
        let end = start + IMAGE_SIZE;

        self.images[start..end].copy_from_slice(image);
        self.next_images[start..end].copy_from_slice(next_image);

        self.actions[idx] = action;
        self.rewards[idx] = reward;
        self.dones[idx] = if done { 1.0 } else { 0.0 };

        self.pos = (self.pos + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    pub fn sample(&self, batch_size: usize, device: &Device<B>) -> RetroBatch<B> {
        let mut rng = rand::rng();
        let indices = sample(&mut rng, self.len, batch_size);

        let mut batch_images = Vec::with_capacity(batch_size * IMAGE_SIZE);

        for i in indices.iter() {
            let start = i * IMAGE_SIZE;
            let end = start + IMAGE_SIZE;
            batch_images.extend_from_slice(&self.images[start..end]);
        }

        let images: Tensor<B, 4> = Tensor::<B, 4>::from_data(
            TensorData::new(
                batch_images, // Vec<f32>, already flattened
                [batch_size, 4, 84, 84],
            ),
            &device,
        );

        let actions = Tensor::<B, 1, Int>::from_ints(
            indices.iter().map(|i| self.actions[i]).collect::<Vec<i32>>().as_slice(),
            &device,
        );

        let rewards = Tensor::<B, 1, Float>::from_floats(
            indices.iter().map(|i| self.rewards[i]).collect::<Vec<f32>>().as_slice(),
            &device,
        );

        let mut batch_images = Vec::with_capacity(batch_size * IMAGE_SIZE);

        for i in indices.iter() {
            let start = i * IMAGE_SIZE;
            let end = start + IMAGE_SIZE;
            batch_images.extend_from_slice(&self.next_images[start..end]);
        }

        let next_images: Tensor<B, 4> = Tensor::<B, 4>::from_data(
            TensorData::new(
                batch_images,
                [batch_size, 4, 84, 84],
            ),
            &device,
        );

        let dones = Tensor::<B, 1, Float>::from_floats(
            indices.iter().map(|i| self.dones[i]).collect::<Vec<f32>>().as_slice(),
            &device,
        );

        RetroBatch { images, actions, rewards, next_images, dones }
    }

    pub fn learning_is_ready(&self) -> bool {
        self.len >= self.min_samples
    }
}
