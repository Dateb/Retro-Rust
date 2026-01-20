use arraydeque::{ArrayDeque, Wrapping};

const STACK_SIZE: usize = 4;

pub struct FrameStack {
    frames: ArrayDeque<Vec<f32>, STACK_SIZE, Wrapping>,
    frame_size: usize,
}

impl FrameStack {
    pub fn new(frame_size: usize) -> Self {
        let zero_frame = vec![0.0; frame_size];

        let mut frames = ArrayDeque::new();
        for _ in 0..STACK_SIZE {
            ArrayDeque::<Vec<f32>, STACK_SIZE, Wrapping>::push_back(
                &mut frames,
                zero_frame.clone()
            );
        }

        Self { frames, frame_size }
    }
}

impl FrameStack {
    pub fn push(&mut self, frame: Vec<f32>) {
        debug_assert_eq!(frame.len(), self.frame_size);
        self.frames.push_back(frame);
    }

    pub fn stacked(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.frames.len() * self.frame_size);

        for frame in self.frames.iter() {
            out.extend_from_slice(frame);
        }

        out
    }

    pub fn clear(&mut self) {
        let zero_frame = vec![0.0; self.frame_size];

        for _ in 0..STACK_SIZE {
            self.push(zero_frame.clone());
        }
    }
}


