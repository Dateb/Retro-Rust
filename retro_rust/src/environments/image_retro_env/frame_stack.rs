use arraydeque::{ArrayDeque, Wrapping};

const STACK_SIZE: usize = 4;

#[derive(Debug)]
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
        self.frames.push_back(frame);
    }

    pub fn stacked(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.frames.len() * self.frame_size);

        let n = self.frames.len();

        for (i, frame) in self.frames.iter().enumerate() {
            if i == n - 1 && n >= 2 {
                out.extend_from_slice(&self.max_last_two());
            } else {
                out.extend_from_slice(frame);
            }
        }
        out
    }

    pub fn clear(&mut self) {
        let zero_frame = vec![0.0; self.frame_size];

        for _ in 0..STACK_SIZE {
            self.push(zero_frame.clone());
        }
    }

    /// Returns the elementwise max over the last 2 frames
    fn max_last_two(&self) -> Vec<f32> {
        // Make sure we have at least 2 frames
        if self.frames.len() < 2 {
            return self.frames.back().cloned().unwrap_or_else(|| vec![0.0; self.frame_size]);
        }

        let last = self.frames.get(self.frames.len() - 1).unwrap();
        let second_last = self.frames.get(self.frames.len() - 2).unwrap();

        last.iter()
            .zip(second_last.iter())
            .map(|(a, b)| a.max(*b))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_pooling_correct_values() {
        let frame_stack = create_frame_stack();

        assert_eq!(
            frame_stack.stacked(),
            vec![0.0, 0.0, 200.0, 100.0, 20.0, 10.0, 44.0, 10.0]
        );
    }

    #[test]
    fn clear_frame_stack_correct_values() {
        let mut frame_stack = create_frame_stack();

        frame_stack.clear();

        assert_eq!(
            frame_stack.stacked(),
            vec![0.0; 8]
        );
    }

    fn create_frame_stack() -> FrameStack {
        let mut frame_stack = FrameStack::new(2);

        // Put some frame in there that whose values will not get pooled
        let frame = vec![200.0, 100.0];
        frame_stack.push(frame);

        // Push 2 final frames for max pooling
        let frame = vec![20.0, 10.0];
        frame_stack.push(frame);

        let frame = vec![44.0, 5.0];
        frame_stack.push(frame);

        frame_stack
    }
}