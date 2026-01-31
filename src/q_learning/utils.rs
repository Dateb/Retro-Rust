#[macro_export]
macro_rules! timeit {
    ($label:expr, $block:block) => {{
        let start = std::time::Instant::now();
        let result = $block;
        println!("{} took {:?}", $label, start.elapsed());
        result
    }};
}

const AVERAGE_WINDOW_SIZE: usize = 100;

pub struct RollingAverage {
    buf: [f32; AVERAGE_WINDOW_SIZE],
    idx: usize,
    len: usize,
    sum: f32,
}

impl RollingAverage {
    pub fn new() -> Self {
        Self {
            buf: [0.0; AVERAGE_WINDOW_SIZE],
            idx: 0,
            len: 0,
            sum: 0.0,
        }
    }

    pub fn push(&mut self, value: f32) {
        if self.len == AVERAGE_WINDOW_SIZE {
            self.sum -= self.buf[self.idx];
        } else {
            self.len += 1;
        }

        self.buf[self.idx] = value;
        self.sum += value;
        self.idx = (self.idx + 1) % AVERAGE_WINDOW_SIZE;
    }

    pub fn average(&self) -> Option<f32> {
        (self.len > 0).then(|| self.sum / self.len as f32)
    }
}
