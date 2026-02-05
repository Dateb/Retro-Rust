pub struct StepInfo {
    pub observation: Vec<f32>,
    pub reward: f32,
    pub is_done: bool
}

pub trait RetroEnv {
    fn step(&mut self, action: usize) -> StepInfo;
    fn reset(&mut self) -> StepInfo;
    fn num_actions(&self) -> usize;
}