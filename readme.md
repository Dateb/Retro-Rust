# Retro Rust - An [OpenAI Retro](https://github.com/openai/retro) API written in Rust

This repository provides rust native API support for the well known
retro games environment zoo; allowing blazingly fast rust models to
learn games on Super Nintendo, Game boy and many more platforms.

## Getting started

A simple starting point may look like this:

    let game_path = var("RETRO_GAME_PATH")
        .expect("RETRO_GAME_PATH environment variable not set");

    let save_state_name = var("SAVE_STATE_NAME")
        .expect("SAVE_STATE_NAME environment variable not set");

    let frame_skip = 4;

    let env = RetroEnv::new(game_path, &save_state_name, frame_skip);

    let policy = ... // Initialise your policy

    for _ in 1..num_episodes {
        let mut step_info = env.reset();
        let mut image = step_info.observation;
        let mut next_action = policy.get_next_action(image, env.num_actions(), &self.device);

        while !step_info.is_done() {
            step_info = env.step(next_action);

            let next_image = step_info.observation;
            let reward = step_info.reward;
            let done = step_info.is_done;

            // Do something with env feedback
        }
    }

## Environment Overview

Environment API is kept close to python environments:

| Function  | Parameters                                                 | Return Value(s)         | Note                                                           |
|-----------|------------------------------------------------------------|-------------------------|----------------------------------------------------------------|
| new       | `game_path: String, save_state_name: &str, frame_skip: u8` | `RetroEnv`              |
| step      | `action: usize`                                            | `(Vec<f32>, f32, bool)` | `action` is a discrete action |
| reset     | `-`                                                        | `Vec<f32>`              |

## Example Benchmark

To evaluate runtime improvements, we benchmarked training performance using Deep Q-Networks (DQN) on the Airstriker (Sega Genesis) environment.

The figure below shows the runtime behavior during DQN training in the game Airstriker on Sega Genesis:

![Runtime image](images/exec_time.png)

For fair comparisons, hyperparameters are replicated across both runs. These are the following:

### Benchmark setup

| Hyperparameter            | Value     |
|---------------------------|-----------|
| Replay Buffer size        | `100.000` |
| Learning starts at sample | `10.000`  |
| Target update frequency   | `5.000`   |
| N steps per train         | `4`       |
| Discount factor           | `0.99`    |
| Batch size                | `128`     |
| Optimizer                 | `Adam`    |
| Learning rate             | `1e-4`    |
| Number of stacked frames  | `4`       |
| Number of skipped frames  | `4`       |

Network architecture: Standard DQN with CNN, see [Mnih et al., 2015](https://arxiv.org/abs/1312.5602)

## Planned features

From top to bottom in priority, the following features are planned
to be added soon:

- Gameplay video recorder
- Vectorized environment support
- Parallelized step rollouts
- RAM based observations
