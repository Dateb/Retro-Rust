<p align="center">
  <img src="images/logo.png" width="280">
</p>

<p align="center">
  A rust-based <a href="https://github.com/openai/retro">Gym Retro</a> API
</p>

# Retro Rust

- Retro game emulator environments designed for reinforcement learning experiments.
- Support for classic games like *Super Mario World* and *Donkey Kong*:

<p align="center">
  <img src="images/demo.gif" width="220" style="margin-right: 20px;">
</p>

- <ul style="list-style: none; padding-left: 0;">
      <li>Designed for native Rust ML workflows:</li>
      <ul style="list-style: none; padding-left: 1em;">
        <li>➜ Ultra-fast training loops</li>
        <li>➜ Zero Python/C++ bindings</li>
      </ul>
  </ul>




## Getting started

A simple starting point may look like this:

    let game_name = "Airstriker";
    let platform = Platform::Genesis;
    let save_state_name = String::from("Level1.state");
    let frame_skip = 4;

    let env = ImageRetroEnv::new(game_name, platform, save_state_name, frame_skip);

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

- Vectorized environment support
- Parallelized step rollouts
- RAM based observations
