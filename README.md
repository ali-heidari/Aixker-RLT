# Aixker-RLT
Lightweight reinforcement learning trainer in Rust — modular, fast, and designed for adaptive AI systems. Part of the AIXKER ecosystem.

![AIxKer Logo](docs/aixker-rlt-banner.png)

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Rust](https://img.shields.io/badge/Rust-2021_edition-orange)

## Overview

Aixker-RLT (AIXKER Reinforcement Learning Trainer) is a minimal, modular, and production-ready framework for training and evaluating reinforcement learning (RL) models — designed to be embedded in real systems like load balancers, protocol agents, and distributed services.

It’s part of the **AIXKER ecosystem**, an initiative to bring **AI-native intelligence** into modern infrastructure — but AIXKER-RLT can also be used as a **standalone RL framework** in any pr[...]

---

## 🚀 Features

- ⚡ **Lightweight Core** — Simple modular design with minimal dependencies.
- 🧩 **Customizable Inputs/Outputs** — Accept any number of parameters for flexible training.
- 🔁 **Continuous Learning** — Supports online and on-device adaptation in production.
- 🧠 **Multi-Agent Ready** — Built to scale across multiple agents or nodes.
- 📊 **Logging & Metrics Hooks** — Integrates easily with external metric collectors.
- 🖥️ **CPU & GPU compute** — Pure-Rust `ndarray` backend by default; optional GPU backend via `wgpu` (`backend = "Gpu"`).

---

## Installation

Use it as a Git dependency in your `Cargo.toml`:

```toml
[dependencies]
aixker-rlt = { git = "https://github.com/ali-heidari/Aixker-RLT" }
```

Or build from source:

```bash
git clone https://github.com/ali-heidari/Aixker-RLT.git
cd Aixker-RLT
cargo build --release
```

---

## 🧬 Example Usage

```rust
use std::sync::Arc;

use aixker_rlt::configurations::Configurations;
use aixker_rlt::{initialize, node::Node, RunningMode};

#[tokio::main]
async fn main() {
    env_logger::init(); // any `log` backend; the crate logs training stats at `info`

    // Load configuration once at startup
    initialize(Arc::new(Configurations {
        input_number: 8,
        output_number: 3,
        log_interval: 50,
        ..Default::default()
    }));

    Node::start(
        // Feature provider: returns the current state as a feature vector
        |_lowest_state| vec![0.0; 8],
        // Reward function: (features, action, step) -> (reward, success)
        |_features, action, _step| (if action == 1 { 0.01 } else { -0.01 }, action == 1),
        RunningMode::Training,
        "my-model.json",
    )
    .await;
}
```

### One configuration per node

`initialize()` sets one process-wide configuration and can only be called
once. To run several agents with different configurations in one process
(or in tests), pass the configuration to the node instead:

```rust
let config = Arc::new(Configurations { /* ... */ });
let node = Node::start_with_config(config, input_bearer, reward_fn, RunningMode::Training, "my-model.json").await;
// `node` is returned when the loop ends, e.g. to call `infer_action(&node, &features)`.
```

`Node::new_with_config` does the same for a node you drive yourself.

### Running modes

`Node::start` takes a `RunningMode`:

| Mode | Description |
| ---- | ----------- |
| `Training` | Continuous online training |
| `TrainingWithInterval` | Training on a fixed interval (`interval_secs`) |
| `Infer` | Inference only — load a trained model and produce actions |

---

## ⚙️ CPU and GPU usage

### CPU (default — fully supported)

By default, training and inference run on the **CPU** through a pure-Rust
`ndarray` implementation. No special flags or features are needed:

```bash
cargo build --release
```

- The async runtime is Tokio's multi-threaded scheduler, so training workers
  use multiple cores out of the box.
- To limit core usage, set the Tokio worker thread count in your host
  application (e.g. `#[tokio::main(worker_threads = 4)]`).

### GPU (selectable via configuration)

A GPU compute backend based on **`wgpu`** (Vulkan / Metal / DX12) runs the
matrix work — the forward pass and backpropagation — as compute shaders. The
small per-batch steps (loss gradient and SGD update) run on the CPU and
the new weights are uploaded back to the GPU. It works on any GPU supported by `wgpu`
(NVIDIA, AMD, Intel, Apple Silicon) without CUDA.

Select it with the `backend` field of `Configurations` (TOML or struct):

```toml
backend = "Gpu"   # default: "Cpu" — the field is optional
```

Behavior:

- **CPU is always the default**; omitting `backend` keeps existing configs
  working unchanged.
- If no compatible GPU adapter is found at startup, the library logs a
  warning and **falls back to CPU automatically** — it never crashes.
- Model checkpoints stay backend-agnostic JSON: the CPU copy of the weights
  is always current, so you can train on GPU and infer on CPU (or vice
  versa) with the same file.

The compute shaders live in `src/*.wgsl` (`forward`, `grad-weights`,
`grad-bias`, `grad-hidden`), orchestrated by `src/models/gpu_model.rs`. GPU/CPU parity is covered by the
`gpu_matches_cpu` unit test (`cargo test gpu_matches_cpu`).

> Note: for very small networks (e.g. 8×16×3) CPU is usually faster — GPU
> dispatch overhead dominates. The GPU backend pays off as `hidden_layers`,
> `input_number`, and `batch_size` grow.

---

## Configuration

Configuration is provided at startup via `initialize(...)` (or per node with `Node::start_with_config`) as a `Configurations` struct, deserializable with `serde`. Every field marked optional can be left out of a TOML/JSON config:

| Field | Description |
| ----- | ----------- |
| `input_number` | Number of input features |
| `output_number` | Number of possible actions |
| `hidden_layers` | Hidden layer size |
| `batch_size` / `total_batches` | Training batch settings |
| `reply_capacity` | Replay buffer capacity |
| `interval_secs` | Interval for `TrainingWithInterval` mode |
| `model_name` | Checkpoint file name |
| `log_interval` | Log one line of training statistics every N batches (`0` = off). See [Training diagnostics](#training-diagnostics). |
| `mode` | `RunningMode` (`Training`, `TrainingWithInterval`, `Infer`) |
| `backend` | Optional `ComputeBackend` (`Cpu` default, `Gpu`) |
| `seed` | Optional `u64`. Seeds weight init, exploration, action sampling and replay sampling. Unset = fresh randomness each run. See [Reproducible runs](#reproducible-runs). |
| `reward_shaping` | Optional `bool`, default `false`. Rewrites rewards with a built-in bonus/penalty based on per-action success rates and frequencies. The penalty is up to about 0.5, so leave it off unless your rewards are on the order of ±1. |
| `learning_rate` | Optional `f32`, default `0.05`. SGD step size, applied once per batch to the batch-averaged gradient. |
| `max_grad_norm` | Optional `f32`, default `1.0`. Clips the gradient's global L2 norm; `0` disables clipping. |
| `entropy_coef` | Optional `f32`, default `0.1`. Entropy bonus: keeps the policy from becoming confident before the rewards support it, so it doesn't lock onto noise. `0` disables it. |
| `epsilon_start` / `epsilon_end` | Optional, default `0.3` / `0.05`. Random-action rate during training, moved linearly from start to end. |
| `temperature_start` / `temperature_end` | Optional, default `1.5` / `1.0`. Sampling temperature during training, same schedule. |
| `exploration_batches` | Optional, default `500`. Batches over which ε and temperature go from start to end. They stay at the end values afterwards. |
| `infer_temperature` | Optional, default `0.5`. Temperature for `infer_action` and `Infer` mode. No random actions at inference. `0` = always the top action (greedy). |

`Configurations` implements `Default`, so in Rust you only need to set what
you change: `Configurations { input_number: 8, output_number: 3, ..Default::default() }`.

### How it learns

The learner is a one-step policy gradient (REINFORCE for contextual
bandits): each action is scored by the reward returned for it, with no
multi-step returns. Per batch, rewards are turned into advantages
(`(r - batch mean) / batch std`), so only *relative* reward matters and the
reward scale does not change the step size. Replayed experiences are
reweighted by `min(π(a|s) / b, 2)` toward the current policy, where `b` is
the probability the action had when it was taken.

### Reproducible runs

Set `seed = 42` (any `u64`) to make a run repeatable. Each source of
randomness gets its own generator, derived from the seed and a stream id:
weight init, action selection (ε and sampling), and each training worker's
replay sampling. Same seed + same inputs + same rewards = identical weights,
on the same build and platform.

`Training` mode is fully repeatable. `TrainingWithInterval` is not: the
worker trains on a timer, so how many experiences exist at each batch
depends on timing. Loading an existing checkpoint also changes the start
point, so delete it (or use a new `model_name`) for a clean repeat.

### Training diagnostics

Every `log_interval` batches the trainer logs one line through the `log`
crate at `info` level (the host application sets up the logger):

```text
batch 200: reward mean=0.00120 std=0.01010 | loss=0.00031 grad_norm=0.04210 | eps=0.050 temp=1.000 | policy=[0.334, 0.331, 0.335] entropy=1.0985 | taken=[0.328, 0.344, 0.328]
```

| Field | Meaning |
| ----- | ------- |
| `reward mean` / `std` | Raw reward from your callback, over the batch |
| `loss` | Policy-gradient loss of the batch |
| `grad_norm` | Global L2 norm of the gradients, before clipping |
| `eps` / `temp` | Current random-action rate and sampling temperature |
| `policy` | Mean probability of each action over the batch's inputs |
| `entropy` | Mean policy entropy in nats (`ln(n_actions)` = uniform, `0` = one action) |
| `taken` | Share of each action in the training batch |

Collapse shows up as `entropy` falling toward 0 while `policy` piles onto one
action, independent of `reward mean`. The latest batch's values are also
available as `node.last_stats` (`Option<TrainStats>`).

### Recommended settings for small, noisy rewards

For rewards like trading returns (around ±0.001-0.01 per step, mostly
noise), start from the defaults and check these:

```toml
reward_shaping = false        # default; the built-in penalty is ~100x a trading reward
learning_rate = 0.05          # default; lower to 0.02 if grad_norm sits at max_grad_norm
max_grad_norm = 1.0
entropy_coef = 0.1            # keep >= 0.03 so noise can't make the policy confident
epsilon_start = 0.3
epsilon_end = 0.05
temperature_start = 1.5
temperature_end = 1.0
exploration_batches = 1000    # about a quarter of total_batches for noisy rewards
infer_temperature = 0.5       # 0 = greedy, which turns an unsure policy into one fixed action
batch_size = 128              # bigger batches average out more noise per step
reply_capacity = 4096         # ~30 batches; older data is only partly corrected
seed = 42                     # for experiments you want to repeat
log_interval = 50
```

- **Reward scale doesn't matter.** Rewards are normalized per batch, so
  ±0.002 learns like ±1. Rewards are still clamped to `[-1, 1]`.
- **Scale your inputs.** Keep features roughly in `[-1, 1]` or z-scored. Raw
  prices make the hidden layer and the gradients huge.
- **Watch the log.** If `entropy` drops toward 0 while `reward mean` is flat,
  the policy is fitting noise: raise `entropy_coef` or lower `learning_rate`.
- **Check on a known problem first.** `tests/toy_suite.rs` has bandit,
  small-signal and pure-noise problems with known answers
  (`cargo test --release --test toy_suite -- --ignored --nocapture`).

- **Inputs:** Any numeric parameters representing the system/environment state
- **Outputs:** Actions or decisions produced by the RL model
- **Logging:** Supports integration with external metric collectors (Falcon Metrics, Prometheus, etc.)

---

## Roadmap

- [x] Dynamic hyperparameter tuning
- [x] Improved sample efficiency for online learning
- [x] CLI for model training and exporting — see [RLT-CLI](https://github.com/ali-heidari/RLT-CLI)
- [x] Integration with AIXKER Metrics Collector
- [x] GPU compute backend (`wgpu` + WGSL shaders)
- [ ] Release first stable API
- [ ] Define inputs

---

## Contributing

We welcome contributions!

1. Fork the repository
2. Submit pull requests for bug fixes or features
3. Open issues for bugs, discussions, or proposals

Please follow the [Rust community style guide](https://doc.rust-lang.org/1.0.0/style/) and add tests for new functionality.

---

## License

AIXKER-RLT is released under the **Apache 2.0 License**.
© 2026 Ali — part of the [AIXKER Project](https://github.com/AIXKER).

> Commercial licensing available through AIXKER for enterprise integration.

---

## Documentation

This repository includes a `docs/` folder containing readable Markdown documentation and an [index.md](docs/index.md) landing page:

- [architecture.md](docs/architecture.md)
- [standards.md](docs/standards.md)

## Standards

This project follows the `ai-agent-standards` conventions. The main AI agent guidance is documented in [.agent/agent-instructions.md](.agent/agent-instructions.md), and shared standard files are available at:

- [ali-heidari/ai-agent-standards](https://github.com/ali-heidari/ai-agent-standards)

## Related Projects / Ecosystem

- **[RLT-CLI](https://github.com/ali-heidari/RLT-CLI)** — command-line interface for training, inference, and export
- **[Aixker Agent](https://github.com/Aixker/aixker-agent)** — AI-native node agent using Aixker-RLT

---

## Contact / Community

- GitHub: [ali-heidari](https://github.com/ali-heidari)
- Email: [ali-heidari@outlook.com](mailto:ali-heidari@outlook.com)
- AIXKER: [https://github.com/AIXKER](https://github.com/AIXKER)
