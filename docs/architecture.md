# AION-RLT Architecture

AION-RLT is a lightweight reinforcement learning trainer designed for embeddable applications — adaptive load balancers, protocol agents, trading bots, or any system that maps numeric features to discrete actions. It is a reusable RL engine, not specific to any domain.

## Design philosophy

- **Minimal & modular** — single responsibility per component
- **Configurable** — input/output dimensions, batch size, hyperparameters all tunable
- **Embeddable** — no opinions on what the application does; just trains & infers
- **Async-ready** — Tokio integration for background training loops

## Core modules

- `src/lib.rs` — library entrypoint, `CONFIG` singleton, `initialize()`
- `src/node.rs` — main orchestrator, model lifecycle, and training state
- `src/model.rs` — neural network implementation and policy gradient updates
- `src/worker.rs` — async training loop and batch processing
- `src/reply_buffer.rs` — experience replay buffer
- `src/infer_action.rs` — inference and action selection logic
- `src/experience.rs` — `Experience` struct definition
- `src/configurations.rs` — `Configurations` struct and config loading
- `src/footstep/` — internal diagnostics module
- `src/models/gpu_model.rs`, `src/*.wgsl` — GPU compute backend (selected via `backend = "Gpu"`)

## Node (`node.rs`)

Central orchestrator managing the RL lifecycle for a single agent.

- Holds `Model` (network weights) and `ReplayBuffer` (experience storage)
- Manages `epsilon` (exploration) and `temperature` (softmax control)
- Tracks training metrics: `batch_history`, `success_rate_mean`, divergence/stuck flags
- Supports `RunningMode`: `Infer`, `Training`, `TrainingWithInterval`

Entrypoint:

```rust
Node::start(
    input_bearer,                  // Fn(u32) -> Vec<f32>
    compute_reward_with_success,   // Fn(&Vec<f32>, u32, u32) -> (f32, bool)
    mode,                          // RunningMode
    model_name,                    // checkpoint file name
).await;
```

`Node::start_with_config(config, ...)` and `Node::new_with_config(config, ...)` take an `Arc<Configurations>` instead of reading the global one set by `initialize()`. Each node keeps its own configuration (`node.config()`), so several differently configured nodes can share a process. `start_with_config` returns the node when its loop ends.

All shared state uses `Arc<RwLock<T>>`: reader threads (inference) don't block each other, the writer thread (training) gets exclusive access when updating.

## Model (`model.rs`)

Two-layer neural network with ReLU hidden activation:

```text
input (f32) → w1 (input_number × hidden_layers) + b1
           → ReLU
           → w2 (hidden_layers × output_number) + b2
           → Softmax (temperature-controlled)
           → action probabilities
```

- **Training algorithm**: REINFORCE policy gradient — ∇J = Σ ∇log(π(a|s)) · R(a)
- **Checkpoints**: model serialized to JSON (`{w1, b1, w2, b2}` via serde)
- Learning rate is held in `Node.lr` (`RwLock<f32>`) and adjustable at runtime

## ReplayBuffer (`reply_buffer.rs`)

Experience storage with uniform random sampling.

- Stores `Experience { features, action, latency_ms, reward, success, timestamp_ms }`
- Samples random batches for training (no prioritization yet)
- Enforces `reply_capacity`: drops oldest entries when full

Separating the buffer decorrelates experience order (learning stability) and decouples training rate from experience generation rate.

## Worker (`worker.rs`)

Async background training loop:

```text
loop {
  sleep(interval)
  if buffer has ≥ batch_size experiences:
    sample batch → reinforce → save checkpoint
    update batch_history, check convergence/divergence
}
```

The worker runs in the background; the application decides when to stop (counter reaches `total_batches`).

## Inference (`infer_action.rs`)

Action selection via softmax + epsilon-greedy:

1. Forward pass: `logits = model.forward(features)`
2. Softmax with temperature: `probs = softmax(logits / temperature)`
3. Epsilon-greedy: with probability ε pick a random action, otherwise the most probable one
4. Returns `(action_idx, logits, probabilities)`

Pure greedy gets stuck in local optima; pure random never exploits learned knowledge. Epsilon balances both, and the Node adjusts it dynamically.

## Configuration (`configurations.rs`)

Provided at startup via `initialize(...)`:

| Field | Meaning |
| ----- | ------- |
| `interval_secs` (u64) | Worker loop interval for `TrainingWithInterval` |
| `batch_size` (u32) | Experiences per training iteration |
| `total_batches` (usize) | Training termination point |
| `input_number` (usize) | Feature vector dimension |
| `output_number` (usize) | Action space size |
| `hidden_layers` (usize) | Hidden layer neurons |
| `reply_capacity` (usize) | Max replay buffer size |
| `model_name` (String) | Checkpoint file name |
| `log_interval` (u64) | Batches between training-statistics log lines (`0` = off) |
| `mode` (RunningMode) | `Training`, `TrainingWithInterval`, or `Infer` |
| `backend` (ComputeBackend, optional) | `Cpu` (default) or `Gpu` |

**Critical contract:** `input_number` must equal the feature vector length and `output_number` must equal the action space size — a mismatch panics at model initialization.

## Common pitfalls

- **Feature dimension mismatch** — `input_number ≠ features.len()` → panic
- **Reward range** — rewards are expected clamped to `[-1.0, 1.0]`
- **Buffer underflow** — training is skipped gracefully until `batch_size` experiences exist
- **Checkpoint conflicts** — multiple agents writing the same `model_name` corrupt the file; use an ID suffix
- **Temperature too low** — nearly deterministic, little exploration
- **Epsilon too high** — mostly random actions, slow learning

## Extending AION-RLT

- **New training algorithms**: modify `Model::reinforce()`, update the loss calculation, document reward expectations
- **Different input/output sizes**: already supported via config — no code changes
- **Custom metrics**: read `batch_history` / `success_rate_mean` through the `RwLock`s and export to your collector

## GPU backend

Setting `backend = "Gpu"` in the configuration routes the forward pass and backpropagation in `Model` through `GpuModel` (`src/models/gpu_model.rs`), which runs the WGSL compute shaders via `wgpu` (Vulkan/Metal/DX12 — no CUDA required):

- `forward.wgsl` — generic linear layer (dispatched per layer; ReLU flag in uniforms)
- `grad-weights.wgsl` / `grad-bias.wgsl` / `grad-hidden.wgsl` — backprop gradients

The loss gradient with respect to the logits and the SGD step run on the CPU, shared by both backends. After each update the CPU weights are uploaded to the GPU buffers, so the CPU copy is always current and JSON checkpoints are backend-agnostic. `Model::sanitize` re-uploads the cleaned weights. If no GPU adapter is available, the library warns and falls back to CPU. Parity with the CPU path is asserted by the `gpu_matches_cpu` test.

For small networks the CPU is typically faster; the GPU pays off as `hidden_layers`, `input_number`, and `batch_size` grow.

## Integration contract

**AION-RLT provides:** action probabilities and predictions from features (`Vec<f32>`); training, model persistence, and the exploration/exploitation tradeoff.

**The application provides:** feature extraction, domain-specific reward calculation, and configuration tuning.
