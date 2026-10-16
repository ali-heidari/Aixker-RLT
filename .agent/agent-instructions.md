# AION-RLT — Agent Instructions

This is the master agent instruction file for this repository, per the
`ai-agent-standards` standard. It MUST live at `.agent/agent-instructions.md`.
Any other agent instruction files MUST be placed inside the `.agent/` folder.

The canonical base conventions are copied under [`.agent/base/`](base/):
`assistant-conventions.md`, `code-conventions.md`, `cicd-conventions.md`,
`commit-conventions.md`, `docs-conventions.md`, `repository-conventions.md`.
Project additions are allowed; overrides of the base standard are not.

## Purpose

**AION-RLT** (also published as Aixker-RLT) is a general-purpose, lightweight
reinforcement learning framework in Rust, designed to be embedded in
applications such as adaptive load balancers, protocol agents, or trading
bots. It is a reusable RL engine, not specific to any domain.

## Big picture

- The canonical AI agent behavior is defined by `ai-agent-standards`
  (`instructions.md` + the six base convention files).
- This file is the project-specific wrapper for those standard instructions.
- Detailed component documentation lives in [`docs/architecture.md`](../docs/architecture.md).

## Key components

| Module | Responsibility |
| ------ | -------------- |
| `src/lib.rs` | Module declarations, `CONFIG` singleton, `initialize()` |
| `src/node.rs` | `Node` orchestrator, `RunningMode`, training state, exploration schedule, optional reward shaping |
| `src/model.rs` | Two-layer NN (ReLU hidden, softmax output), policy gradient with normalized advantages, truncated importance weights, entropy bonus, clipping; JSON checkpoints |
| `src/worker.rs` | Async background training loop (batch sampling, weight updates) |
| `src/reply_buffer.rs` | Replay buffer: capacity-bounded storage + uniform random sampling (caller-supplied RNG) |
| `src/infer_action.rs` | Action selection: ε + temperature sampling while training; `infer_action` uses no ε and `infer_temperature` |
| `src/experience.rs` | `Experience` struct: features, action, reward, raw_reward, behavior_prob, success, ... |
| `src/configurations.rs` | `Configurations` struct (`Default`, exploration schedule) |
| `src/diagnostics.rs` | `TrainStats`: per-batch statistics, logged every `log_interval` batches |
| `src/rng.rs` | Seeded RNG streams derived from `Configurations::seed` |
| `src/footstep/` | Internal diagnostics module |
| `src/models/gpu_model.rs` + `src/*.wgsl` | GPU compute backend (wgpu), enabled via `Configurations::backend = Gpu` |

## Public API contract

The entrypoint is `Node::start`:

```rust
pub async fn start<F, H>(
    input_bearer: F,                  // Fn(u32) -> Vec<f32>: provides feature vectors
    compute_reward_with_success: H,   // Fn(&Vec<f32>, u32, u32) -> (f32, bool): reward + success
    mode: RunningMode,                // Infer | Training | TrainingWithInterval
    model_name: &str,                 // checkpoint file name
)
```

Call `aixker_rlt::initialize(config)` exactly once before `Node::start`, or
pass a config per node with `Node::start_with_config` / `Node::new_with_config`.
All shared state (`model`, `buffer`, `epsilon`, `temperature`, `lr`) is behind
`Arc<RwLock<...>>` for concurrent inference/training access.

## Critical invariants

- `input_number` MUST match the feature vector length returned by the input
  bearer; `output_number` MUST match the action space size. Mismatch panics at
  model initialization.
- Rewards are expected in `[-1.0, 1.0]` (clamped in the worker). Their scale
  does not matter otherwise: advantages are normalized per batch.
- No `thread_rng()` on the learning path. All randomness goes through
  `src/rng.rs` streams so `seed` makes `Training` runs repeatable; the
  `determinism` test guards this.
- The toy suite (`cargo test --release --test toy_suite -- --ignored`) MUST
  pass on all seeds after any change to the learning algorithm.
- Training is skipped gracefully until the buffer holds `batch_size`
  experiences.
- Multiple agents MUST NOT share a checkpoint `model_name` (corruption risk —
  use an ID suffix).
- The compute backend defaults to CPU; GPU is opt-in via
  `Configurations::backend = Gpu` and MUST fall back to CPU (with a warning)
  when no adapter is available. The GPU runs only the forward pass and
  backpropagation; the loss gradient, clipping and SGD step are shared CPU
  code. GPU and CPU MUST stay numerically equivalent — the `gpu_matches_cpu`
  test guards this; keep it passing when touching `model.rs`,
  `gpu_model.rs`, or any `.wgsl` shader.

## Project-specific conventions

- Key files: `Cargo.toml`, `src/lib.rs`, `README.md`, `LICENSE`,
  `docs/index.md`.
- Do not invent build, packaging, or CI steps unless explicit manifest/build
  files exist in this repo.
- Prefer small, runnable examples and use exact file paths in code snippets.
- Dependencies: `ndarray`/`ndarray-rand` (math), `tokio` (async worker),
  `serde`/`serde_json` (checkpoints), `aion-math` (math utilities),
  `wgpu`/`pollster` (experimental GPU), `log` (logging — initialized by the
  host application).

## Files to inspect first

- `README.md`
- `.agent/agent-instructions.md` (this file)
- `.agent/base/assistant-conventions.md`
- `.agent/base/code-conventions.md`
- `.agent/base/cicd-conventions.md`
- `.agent/base/commit-conventions.md`
- `.agent/base/docs-conventions.md`
- `.agent/base/repository-conventions.md`
- `docs/architecture.md`

## Interactive workflow (from assistant-conventions)

1. State the overall task before starting work.
2. Explain each step clearly and concisely.
3. Before writing code, describe the exact code change planned.
4. Ask whether the developer understands or agrees before applying the change.
5. Apply the code only after the developer confirms, then proceed.

## Commit conventions (from commit-conventions)

- Format: `type(scope): short summary`, blank line, optional body, blank line,
  `Refs: #<issue>`.
- Allowed types: `feat`, `fix`, `chore`, `docs`, `ci`, `style`, `refactor`,
  `test`.

## Documentation requirements

- A `docs/` folder is required at the repository root.
- `docs/index.md` must exist and list links to the other Markdown files in
  `docs/`.
- When code or standards change, update `README.md` and the relevant Markdown
  files together.

## Agent behavior rules

- Do not invent project-specific build or CI steps unless you find explicit
  files such as `Cargo.toml`, `Makefile`, or `.github/workflows`.
- When the project rules are unclear, ask a clarifying question instead of
  guessing.

## Sync note

If the user requests `sync instructions`, re-read the canonical base convention
files from `ai-agent-standards`, then update this file (`.agent/agent-instructions.md`)
and the files under `.agent/base/` to remain aligned with the canonical guidance.
