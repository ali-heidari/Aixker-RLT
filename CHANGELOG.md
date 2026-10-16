# Changelog

All notable changes to this crate are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate
follows [Semantic Versioning](https://semver.org/) (0.x: a minor bump may
break).

## [Unreleased]

### Added

- Training diagnostics: every `log_interval` batches the worker logs raw
  reward mean/std, loss, gradient norm, ε, temperature, the policy's mean
  action distribution and its entropy at `info` level. The same values are
  stored in `Node::last_stats` as a public `TrainStats`.

- `Node::new_with_config` and `Node::start_with_config` take an
  `Arc<Configurations>` directly, so several nodes with different
  configurations can run in one process. `start_with_config` returns the
  node when its loop ends. `Node::config()` returns a node's configuration.
  `initialize()` + `Node::start` work as before.

- `Configurations::seed: Option<u64>`. When set, weight init, ε
  exploration, action sampling and replay sampling each use a `StdRng`
  derived from the seed and a stream id (one per worker), so `Training`
  runs repeat exactly. Unset keeps fresh randomness.
- `impl Default for Configurations`, for `..Default::default()`.
- `Node::weights()`: a flat copy of all model parameters.

- `Configurations::reward_shaping: bool` (default `false`).
- `Configurations::learning_rate` (default `0.05`) and
  `Configurations::max_grad_norm` (default `1.0`, `0` = off): gradient
  clipping by global L2 norm.
- Exploration schedule: `epsilon_start`/`epsilon_end` (0.3 → 0.05),
  `temperature_start`/`temperature_end` (1.5 → 1.0) and
  `exploration_batches` (500). Inference temperature: `infer_temperature`
  (0.5, `0` = greedy).
- `Configurations::exploration_at(batch)`: the (ε, temperature) of the
  schedule after `batch` batches.

### Changed

- **Breaking (behavior):** gradients are averaged over the batch instead of
  summed, and clipped to `max_grad_norm`. The learning rate is now a
  constant from the config (default `0.05`). Before, it was hard-coded to
  `1e-5` with summed gradients (effectively `6.4e-4` per sample) and decayed
  toward `1e-6` by a rule that ran on every step.
- **Breaking (behavior):** the policy gradient trains on normalized
  advantages, `(r - batch mean) / batch std`, instead of the raw reward.
  Only relative rewards within a batch matter now, and adding a constant
  to every reward changes nothing.
- **Breaking (behavior):** exploration follows a fixed linear schedule.
  Before, ε and temperature were tuned automatically and jumped to their
  caps (ε 0.7, temperature 7) whenever the policy preferred one action, so a
  learned policy was mostly ignored.
- **Breaking (behavior):** `infer_action` no longer explores. It never takes
  random actions and samples at `infer_temperature` (default 0.5). `Infer`
  mode acts the same way. Before, it used the node's current ε and
  temperature (up to 0.7 and 7 on a training node; 0.01 and 0.5 in `Infer`
  mode).
- `Node::set_default_factors` now sets the schedule's start values for
  training modes, and ε 0 with `infer_temperature` for `Infer`.
- **Fix:** replayed experiences are reweighted by a truncated importance
  weight `min(π(a|s) / b, 2)`. Before, old and ε-random actions were
  trained on as if the current policy had chosen them.
- **Fix:** the policy gradient now uses the full softmax gradient
  `-r·(onehot(a) - π)`. Before, only the chosen action's term was applied,
  so the update was not the gradient of any loss.

- **Breaking (behavior):** reward shaping is now **off by default**, and the
  learner trains on the raw reward from your callback. Before, it was always
  on in `Training`/`TrainingWithInterval` and added a penalty of up to about
  0.5 to failed or over-used actions. That swamped small rewards and taught
  a single fixed action. Set `reward_shaping = true` to get the old
  behavior.

- **Breaking for struct literals:** `Configurations` has new public fields.
  Code that builds it with a struct literal must add them or end with
  `..Default::default()`. TOML/JSON configs are unaffected: new fields are
  optional.
- No `thread_rng()` is used on the learning path anymore.
- Ties in the "least-used action" passed to the input callback are broken
  by action id; before, they depended on `HashMap` order and changed between
  runs.

- `log_interval` is now used: it is the number of batches between
  statistics log lines (`0` turns the log off). It was ignored before.
- `Node::loss` and `Node::batch_history` now hold the batch's
  policy-gradient loss. Before, they held the batch's mean reward under
  the name "loss".
- The GPU backend now runs only the forward pass and backpropagation; the
  loss gradient and the SGD step run on the CPU for both backends. The
  `softmax`, `reinforce` and `weight-update` shaders were removed.
- A missing or unreadable checkpoint no longer creates an empty file at
  startup; the node starts with fresh weights and writes the file on its
  first save.
