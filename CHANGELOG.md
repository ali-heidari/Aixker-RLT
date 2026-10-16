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

### Changed

- **Breaking (behavior):** gradients are averaged over the batch instead of
  summed, and clipped to `max_grad_norm`. The learning rate is now a
  constant from the config (default `0.05`). Before, it was hard-coded to
  `1e-5` with summed gradients (effectively `6.4e-4` per sample) and decayed
  toward `1e-6` by a rule that ran on every step.
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
