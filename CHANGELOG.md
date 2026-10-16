# Changelog

All notable changes to this crate are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate
follows [Semantic Versioning](https://semver.org/) (0.x: a minor bump may
break).

## [Unreleased]

## [0.2.0] - 2026-10-16

This release fixes the learning bugs that made every trained agent collapse
to one fixed action. See [docs/learning-fixes-0.2.0.md](docs/learning-fixes-0.2.0.md)
for the analysis, test results and upgrade notes.

### Breaking

- **Reward shaping is off by default**, and the learner trains on the raw
  reward from your callback. Before, it was always on in
  `Training`/`TrainingWithInterval` and added a penalty of up to about 0.5
  to failed or over-used actions. That swamped small rewards and taught a
  single fixed action. Set `reward_shaping = true` to get the old behavior.
- **Advantages instead of raw rewards.** The policy gradient trains on
  `(r - batch mean) / batch std`. Only relative rewards within a batch
  matter now, and adding a constant to every reward changes nothing.
- **Averaged, clipped gradients and a constant learning rate.** Gradients
  are averaged over the batch instead of summed, and clipped to
  `max_grad_norm`. The learning rate comes from `learning_rate` (default
  `0.05`). Before, it was hard-coded to `1e-5` with summed gradients
  (effectively `6.4e-4` per sample) and decayed toward `1e-6` by a rule that
  ran on every step.
- **Fixed exploration schedule.** ε and temperature move linearly from
  start to end values over `exploration_batches`. Before, they were tuned
  automatically and jumped to their caps (ε 0.7, temperature 7) whenever the
  policy preferred one action, so a learned policy was mostly ignored.
- **`infer_action` no longer explores.** It never takes random actions and
  samples at `infer_temperature` (default 0.5; `0` = greedy). `Infer` mode
  acts the same way. Before, it used the node's current ε and temperature
  (up to 0.7 and 7 on a training node; 0.01 and 0.5 in `Infer` mode).
- **`Configurations` has new public fields.** Code that builds it with a
  struct literal must add them or end with `..Default::default()`.
  TOML/JSON configs are unaffected: every new field is optional.
- **`Node::loss` and `Node::batch_history` hold the real loss.** Before,
  they held the batch's mean reward under the name "loss".

### Added

- Configuration options, all optional:
  - `seed: Option<u64>`. When set, weight init, ε exploration, action
    sampling and replay sampling each use a `StdRng` derived from the seed
    and a stream id (one per worker), so `Training` runs repeat exactly.
  - `reward_shaping: bool` (default `false`).
  - `learning_rate` (default `0.05`) and `max_grad_norm` (default `1.0`,
    `0` = off): gradient clipping by global L2 norm.
  - `entropy_coef` (default `0.1`): entropy bonus `-β·H(π)` in the loss. It
    stops the policy from becoming confident on noise, or before a weak
    signal is clear.
  - `epsilon_start`/`epsilon_end` (0.3 → 0.05),
    `temperature_start`/`temperature_end` (1.5 → 1.0),
    `exploration_batches` (500) and `infer_temperature` (0.5).
- Training diagnostics: every `log_interval` batches the worker logs, at
  `info` level, raw reward mean/std, loss, gradient norm (before clipping),
  ε, temperature, the policy's mean action distribution and its entropy.
  The same values are kept in `Node::last_stats` as a public `TrainStats`.
- `Node::new_with_config` and `Node::start_with_config` take an
  `Arc<Configurations>`, so nodes with different configurations can run in
  one process. `start_with_config` returns the node when its loop ends.
  `initialize()` + `Node::start` work as before.
- `Node::config()`, `Node::weights()` (a flat copy of all parameters),
  `impl Default for Configurations`, `Configurations::exploration_at(batch)`.
- Tests: toy problems with known answers (`tests/toy_suite.rs`, run with
  `cargo test --release --test toy_suite -- --ignored`), a determinism test,
  a finite-difference gradient check, and more unit tests.

### Fixed

- Replayed experiences are reweighted by a truncated importance weight
  `min(π(a|s) / b, 2)`, where `b` is stored when the action is taken. Before,
  old and ε-random actions were trained on as if the current policy had
  chosen them.
- The policy gradient uses the full softmax gradient `-r·(onehot(a) - π)`.
  Before, only the chosen action's term was applied, so the update was not
  the gradient of any loss.
- Experiences with a non-finite reward or input are dropped with a warning.
  Before, one NaN reward passed the `[-1, 1]` clamp and turned every weight
  into NaN.
- An update with a non-finite gradient is skipped with a warning.
  `sanitize()` and the uniform-action fallback now warn instead of silently
  resetting weights or acting at random.
- Training works with any number of actions. The per-batch report assumed
  exactly 3 and panicked otherwise. Its per-action lines are now labelled
  "Action 0", "Action 1", ... instead of "High/Normal/Low pressure".
- The checkpoint is saved once more when the training loop ends. Before,
  the last batch's weights were never written.
- `TrainingWithInterval` yields to the runtime after each step, so its
  background worker also runs on a single-threaded tokio runtime.
- Action selection reads ε and temperature with read locks instead of write
  locks.

### Changed

- `log_interval` is now used: it is the number of batches between
  statistics log lines (`0` turns the log off). It was ignored before.
- `Node::set_default_factors` sets the schedule's start values for training
  modes, and ε 0 with `infer_temperature` for `Infer`.
- No `thread_rng()` is used on the learning path anymore.
- Ties in the "least-used action" passed to the input callback are broken
  by action id. Before, they depended on `HashMap` order and changed between
  runs.
- The GPU backend runs only the forward pass and backpropagation. The loss
  gradient, clipping and SGD step run on the CPU for both backends. The
  `softmax`, `reinforce` and `weight-update` shaders were removed.
- A missing or unreadable checkpoint no longer creates an empty file at
  startup. The node starts with fresh weights and writes the file on its
  first save.

## [0.1.0]

- Initial release.
