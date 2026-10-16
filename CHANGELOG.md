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

### Changed

- `log_interval` is now used: it is the number of batches between
  statistics log lines (`0` turns the log off). It was ignored before.
- `Node::loss` and `Node::batch_history` now hold the batch's
  policy-gradient loss. Before, they held the batch's mean reward under
  the name "loss".
- The GPU backend now runs only the forward pass and backpropagation; the
  loss gradient and the SGD step run on the CPU for both backends. The
  `softmax`, `reinforce` and `weight-update` shaders were removed.
