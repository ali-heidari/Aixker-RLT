# Learning fixes in 0.2.0

Version 0.1.0 made every trained agent collapse to one fixed action, such as
"always hold" or "always buy", whatever its inputs were. This page records
why it happened, what changed in 0.2.0, and how the fixes were checked. The
full list of changes is in [CHANGELOG.md](../CHANGELOG.md).

Line numbers below refer to commit `fa9d711`, the last 0.1.0 commit.

## The algorithm

The crate learns a **contextual bandit policy with REINFORCE**, a one-step
policy gradient. A two-layer network maps input features to one score
(logit) per action, and a softmax turns the scores into probabilities. Each
step, the agent picks an action, the reward callback scores it, and the
experience goes into a replay buffer. Every `batch_size` steps a batch is
drawn from the buffer and trained on. There are no multi-step returns: each
reward is credited only to the action just taken.

**One training step in 0.2.0:**

1. Turn rewards into advantages: `A = (r - batch mean) / batch std`.
2. Weight each sample by `ρ = min(π(a|s) / b, 2)`. `π(a|s)` is the current
   policy's probability for the action; `b` is the probability the action
   had when it was taken.
3. Minimize `-mean(ρ·A·log π(a|s) + β·H(π))`, where `H` is the policy entropy
   and `β` = `entropy_coef`.
4. Average the gradient over the batch, clip it to `max_grad_norm`, and take
   one SGD step of size `learning_rate`.

While training, the agent explores with ε-random actions and a sampling
temperature on a linear schedule. At inference it takes no random actions.

## What was wrong in 0.1.0

### Why agents collapsed

- **Reward shaping swamped the reward** (`src/node.rs:426-464`). It was always
  on in training and could not be turned off. Failed steps got a penalty of
  up to about 0.5, based only on per-action counters (success rate, how often
  the action was taken). A trading reward of 0.001-0.01 is 30-300× smaller.
  The penalty ignores the input, so the best the network could learn was one
  fixed action.
- **Exploration ran backwards** (`src/node.rs:374-412`). Once the policy
  preferred an action, a `diverging` flag sent ε straight to its cap of 0.7
  and the temperature to 7. What the agent learned barely changed its
  behavior.
- **No baseline, summed gradients** (`src/model.rs:127-175`). The raw reward
  was the only weight. With almost every shaped reward negative, weights
  drifted the same way every batch. Gradients were summed over the batch and
  never clipped. `sanitize()` reset any weight above 1e6 to 0 without a
  warning, which explains the "weights reach about 1e6" report.
- **No seed control.** `thread_rng()` drove weight init, action sampling and
  replay sampling, so no run could be repeated.

### Other bugs

- **Off-policy replay.** Old and ε-random actions from the buffer were trained
  on as if the current policy had chosen them.
- **Inference explored.** `infer_action` always applied ε and sampled from the
  softmax. It never took the argmax.
- **Wrong softmax gradient.** Only the chosen action's term `-r·(1 - p_a)` was
  applied, so the update was not the gradient of any loss.
- **Learning rate decayed per step.** A rule that was meant to run every 50
  batches ran on every step, and drove the rate toward 1e-6.
- **"Loss" was the mean reward,** so collapse never showed in the logs.
- **NaN rewards** passed the `[-1, 1]` clamp and turned every weight into NaN.
- **`report()` assumed 3 actions** and panicked with any other count.
- **The last batch was never saved** to the checkpoint.
- **`TrainingWithInterval` never yielded,** so on a single-threaded tokio
  runtime its worker never trained.

## What changed

| Problem | Fix | Commit |
| ------- | --- | ------ |
| Reward shaping | `reward_shaping` option, off by default | `d4777e4` |
| Summed, unclipped, partial gradient; per-step lr decay | Full softmax gradient, averaged and clipped; constant `learning_rate` | `77ed7de` |
| No baseline; off-policy replay | Normalized advantages; truncated importance weights | `fe4532f` |
| Exploration caps; exploring inference | Linear schedule; `infer_temperature` with no random actions | `be5e1c2` |
| Early over-confidence on weak or noisy signals | Entropy bonus `entropy_coef` | `1817b99` |
| No seed control | `seed`, one `StdRng` per stream | `6addcbc` |
| No visibility | Training statistics every `log_interval` batches | `404f84d` |
| `report()` with ≠ 3 actions | Loops over all actions | `0c58f41` |
| NaN / Inf | Bad experiences dropped, bad updates skipped, warnings | `dbef43e` |
| Last batch not saved | Final checkpoint save | `4623b0c` |
| `TrainingWithInterval` starved its worker | Yields after each step | `1896b4a` |

Supporting changes:

- **`f4a6548`.** The GPU backend now runs only the forward pass and
  backprop. The loss gradient, clipping and SGD step are shared CPU code.
- **`c67512b`.** Each `Node` can carry its own configuration.

Not done: a learned value-head baseline. It would help when the average
reward differs a lot between states, but it needs a checkpoint format change
and GPU backward changes.

## How the fixes were checked

`tests/toy_suite.rs` holds small problems with known answers: 4 random input
vectors (contexts), 3 actions, and one best action per context. Each test
runs with seeds 1, 2 and 3 and must pass on all of them:

```bash
cargo test --release --test toy_suite -- --ignored --nocapture
```

- **Bandit.** Reward +1 for the best action, 0 otherwise. Pass: the best
  action on ≥ 95% of inference calls.
- **Small-signal.** Reward ±0.002 plus Gaussian noise with std 0.01, the
  scale of trading returns. Pass: ≥ 90%.
- **Null.** The reward is pure noise. Pass: no action above 50% of inference
  calls, and policy entropy ≥ 0.989 (uniform over 3 actions = 1.099).
- **Delayed reward.** Not tested, because the learner is one-step only.

### Results at each stage

Bandit and small-signal: share of best actions. Null: largest share of any
one action. Seeds 1 / 2 / 3.

| Stage | Bandit | Small-signal | Null | Passing |
| ----- | ------ | ------------ | ---- | ------- |
| 0.1.0 | 0.487 / 0.494 / 0.500 ✗ | 0.354 / 0.305 / 0.327 ✗ | 0.352 / 0.360 / 0.338 ✓ | 1 / 3 |
| Shaping off | 0.477 / 0.487 / 0.504 ✗ | 0.348 / 0.297 / 0.323 ✗ | 0.350 / 0.364 / 0.339 ✓ | 1 / 3 |
| Averaged, clipped gradients | 0.995 / 0.994 / 0.988 ✓ | 0.357 / 0.311 / 0.329 ✗ | 0.353 / 0.364 / 0.339 ✓ | 2 / 3 |
| Advantages, importance weights | 0.989 / 0.995 / 0.991 ✓ | 0.991 / 0.986 / 0.993 ✓ | 0.563 / 0.538 / 0.578 ✗ | 2 / 3 |
| Exploration schedule | 1.000 / 1.000 / 1.000 ✓ | 1.000 / 0.752 / 1.000 ✗ | 0.505 / 0.475 / 0.360 ✗ | 1 / 3 |
| **Entropy bonus (0.2.0 defaults)** | 1.000 / 1.000 / 1.000 ✓ | 0.998 / 0.998 / 0.999 ✓ | 0.413 / 0.377 / 0.354 ✓ | **3 / 3** |

- **0.1.0.** The bandit collapsed onto one action for every context (77-83%
  of calls). The null test passed only because nothing was learned.
- **Advantages.** These made learning independent of the reward scale, so
  ±0.002 rewards were learned. Noise then also moved the policy a little.
- **Exploration schedule.** With low exploration, one noisy context locked in
  on a wrong action. Before, constant high exploration had hidden this. The
  entropy bonus fixed it, and also kept the null policy near uniform.
- **GPU backend.** It gives the same results as the CPU at every printed
  digit (`TOY_BACKEND=gpu`).
- **Recommended settings.** The README's settings for small, noisy rewards
  pass small-signal and null on all seeds
  (`recommended_settings_small_noisy_rewards`).

### How the defaults were chosen

Each sweep changes one setting at the stage where it was introduced.

| Setting | Values tried | Result | Default |
| ------- | ------------ | ------ | ------- |
| `learning_rate` | 0.01, 0.05, 0.2 | 0.01 too slow for the bandit; 0.2 drifted on small-signal | 0.05 |
| `infer_temperature` | 0 (greedy), 0.5, 1.0 | Greedy fails the null test on every seed | 0.5 |
| `entropy_coef` | 0.01, 0.03, 0.1 | All pass; 0.1 leaves the widest null margin (0.413 vs 0.50) | 0.1 |

**Why inference isn't greedy by default.** When rewards are noise, the policy
is close to uniform but never exactly uniform. Argmax turns any small tilt
into "always action X", which is the original collapse. A low temperature
keeps an unsure policy unsure, while a confident policy still picks its best
action almost every time. `infer_temperature = 0` still gives pure argmax.

## Upgrading from 0.1.0

1. **Delete old checkpoints** (`./models/<id>.<model_name>`). The file format
   is unchanged, so collapsed 0.1.0 weights would load silently.
2. **Fix struct literals.** If you build `Configurations` in code, add
   `..Default::default()`. TOML/JSON configs need no change.
3. **Set `log_interval`** (e.g. 50) and show `info` logs for `aixker_rlt`.
   Watch `entropy` and `policy`.
4. **Scale input features** to roughly `[-1, 1]` or z-scores.
5. **Keep rewards in `[-1, 1]`.** They are clamped there; otherwise the scale
   no longer matters.
6. **Shaping.** If a client relied on it, `reward_shaping = true` restores
   it. Try without it first.

For small, noisy rewards, start from the README section "Recommended
settings for small, noisy rewards", and run with 2-3 seeds before trusting a
result.
