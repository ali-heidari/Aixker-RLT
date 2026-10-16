# Report: fixing the learning collapse in aixker-rlt

Branch `agent/fix-learning`, based on `develop` at `fa9d711`.
Crate version 0.1.0 → **0.2.0**. Full analysis: [FINDINGS.md](FINDINGS.md).
Every change: [CHANGELOG.md](CHANGELOG.md).

**Result.** All four reported problems were real, and all are fixed. Six
more bugs were found and fixed. On small problems with known answers, the
learner now finds the right action every time (≥ 0.998 for the toy bandits,
on 3 seeds each), stays near uniform when rewards are pure noise, and gives
the same results on the GPU backend.

## The algorithm in plain words

The crate learns a **contextual bandit policy with REINFORCE** (a one-step
policy gradient). A two-layer network maps the input features to one score
(logit) per action, and a softmax turns the scores into action
probabilities. Each step, the agent picks an action, your callback returns a
reward, and the experience goes into a replay buffer. Every `batch_size`
steps, a batch is drawn from the buffer, and each sample's action becomes
more or less likely depending on its reward. There are no multi-step
returns: every reward is credited only to the action just taken.

After this work, one training step does this:

1. Turn rewards into advantages: `A = (r - batch mean) / batch std`.
2. Weight each sample by `ρ = min(π(a|s) / b, 2)`. Here `π(a|s)` is the
   current policy's probability for the action, and `b` is the probability
   the action had when it was taken.
3. Minimize `-mean(ρ·A·log π(a|s) + β·H(π))`, where `H` is the policy entropy
   and β = `entropy_coef`.
4. Average the gradient over the batch, clip it to `max_grad_norm`, and take
   one SGD step of size `learning_rate`.

While training, the agent explores with ε-random actions and a temperature,
following a linear schedule. At inference there are no random actions.

## The reported problems

| # | Problem | Confirmed? | Fix | Commit |
| - | ------- | ---------- | --- | ------ |
| 1 | Reward shaping adds about −0.3 to every failure and can't be turned off | Yes. The penalty is up to ~0.5 and depends only on per-action counters, so the best the network can learn is one fixed action. | `reward_shaping` option, **off by default** | `d4777e4` |
| 2 | Exploration ends at its caps (70% random, temperature 7) | Yes. Any learned preference set the `diverging` flag, which sent ε to 0.7 in one step. | Linear schedule 0.3 → 0.05 (ε) and 1.5 → 1.0 (T) | `be5e1c2` |
| 3 | No baseline, gradients summed, weights reach 1e6 | Yes. Shaped rewards were almost all negative, so weights drifted one way every batch. "1e6" is the `sanitize()` threshold, where weights were silently reset to 0. | Averaged and clipped gradients (`77ed7de`); normalized advantages as the baseline (`fe4532f`) | `77ed7de`, `fe4532f` |
| 4 | No seed control | Yes. `thread_rng()` in weight init, action sampling and replay sampling. | `seed` option with one `StdRng` per stream; determinism test | `6addcbc` |

## Extra bugs found and fixed

| Bug | Effect | Commit |
| --- | ------ | ------ |
| Replay trained on old and ε-random actions as if the current policy had chosen them | Biased updates toward whatever the old behavior did often | `fe4532f` (importance weights) |
| `infer_action` explored at inference | Decisions were partly random (ε up to 0.7, T up to 7 on a training node) | `be5e1c2` |
| Only the chosen action's part of the softmax gradient was applied | The update was not the gradient of any loss | `77ed7de` |
| The learning rate decayed on every step, toward 1e-6 | Learning stalled; small rewards never moved the policy | `77ed7de` |
| "Loss" was really the batch's mean reward | Collapse was invisible in logs | `404f84d` (diagnostics) |
| One NaN reward survived the clamp | All weights became NaN, then were silently zeroed | `dbef43e` |
| `report()` assumed exactly 3 actions | Training panicked with 2, 4, 5 ... actions | `0c58f41` |
| The last batch was never saved to the checkpoint | An `Infer` node loaded weights one batch old | `4623b0c` |
| `TrainingWithInterval` never yielded | On a single-threaded runtime the worker never trained | `1896b4a` |
| Low exploration let a policy lock in on a weak signal | One context in the small-signal test learned the wrong action | `1817b99` (entropy bonus) |

Supporting changes:

- **`f4a6548`.** The GPU now runs only the forward pass and backprop, so
  clipping and gradient norms are possible.
- **`c67512b`.** Each node can carry its own config, so tests can run
  several seeds in one process.

## Toy suite results

Four contexts, three actions, 3 seeds per test
(`cargo test --release --test toy_suite -- --ignored --nocapture`).
Bandit and small-signal: share of inference calls that pick the best action
(pass ≥ 0.95 / ≥ 0.90). Null: the largest share of any action and the policy
entropy (pass ≤ 0.50 and ≥ 0.989; uniform = 1.099).

| Stage | Bandit | Small-signal (±0.002 + noise 0.01) | Null (pure noise) | Passing |
| ----- | ------ | ---------------------------------- | ----------------- | ------- |
| Before fixes | 0.487 / 0.494 / 0.500 ✗ | 0.354 / 0.305 / 0.327 ✗ | 0.352 / 0.360 / 0.338 ✓ (learns nothing) | 1 / 3 |
| Fix 1: shaping off | 0.477 / 0.487 / 0.504 ✗ | 0.348 / 0.297 / 0.323 ✗ | 0.350 / 0.364 / 0.339 ✓ | 1 / 3 |
| Fix 2: averaged + clipped gradients | 0.995 / 0.994 / 0.988 ✓ | 0.357 / 0.311 / 0.329 ✗ | 0.353 / 0.364 / 0.339 ✓ | 2 / 3 |
| Fix 3: advantages + importance weights | 0.989 / 0.995 / 0.991 ✓ | 0.991 / 0.986 / 0.993 ✓ | 0.563 / 0.538 / 0.578 ✗ | 2 / 3 |
| Fix 4: exploration schedule | 1.000 / 1.000 / 1.000 ✓ | 1.000 / 0.752 / 1.000 ✗ | 0.505 / 0.475 / 0.360 ✗ | 1 / 3 |
| Round 1: entropy bonus β = 0.1 | 1.000 / 1.000 / 1.000 ✓ | 0.998 / 0.998 / 0.999 ✓ | 0.413 / 0.377 / 0.354 ✓ | 3 / 3 |
| **Final** (defaults, CPU) | 1.000 / 1.000 / 1.000 ✓ | 0.998 / 0.998 / 0.999 ✓ | 0.413 / 0.377 / 0.354 ✓ | **3 / 3** |
| Final, GPU backend | identical | identical | identical | 3 / 3 |

- **Before:** the bandit collapses onto one action for every context (77-83%
  of calls), which is the reported behavior.
- **Fix 3:** the learner becomes scale-free, so ±0.002 rewards are learned.
  Noise then also moves the policy a little.
- **Fix 4:** the regression is real. With low exploration, one noisy context
  locked in on a wrong action. Before, constant high exploration hid this.
- **Round 1:** an entropy bonus fixed both remaining failures. One of the 3
  allowed investigation rounds was used.
- **Delayed-reward test:** skipped. The learner is one-step only.
- The sweeps behind the defaults (learning rate, inference temperature,
  entropy coefficient) are in [FINDINGS.md](FINDINGS.md).

## What changes for a downstream user (aixker-dealer)

**Behavior changes you get by upgrading:**

- **Raw reward.** Training uses the raw reward. Shaping is gone unless you
  set `reward_shaping = true`.
- **Relative rewards.** Only relative rewards within a batch matter. A
  constant offset does nothing, and the scale doesn't change the step size.
- **Exploration.** ε and temperature follow the schedule. They no longer
  react to the `diverging` flag.
- **Inference.** `infer_action` and `Infer` mode never take random actions.
  They sample at `infer_temperature` (0.5). Set it to `0` for pure argmax,
  but argmax turns an unsure policy into one fixed action.
- **Loss value.** `Node::loss` / `batch_history` hold the real loss, not the
  mean reward.
- **Report labels.** The per-batch report labels actions "Action 0, 1, ..."
  instead of "High/Normal/Low pressure".

**What you must change:**

1. **Delete old checkpoints** (`./models/<id>.<model_name>`). Weights trained
   by the old learner are collapsed or were reset to 0 by `sanitize()`. The
   file format is unchanged, so an old file would silently load.
2. **Fix struct literals.** If you build `Configurations` with a struct
   literal, add `..Default::default()`. TOML/JSON configs need no change.
3. **Turn on the stats log.** Set `log_interval` (e.g. 50) and make sure
   your logger shows `info` for `aixker_rlt`. Watch `entropy` and `policy`.
4. **Scale your inputs** to roughly `[-1, 1]` or z-scores. Raw prices make
   the hidden layer and the gradients huge.
5. **Keep rewards in `[-1, 1]`.** They are clamped there. Per-step trading
   returns already fit.

## Recommended settings for aixker-dealer's next run

These are the README's "small, noisy rewards" settings. They pass the
small-signal and null tests on all 3 seeds
(`recommended_settings_small_noisy_rewards`).

```toml
reward_shaping = false        # the default; say it explicitly
learning_rate = 0.05          # lower to 0.02 if grad_norm sits at max_grad_norm
max_grad_norm = 1.0
entropy_coef = 0.1
epsilon_start = 0.3
epsilon_end = 0.05
temperature_start = 1.5
temperature_end = 1.0
exploration_batches = 1000    # about a quarter of total_batches
infer_temperature = 0.5
batch_size = 128
reply_capacity = 4096         # ~30 batches; older data is only partly corrected
seed = 42                     # then change it to check results hold across seeds
log_interval = 50
```

- **Check across seeds.** Run with at least 2-3 seeds before trusting a
  result.
- **Collapse alarm.** If `entropy` falls toward 0 while `reward mean` stays
  flat, the policy is fitting noise. Raise `entropy_coef` or lower
  `learning_rate`.

## Not done

- **Value-head baseline.** A learned value head would be a better baseline
  when the average reward differs a lot between market states. It needs a
  checkpoint format change and GPU backward changes, so it's left as a
  follow-up.
- **Repeatable interval mode.** `TrainingWithInterval` can't be fully
  repeatable, because it trains on a timer.
- **Untouched docs.** Your uncommitted `docs/index.md` and `docs/roadmap.md`
  were left as they were and are not part of any commit.

## Commit to pin

**`1568669`** (`chore: release 0.2.0`) is the last commit that changes code.
The branch head adds only this report on top of it.
