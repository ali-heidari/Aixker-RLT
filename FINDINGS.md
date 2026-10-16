# Findings: why agents collapse to one action

Recon of `aixker-rlt` 0.1.0 at commit `fa9d711` (branch `develop`).
Line numbers refer to that commit.

Short answer: the learner mostly trains on the built-in reward shaping, not on
the caller's reward. The shaping depends on per-action statistics, not on the
input, so the best the network can learn is "always pick action X". Exploration,
the missing baseline, summed gradients and off-policy replay make it worse.

## Build and tests

- `cargo build` and `cargo test` pass. The only test is `gpu_matches_cpu`.
  It ran on a real GPU adapter on this machine (it did not skip).
- There are no cargo features. The GPU backend is a runtime switch
  (`Configurations::backend = Gpu`) and is always compiled, so `cargo check`
  covers it. There is no `--features` flag to check.
- `cargo clippy` reports 7 warnings in the library (style only).

## The algorithm in plain words

The crate is a **one-step policy-gradient learner (REINFORCE) for contextual
bandits**. Each step is scored on its own. There are no multi-step returns
and no bootstrapping: an `Experience` has no "next state".

- **Network** (`src/model.rs:80-93`). Two layers:
  `logits = relu(x·W1 + b1)·W2 + b2`, one logit per action.
  The policy is `softmax(logits / temperature)`.
- **Acting** (`src/node.rs:466-519`, `src/infer_action.rs:9-51`). For each
  input, `infer_action` computes the softmax at the current temperature. With
  probability ε it picks a uniform random action. Otherwise it samples from the
  softmax. The caller's callback returns `(reward, success)`.
- **Reward shaping** (`src/node.rs:426-464`). The reward may be rewritten
  (see problem 1). The rewritten reward replaces the raw one in the buffer.
- **Replay buffer** (`src/reply_buffer.rs`). A ring buffer of
  `reply_capacity` experiences. `sample(n)` draws `n` items uniformly, with
  replacement.
- **When training runs.** In `Training` mode, `Worker::do_once` runs once every
  `batch_size` steps (`src/node.rs:152-156`). In `TrainingWithInterval` mode a
  tokio task trains every `interval_secs` (`src/worker.rs:78-111`).
- **Loss** (`src/model.rs:127-175`). The intended loss is
  `L = -Σ_i r_i · log π(a_i | s_i)`, computed at temperature 1, with one plain
  SGD step per batch. The code only keeps part of its gradient (see extra
  bug E1).
- **Reported "loss".** The value stored in `batch_history` and logged as
  "Loss" is the batch's **mean reward**, not a loss
  (`src/model.rs:174`, `src/models/gpu_model.rs:549`).
- **Exploration** lives in two places: ε and temperature in `infer_action`,
  and the automatic ε/temperature/learning-rate tuning in `Node`
  (`adjust_factors`, `adjust_lr`, `src/node.rs:374-412`).

The GPU path (`src/models/gpu_model.rs`, `src/*.wgsl`) runs the same math
and has the same bugs.

## Reported problems

### 1. Reward shaping swamps the real reward — CONFIRMED

- **Where.** `Node::next` calls `adjust_settings` whenever `enable_adjustment`
  is true (`src/node.rs:481-485`). `set_default_factors` sets it to true for
  both training modes (`src/node.rs:168`, `173`). `Node::start` always calls
  `set_default_factors` (`src/node.rs:105`). The flag is private, so a caller
  cannot turn shaping off.
- **What it does** (`src/node.rs:455-461`). With `diff = 1/n - ratio[a]`:
  - on `success == false`: `r - 0.2·|diff| - 0.3·(1 - success_rate[a])`
  - on `success == true`: `r · (1 + 0.3·diff + 0.4·(1 - success_rate[a]))`
- **When.** It starts once the action ratios or success rates have variance
  above 0.01, or after 300 batches (`src/node.rs:445-450`). Any policy that
  prefers one action passes that threshold quickly.
- **Size.** With a 0-50% success rate the penalty is 0.15-0.3 plus up to 0.13
  from `|diff|`. A trading reward of 0.001-0.01 is 30-300× smaller. The report's
  "about −0.3, about 100×" matches.
- **Effect on learning.** The penalty depends only on per-action counters
  (success rate, how often the action was taken). It does not depend on the
  input. So the strongest signal in the buffer is "action X fails less
  often overall". The best response to that signal is the same action for
  every input. That is the collapse. The shaped value also overwrites the raw
  reward in the buffer (`src/node.rs:487-495`), so the real signal is lost.
  `|diff|` punishes rare actions as well as frequent ones, which is the
  opposite of the comment above it.

### 2. Exploration ends at its caps — CONFIRMED

- **Where.** `set_loss` calls `adjust_factors(!diverging)` after every batch
  past batch 50 (`src/node.rs:176-190`).
- **What it does** (`src/node.rs:374-392`).
  - Going down: `ε·0.99` (floor 0.01), `T·0.99` (floor 0.5).
  - Going up: `ε + 0.99` (cap 0.7), `T + 0.99` (cap 7.0).
  One "up" step sends ε straight to 0.7. Four send T to 7.
- **Trigger.** `diverging` is true when
  `(var(action_ratios) + var(success_rates)) / 2 > 0.01` (`src/node.rs:437-438`).
  Ratios of `[0.5, 0.3, 0.2]` already give a variance of 0.0156. Success rates
  differ by action in almost any real task.
- **Effect on learning.** As soon as the policy learns a preference, the node
  calls that "diverging" and maxes out exploration. At ε = 0.7 and T = 7 the
  agent acts almost at random, so what it learned barely changes behavior.
  Exploration only decays while the policy is near uniform, which is the
  reverse of what a schedule should do. `diverging` also changes what the
  caller's `input_bearer` receives (`src/node.rs:128-144`).

### 3. No baseline, gradients summed, weights explode — CONFIRMED

- **No baseline.** `reinforce` multiplies by the raw reward `r_i`
  (`src/model.rs:155-157`). Nothing is subtracted and there is no value head.
  With shaping, almost every stored reward is negative. A REINFORCE update
  with all-negative rewards pushes the chosen actions' logits down every
  batch. That is a steady drift in one direction, not noise that averages
  out.
- **Summed, not averaged.** `h.t().dot(...)`, `x.t().dot(...)` and
  `sum_axis(Axis(0))` sum over the batch (`src/model.rs:159-164`). The step
  size grows with `batch_size` (×64 at the default).
- **Nothing pulls weights back.** There is no gradient clipping and no weight
  decay. Raw (unscaled) trading features make `h` large, which makes the next
  gradients larger.
- **Why "about 1e6".** `Model::sanitize` resets every weight that is not
  finite or has `|w| > 1e6` to **0** (`src/model.rs:201-226`). It runs before
  each checkpoint save (`src/node.rs:331`). So the 1e6 ceiling in the report is
  this threshold. Weights grow until they hit it, then are silently zeroed.

### 4. No seed control — CONFIRMED

- `rand::thread_rng()` is used for weight init (`src/model.rs:30`), action
  sampling and ε exploration (`src/infer_action.rs:10`), and replay sampling
  (`src/reply_buffer.rs:33`).
- `Configurations` has no seed field (`src/configurations.rs:16-30`).
- No run can be repeated, so a failure cannot be debugged by replaying it.
- There is one worker per node (`Worker::new(1)`), not a pool of threads.

## Extra checks

### Off-policy replay without correction — CONFIRMED (separate bug)

- The buffer keeps up to `reply_capacity` old experiences, sampled uniformly
  (`src/reply_buffer.rs:32-37`, `src/worker.rs:67`).
- Those actions came from older policies, mixed with ε-random actions, at
  temperatures up to 7. The gradient is computed for the **current** policy at
  temperature **1** (`src/model.rs:152`), as if the data were on-policy.
- Nothing records the probability the action had when it was taken, so no
  importance weight can be computed.
- Effect: the update is biased toward whatever the old behavior did often.
  Together with the missing baseline, frequent actions get the biggest pushes.

### `infer_action` explores at inference time — CONFIRMED

- `infer_action` always applies ε-random actions and samples from the softmax
  (`src/infer_action.rs:41-48`). It never takes the argmax.
- In `Infer` mode the node uses ε = 0.01 and T = 0.5 (`src/node.rs:167`). A
  caller that uses a training node for decisions gets ε up to 0.7 and T up
  to 7.
- `docs/architecture.md` says it picks "the most probable one". The code does
  not.

### Gradient clipping and learning rate — NO CLIPPING; LR CHANGES PER STEP

- There is no gradient clipping anywhere.
- One SGD step per batch, as intended (`src/model.rs:166-169`).
- But `adjust_lr` runs inside `adjust_settings`, which runs on **every
  environment step**, whenever `batch_history.len() % 50 == 0`
  (`src/node.rs:440-443`). That condition stays true for all `batch_size` steps
  of that window, including the window before the first batch. So the learning
  rate is multiplied by `0.99` up to 64 times in a row (×0.53) every 50 batches,
  until it hits its floor of `1e-6`. When "stuck", `lr + 0.99` jumps straight
  to the `1e-4` cap (`src/node.rs:394-406`).

### NaN / Inf handling — WEAK

- `reinforce` warns if `w1` has a NaN but still applies the update, and does
  not check `b1`, `w2`, `b2` (`src/model.rs:171-173`).
- A NaN reward survives `clamp(-1.0, 1.0)` (`src/worker.rs:39`) and turns every
  weight into NaN in one batch. `sanitize` then sets them all to 0 without a
  warning.
- `infer_action` falls back to a uniform policy when the probabilities sum to
  NaN or 0 (`src/infer_action.rs:32-39`). That hides a broken model.
- `load_model` replaces every `null` in the JSON with `0.0`
  (`src/model.rs:193`), turning saved NaNs into zeros.
- Input features are never checked.

## Other bugs found

- **E1. Wrong softmax gradient** (`src/model.rs:154-157`,
  `src/reinforce.wgsl`). The gradient of `-r·log softmax(z)_a` is
  `-r·(onehot(a) - p)` across **all** actions. The code keeps only the chosen
  action's term, `-r·(1 - p_a)`, and sets the rest to 0. The update is not the
  gradient of any loss. The other actions' logits never move directly.
- **E2. "Loss" is the mean reward** (`src/model.rs:174`). Collapse is hard to
  see because nothing logs a real loss, gradient norm or entropy.
- **E3. `report()` assumes 3 actions** (`src/node.rs:219-220`, `263-300`,
  `509-510`). `try_into::<[f32; 3]>().unwrap()` panics whenever
  `output_number != 3`. The labels ("High pressure" ...) come from a
  load-balancer use case. **Fixed**: the report now loops over all actions.
- **E4. Training temperature ≠ acting temperature.** Actions are sampled at T
  (0.5-7) but `reinforce` trains `softmax(z / 1)`. (Part of the off-policy
  problem.)
- **E5. `TrainingWithInterval` never yields.** The main loop only awaits in
  `Infer` mode (`src/node.rs:121-124`). On a single-threaded tokio runtime the
  background worker never gets to run. It works on the multi-thread runtime.
- **E6. Read paths take write locks.** `infer_action` reads ε and T through
  `.write()` (`src/infer_action.rs:24`, `42`), which blocks other readers.

## Delayed-reward test

Skipped. The algorithm is one-step only: each experience stores one reward for
one action and no next state, and there is no return or bootstrapping. A
delayed-reward test would test a feature the crate does not have.

## Toy suite results

`tests/toy_suite.rs`, run with
`cargo test --release --test toy_suite -- --ignored --nocapture`.
Seeds 1, 2, 3. Network 8 → 16 → 3, batch 64, replay 2,048.
Budgets: bandit 1,500 batches, small-signal 4,000, null 3,000.

- **Bandit / small-signal** columns: share of inference calls that return
  the context's best action. Pass ≥ 0.95 / ≥ 0.90.
- **Null** column: the largest share of any one action at inference, then
  the trained policy's entropy (uniform = ln 3 = 1.0986). Pass: share ≤ 0.50
  and entropy ≥ 0.989.
- Inference is evaluated through `infer_action` after
  `Node::set_default_factors(Infer)`, as an `Infer` node would run.

| Stage | Bandit (s1 / s2 / s3) | Small-signal (s1 / s2 / s3) | Null (s1 / s2 / s3) | Passing |
| ----- | --------------------- | --------------------------- | ------------------- | ------- |
| Before fixes | 0.487 / 0.494 / 0.500 ✗ | 0.354 / 0.305 / 0.327 ✗ | 0.352, 1.099 / 0.360, 1.099 / 0.338, 1.099 ✓ | 1 / 3 |
| Fix 1: shaping off | 0.477 / 0.487 / 0.504 ✗ | 0.348 / 0.297 / 0.323 ✗ | 0.350, 1.099 / 0.364, 1.099 / 0.339, 1.099 ✓ | 1 / 3 |
| Fix 2: averaged + clipped gradients, lr 0.05 | 0.995 / 0.994 / 0.988 ✓ | 0.357 / 0.311 / 0.329 ✗ | 0.353, 1.099 / 0.364, 1.099 / 0.339, 1.099 ✓ | 2 / 3 |
| Fix 3: normalized advantage + importance weights | 0.989 / 0.995 / 0.991 ✓ | 0.991 / 0.986 / 0.993 ✓ | 0.563, 1.073 / 0.538, 1.080 / 0.578, 1.047 ✗ | 2 / 3 |
| Fix 4: exploration schedule, inference T 0.5 | 1.000 / 1.000 / 1.000 ✓ | 1.000 / 0.752 / 1.000 ✗ | 0.505, 1.079 / 0.475, 1.086 / 0.360, 1.098 ✗ | 1 / 3 |
| Round 1: entropy bonus β = 0.1 | 1.000 / 1.000 / 1.000 ✓ | 0.998 / 0.998 / 0.999 ✓ | 0.413, 1.091 / 0.377, 1.097 / 0.354, 1.098 ✓ | 3 / 3 |
| **Final** (defaults, CPU) | 1.000 / 1.000 / 1.000 ✓ | 0.998 / 0.998 / 0.999 ✓ | 0.413, 1.091 / 0.377, 1.097 / 0.354, 1.098 ✓ | 3 / 3 |
| Final, GPU backend | 1.000 / 1.000 / 1.000 ✓ | 0.998 / 0.998 / 0.999 ✓ | 0.413, 1.091 / 0.377, 1.097 / 0.354, 1.098 ✓ | 3 / 3 |
| Final, README recommended settings | — | 0.999 / 0.998 / 0.998 ✓ | 0.337, 1.098 / 0.400, 1.096 / 0.382, 1.097 ✓ | 2 / 2 |

Before fixes:

- **Bandit** collapses: one action takes 77-83% of all calls, for every
  context. That's the reported "always hold" behavior.
- **Small-signal** learns nothing. The policy stays uniform because the
  learning rate decays to 1e-6 and the shaping penalty drowns the ±0.002
  signal.
- **Null** passes only because nothing is learned at all, for the same
  reasons.

Fix 2 learning-rate sweep (same suite, only `learning_rate` changed):

| `learning_rate` | Bandit | Small-signal | Null |
| --------------- | ------ | ------------ | ---- |
| 0.01 | 0.615 / 0.708 / 0.654 ✗ | 0.352 / 0.300 / 0.321 ✗ | ✓ |
| **0.05** (default) | 0.995 / 0.994 / 0.988 ✓ | 0.357 / 0.311 / 0.329 ✗ | ✓ |
| 0.2 | 0.993 / 0.997 / 0.993 ✓ | 0.378 / 0.340 / 0.374 ✗ | ✓ |

- 0.01 is too slow for 1,500 batches. 0.05 and 0.2 both learn the bandit;
  0.05 keeps a margin on both sides.
- Small-signal still fails at every rate. Its raw rewards are ±0.002, so the
  gradient is ~500× smaller than in the bandit. That's fixed by advantage
  normalization (fix 3), not by a bigger step.
- At 0.2, small-signal starts drifting toward one action (0.46-0.48 of
  calls) without learning the right one: the missing baseline at work.

Fix 3 notes:

- **What was fixed.** The learner is a policy gradient, so it now uses an
  advantage instead of the raw reward: `A = (r - batch mean) / batch std`.
  The batch mean is the baseline. Dividing by the std makes the step size
  independent of the reward scale, so ±0.002 rewards move the policy as
  much as ±1 rewards do.
- **Why no running mean or value head.** Centering per batch already
  subtracts a baseline, so a separate running mean would be cancelled by
  it. A learned value head would change the checkpoint format and the GPU
  backward pass. It isn't clean in this code, so it's left as a follow-up.
- **Off-policy fix.** Each experience now stores `b`, the probability the
  acting rule gave its action (ε mix and sampling temperature included).
  Training weights each sample by `ρ = min(π(a|s) / b, 2)`, where `π` is the
  current policy. This is truncated importance sampling: replayed and
  exploratory actions are reweighted toward what the current policy would
  do, and the cap of 2 stops one rare sample from dominating a batch.
- **New failure: null.** The policy itself stays close to uniform
  (entropy 1.05-1.08, above the 0.989 bar). But the old inference rule
  (sampling at T = 0.5, plus ε) sharpens its small, noise-driven tilt past
  50% for one action. Scale-free learning means pure noise now moves the
  policy a little. Fix 4 changes inference, so this is checked again there.

Fix 4: inference temperature sweep (same training; only inference
changes):

| `infer_temperature` | Bandit | Small-signal | Null |
| ------------------- | ------ | ------------ | ---- |
| 0 (greedy) | 1.000 / 1.000 / 1.000 ✓ | 1.000 / 0.750 / 1.000 ✗ | 1.000 / 1.000 / 0.750 ✗ |
| **0.5** (default) | 1.000 / 1.000 / 1.000 ✓ | 1.000 / 0.752 / 1.000 ✗ | 0.505 / 0.475 / 0.360 ✗ |
| 1.0 | 0.998 / 0.999 / 1.000 ✓ | 0.994 / 0.769 / 0.999 ✗ | 0.425 / 0.402 / 0.351 ✓ |

Fix 4 notes:

- **Greedy fails the null test by construction.** When rewards are noise,
  the policy is close to uniform but never exactly uniform. Argmax turns any
  tiny tilt into "always action X", which is the reported collapse. So
  inference samples at a low temperature instead (`infer_temperature`, 0 =
  greedy is still available).
- **Small-signal seed 2 regressed.** One context settled early on a wrong
  action (p ≈ 0.85 vs 0.10 for the right one at T = 1). The policy-gradient
  push on an action shrinks with its probability, so it can't climb back in
  4,000 batches. Fix 3 only passed because the old rule kept ε near 0.7 the
  whole time. With low exploration this local optimum is real.
- **Null at T = 0.5** fails seed 1 by 0.005. The trained policy is not
  confident (entropy 1.08), but noise still tilts it a little.

### Investigation round 1: entropy bonus

Two problems were left after fix 4. Small-signal seed 2 got stuck on a wrong
action, and the null policy drifted with the noise. Both come from the
policy becoming confident before the evidence supports it. The standard
remedy is an entropy bonus: the loss gets `-β · H(π)`, which pulls the policy
back toward uniform unless the advantages push harder. It works on
normalized advantages, so `β` does not depend on the reward scale.

| `entropy_coef` β | Bandit | Small-signal | Null (largest share) |
| ---------------- | ------ | ------------ | -------------------- |
| 0 (fix 4) | 1.000 / 1.000 / 1.000 ✓ | 1.000 / 0.752 / 1.000 ✗ | 0.505 / 0.475 / 0.360 ✗ |
| 0.01 | 1.000 / 1.000 / 1.000 ✓ | 1.000 / 0.999 / 1.000 ✓ | 0.486 / 0.469 / 0.348 ✓ |
| 0.03 | 1.000 / 1.000 / 1.000 ✓ | 1.000 / 0.999 / 1.000 ✓ | 0.474 / 0.432 / 0.338 ✓ |
| **0.1** (default) | 1.000 / 1.000 / 1.000 ✓ | 0.998 / 0.998 / 0.999 ✓ | 0.413 / 0.377 / 0.354 ✓ |

- Every β tried passes all 9 runs. β = 0.1 is the default because it leaves
  the widest margin on the null test (0.413 vs the 0.50 limit), at almost
  no cost on the bandits.
- Seed 2 of small-signal now finds the right action. The bonus kept that
  context from locking in early.
- One round was needed, out of the 3 allowed.

### Final runs

- **CPU, defaults.** 9/9 pass in about 9 s (release build). The results
  match the round-1 β = 0.1 run digit for digit, as seeded runs should.
- **GPU backend** (`TOY_BACKEND=gpu`). Same results at every printed digit;
  5 m 17 s, because per-dispatch overhead dominates for an 8 → 16 → 3
  network.
- **README "recommended settings for small, noisy rewards"** (batch 128,
  replay 4,096, 1,000 exploration batches; test
  `recommended_settings_small_noisy_rewards`). Small-signal and null pass
  on all seeds.

## Status of every finding

| Finding | Status |
| ------- | ------ |
| 1. Reward shaping swamps the reward | Fixed: `reward_shaping`, off by default |
| 2. Exploration stuck at its caps | Fixed: linear schedule, no exploration at inference |
| 3. No baseline, summed gradients, exploding weights | Fixed: normalized advantages, averaged and clipped gradients |
| 4. No seed control | Fixed: `seed` with per-stream `StdRng`; `determinism` test |
| Off-policy replay | Fixed: truncated importance weights from stored behavior probabilities |
| `infer_action` explores | Fixed: no ε, `infer_temperature` (0.5, or 0 for greedy) |
| No gradient clipping; per-step lr decay | Fixed: `max_grad_norm`; constant `learning_rate` |
| Weak NaN/Inf handling | Fixed: bad experiences dropped, bad updates skipped, warnings |
| E1. Wrong softmax gradient | Fixed: full gradient; finite-difference test |
| E2. "Loss" is mean reward | Fixed: real loss, plus the statistics log |
| E3. `report()` assumes 3 actions | Fixed |
| E4. Training vs acting temperature | Fixed by the importance weights (`b` includes the acting temperature) |
| E5. `TrainingWithInterval` never yields | Fixed |
| E6. Write locks on read paths | Fixed in the action-selection rework |
| Last batch not saved | Fixed (found while writing tests) |
| Value-head baseline | Not done: needs a checkpoint format change and GPU backward changes |

