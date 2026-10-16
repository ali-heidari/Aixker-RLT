//! Small environments with known answers, shared by the integration tests.
//! Not part of the crate's public API.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use aixker_rlt::configurations::Configurations;
use aixker_rlt::infer_action::infer_action;
use aixker_rlt::node::Node;
use aixker_rlt::RunningMode;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

pub const N_ACTIONS: usize = 3;
pub const N_FEATURES: usize = 8;
pub const N_CONTEXTS: usize = 4;

/// The seeds every toy test runs with.
pub const SEEDS: [u64; 3] = [1, 2, 3];

/// A contextual bandit: a few fixed input vectors, each with one best action.
pub struct Contexts {
    pub features: Vec<Vec<f32>>,
    pub best: Vec<usize>,
}

impl Contexts {
    /// `N_CONTEXTS` random vectors in [-1, 1]. Best actions are a shuffled
    /// `[0, 1, 2, x]`, so no single action is right for every context.
    pub fn random(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed ^ 0xC0FF_EE00);
        let features = (0..N_CONTEXTS)
            .map(|_| (0..N_FEATURES).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        let mut best: Vec<usize> = (0..N_CONTEXTS).map(|i| i % N_ACTIONS).collect();
        best[N_CONTEXTS - 1] = rng.gen_range(0..N_ACTIONS);
        best.shuffle(&mut rng);
        Self { features, best }
    }
}

/// Reward of taking `action` in context `ctx`, with the env's own rng for
/// any noise.
pub type RewardFn = dyn Fn(&Contexts, usize, usize, &mut StdRng) -> f32;

/// Bandit reward: +1 for the best action, 0 otherwise.
pub fn bandit_reward(c: &Contexts, ctx: usize, action: usize, _: &mut StdRng) -> f32 {
    if action == c.best[ctx] {
        1.0
    } else {
        0.0
    }
}

/// Trading-scale reward: ±0.002 plus Gaussian noise with std 0.01.
pub fn small_signal_reward(c: &Contexts, ctx: usize, action: usize, rng: &mut StdRng) -> f32 {
    let signal = if action == c.best[ctx] { 0.002 } else { -0.002 };
    signal + 0.01 * gaussian(rng)
}

/// Pure noise, independent of the action: N(0, 0.01²).
pub fn null_reward(_: &Contexts, _: usize, _: usize, rng: &mut StdRng) -> f32 {
    0.01 * gaussian(rng)
}

/// Standard normal sample (Box-Muller), to avoid an extra dependency.
pub fn gaussian(rng: &mut StdRng) -> f32 {
    let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
    let u2: f32 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Test configuration. `model_name` must be unique per test run, since
/// checkpoints are files under `./models`.
pub fn config(seed: Option<u64>, total_batches: usize, model_name: &str) -> Configurations {
    Configurations {
        interval_secs: 0,
        batch_size: 64,
        total_batches,
        input_number: N_FEATURES,
        output_number: N_ACTIONS,
        hidden_layers: 16,
        reply_capacity: 2_048,
        model_name: model_name.to_owned(),
        log_interval: 0,
        mode: RunningMode::Training,
        seed,
        ..Default::default()
    }
}

const NODE_ID: &str = "toy";

fn checkpoint_path(config: &Configurations) -> String {
    format!("./models/{NODE_ID}.{}", config.model_name)
}

/// Train a fresh node with `Node::start_with_config` in `Training` mode.
/// Contexts are drawn uniformly by an env rng seeded with `env_seed`.
/// Any old checkpoint with the same name is removed first, and the new one
/// afterwards.
pub fn train(
    config: Configurations,
    contexts: Rc<Contexts>,
    env_seed: u64,
    reward: Rc<RewardFn>,
) -> Arc<Node> {
    let path = checkpoint_path(&config);
    let _ = std::fs::remove_file(&path);

    // (env rng, current context)
    let env = Rc::new(RefCell::new((StdRng::seed_from_u64(env_seed), 0usize)));
    let (env_in, ctx_in) = (env.clone(), contexts.clone());
    let input_bearer = move |_lowest_state: u32| {
        let mut env = env_in.borrow_mut();
        let ctx = env.0.gen_range(0..N_CONTEXTS);
        env.1 = ctx;
        ctx_in.features[ctx].clone()
    };
    let (env_r, ctx_r) = (env, contexts);
    let reward_fn = move |_inputs: &Vec<f32>, action: u32, _counter: u32| {
        let mut env = env_r.borrow_mut();
        let ctx = env.1;
        let r = reward(&ctx_r, ctx, action as usize, &mut env.0);
        (r, r > 0.0)
    };

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    let node = runtime.block_on(Node::start_with_config(
        Arc::new(config),
        input_bearer,
        reward_fn,
        RunningMode::Training,
        NODE_ID,
    ));
    let _ = std::fs::remove_file(&path);
    node
}

/// What the trained node does at inference.
#[derive(Debug)]
pub struct Eval {
    /// Share of inference calls that returned the context's best action.
    pub best_rate: f32,
    /// Share of inference calls per action, over all contexts.
    pub action_freq: Vec<f32>,
    /// Entropy (nats) of the trained policy at temperature 1, averaged over
    /// the last training batch's inputs.
    pub policy_entropy: f32,
}

/// Call `infer_action` `samples` times per context, with the node switched
/// to its `Infer` settings.
pub fn evaluate(node: &Arc<Node>, contexts: &Contexts, samples: usize) -> Eval {
    Node::set_default_factors(node, RunningMode::Infer);
    let mut best = 0usize;
    let mut freq = vec![0.0f32; N_ACTIONS];
    let total = (samples * N_CONTEXTS) as f32;
    for (ctx, features) in contexts.features.iter().enumerate() {
        for _ in 0..samples {
            let (action, _, _) = infer_action(node, features);
            freq[action] += 1.0 / total;
            if action == contexts.best[ctx] {
                best += 1;
            }
        }
    }
    let policy_entropy = node
        .last_stats
        .read()
        .unwrap()
        .as_ref()
        .map(|s| s.entropy)
        .unwrap_or(f32::NAN);
    Eval {
        best_rate: best as f32 / total,
        action_freq: freq,
        policy_entropy,
    }
}
