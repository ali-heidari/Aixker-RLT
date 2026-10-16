//! Toy problems with known answers. Each runs with 3 seeds and must pass on
//! all of them. They are slow in debug builds, so run them with:
//!
//! ```bash
//! cargo test --release --test toy_suite -- --ignored --nocapture
//! ```
//!
//! Set `TOY_LOG=N` to print the training statistics every N batches, and
//! `TOY_LR=x` / `TOY_INFER_T=x` / `TOY_ENT=x` to override the default
//! learning rate / inference temperature / entropy coefficient, and
//! `TOY_BACKEND=gpu` to train on the GPU backend.
//!
//! There is no delayed-reward test: the learner is one-step only (no returns
//! and no bootstrapping), see FINDINGS.md.

mod common;

use std::rc::Rc;
use std::sync::Once;

use aixker_rlt::configurations::Configurations;
use common::{
    bandit_reward, config, evaluate, null_reward, small_signal_reward, train, Contexts, Eval,
    RewardFn, N_ACTIONS, SEEDS,
};

/// Inference calls per context during evaluation.
const SAMPLES: usize = 250;

struct StderrLogger;

impl log::Log for StderrLogger {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Info && m.target().starts_with("aixker_rlt")
    }
    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) && record.args().to_string().starts_with("batch ") {
            eprintln!("{}", record.args());
        }
    }
    fn flush(&self) {}
}

static LOGGER: StderrLogger = StderrLogger;
static INIT: Once = Once::new();

/// Stats log interval from `TOY_LOG` (0 = off).
fn log_interval() -> u64 {
    let n = std::env::var("TOY_LOG").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    if n > 0 {
        INIT.call_once(|| {
            log::set_logger(&LOGGER).ok();
            log::set_max_level(log::LevelFilter::Info);
        });
    }
    n
}

/// Changes applied to the test config before training.
type Tweak = dyn Fn(&mut Configurations);

/// Train and evaluate one seed.
fn run(name: &str, seed: u64, batches: usize, reward: Rc<RewardFn>, tweak: &Tweak) -> Eval {
    let contexts = Rc::new(Contexts::random(seed));
    let mut cfg = config(Some(seed), batches, &format!("toy-{name}-{seed}.json"));
    tweak(&mut cfg);
    cfg.log_interval = log_interval();
    if let Some(lr) = std::env::var("TOY_LR").ok().and_then(|v| v.parse().ok()) {
        cfg.learning_rate = lr;
    }
    if let Some(t) = std::env::var("TOY_INFER_T").ok().and_then(|v| v.parse().ok()) {
        cfg.infer_temperature = t;
    }
    if std::env::var("TOY_BACKEND").is_ok_and(|v| v.eq_ignore_ascii_case("gpu")) {
        cfg.backend = aixker_rlt::ComputeBackend::Gpu;
    }
    if let Some(b) = std::env::var("TOY_ENT").ok().and_then(|v| v.parse().ok()) {
        cfg.entropy_coef = b;
    }
    let node = train(cfg, contexts.clone(), seed + 100, reward);
    evaluate(&node, &contexts, SAMPLES)
}

fn report(name: &str, seed: u64, eval: &Eval, pass: bool) {
    let freq: Vec<String> = eval.action_freq.iter().map(|f| format!("{f:.3}")).collect();
    println!(
        "RESULT test={name} seed={seed} best_rate={:.3} action_freq=[{}] policy_entropy={:.4} pass={pass}",
        eval.best_rate,
        freq.join(", "),
        eval.policy_entropy,
    );
}

/// Run all seeds, report each, then fail if any seed failed.
fn check(name: &str, batches: usize, reward: Rc<RewardFn>, pass: impl Fn(&Eval) -> bool) {
    check_with(name, batches, reward, &|_| {}, pass)
}

fn check_with(
    name: &str,
    batches: usize,
    reward: Rc<RewardFn>,
    tweak: &Tweak,
    pass: impl Fn(&Eval) -> bool,
) {
    let mut failed = Vec::new();
    for seed in SEEDS {
        let eval = run(name, seed, batches, reward.clone(), tweak);
        let ok = pass(&eval);
        report(name, seed, &eval, ok);
        if !ok {
            failed.push(seed);
        }
    }
    assert!(failed.is_empty(), "{name} failed for seeds {failed:?}");
}

/// 4 contexts, 3 actions, reward +1 for the best action and 0 otherwise.
/// Pass: the best action on at least 95% of inference calls.
#[test]
#[ignore]
fn contextual_bandit() {
    check("bandit", 1_500, Rc::new(bandit_reward), |e| e.best_rate >= 0.95);
}

/// Same, with trading-scale rewards: ±0.002 plus N(0, 0.01²) noise.
/// Pass: the best action on at least 90% of inference calls.
#[test]
#[ignore]
fn small_signal_bandit() {
    check("small-signal", 4_000, Rc::new(small_signal_reward), |e| {
        e.best_rate >= 0.90
    });
}

/// Rewards are pure noise, independent of the action.
/// Pass: no action above 50% of inference calls, and the trained policy's
/// entropy stays at least 90% of the uniform entropy ln(3).
#[test]
#[ignore]
fn null_rewards() {
    let min_entropy = 0.9 * (N_ACTIONS as f32).ln();
    check("null", 3_000, Rc::new(null_reward), move |e| {
        e.action_freq.iter().all(|&f| f <= 0.5) && e.policy_entropy >= min_entropy
    });
}

/// The README's "recommended settings for small, noisy rewards" (bigger
/// batches and replay buffer, longer exploration), on the small-signal and
/// null problems. Same pass criteria as above.
#[test]
#[ignore]
fn recommended_settings_small_noisy_rewards() {
    let recommended = |c: &mut Configurations| {
        c.batch_size = 128;
        c.reply_capacity = 4_096;
        c.exploration_batches = 1_000;
    };
    check_with("rec-small-signal", 4_000, Rc::new(small_signal_reward), &recommended, |e| {
        e.best_rate >= 0.90
    });
    let min_entropy = 0.9 * (N_ACTIONS as f32).ln();
    check_with("rec-null", 3_000, Rc::new(null_reward), &recommended, move |e| {
        e.action_freq.iter().all(|&f| f <= 0.5) && e.policy_entropy >= min_entropy
    });
}

