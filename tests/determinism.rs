//! Same seed + same inputs => identical weights; different seed => not.

mod common;

use std::rc::Rc;

use common::{bandit_reward, config, train, Contexts, RewardFn};

const BATCHES: usize = 1_000;
const ENV_SEED: u64 = 99;

fn run(seed: u64, name: &str) -> Vec<f32> {
    let contexts = Rc::new(Contexts::random(ENV_SEED));
    let reward: Rc<RewardFn> = Rc::new(bandit_reward);
    train(config(Some(seed), BATCHES, name), contexts, ENV_SEED, reward).weights()
}

#[test]
fn same_seed_gives_identical_weights() {
    let a = run(42, "determinism-a.json");
    let b = run(42, "determinism-b.json");
    let c = run(43, "determinism-c.json");
    assert_eq!(a, b, "same seed, same inputs: weights must be identical");
    assert_ne!(a, c, "different seeds: weights must differ");
}
