//! Random number streams. With `Configurations::seed` set, every source of
//! randomness gets its own `StdRng` derived from that seed and a stream
//! index, so runs repeat exactly. Without a seed each stream draws fresh
//! entropy from the OS.

use rand::rngs::StdRng;
use rand::SeedableRng;

/// Weight initialization.
pub(crate) const STREAM_INIT: u64 = 0;
/// Action selection: ε-random actions and sampling from the policy.
pub(crate) const STREAM_ACTING: u64 = 1;
/// Training worker `i` uses `STREAM_WORKER + i` (replay sampling).
pub(crate) const STREAM_WORKER: u64 = 1_000;

/// SplitMix64 finalizer: spreads nearby inputs over the whole u64 range.
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Seed of stream `stream` under the main seed `seed`.
pub(crate) fn derive_seed(seed: u64, stream: u64) -> u64 {
    splitmix64(seed ^ splitmix64(stream))
}

/// The generator for `stream`: seeded when `seed` is set, OS entropy
/// otherwise.
pub(crate) fn stream_rng(seed: Option<u64>, stream: u64) -> StdRng {
    match seed {
        Some(seed) => StdRng::seed_from_u64(derive_seed(seed, stream)),
        None => StdRng::from_entropy(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn streams_are_repeatable_and_distinct() {
        let a: u64 = stream_rng(Some(7), STREAM_ACTING).gen();
        let b: u64 = stream_rng(Some(7), STREAM_ACTING).gen();
        let c: u64 = stream_rng(Some(7), STREAM_INIT).gen();
        let d: u64 = stream_rng(Some(8), STREAM_ACTING).gen();
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
        assert_ne!(derive_seed(1, STREAM_WORKER + 1), derive_seed(1, STREAM_WORKER + 2));
    }
}
