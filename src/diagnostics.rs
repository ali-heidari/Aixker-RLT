//! Training statistics, computed for every batch and logged every
//! `log_interval` batches. They make policy collapse visible while it
//! happens: falling entropy, one action taking over the policy, exploding
//! gradient norms, or a reward signal that is mostly noise.

use std::fmt;

/// Statistics of one training batch.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrainStats {
    /// Batch number (1-based).
    pub batch: u32,
    /// Mean of the raw rewards returned by the reward callback.
    pub reward_mean: f32,
    /// Standard deviation of the raw rewards.
    pub reward_std: f32,
    /// Policy-gradient loss of the batch.
    pub loss: f32,
    /// Global L2 norm of the weight gradients, before any clipping.
    pub grad_norm: f32,
    /// Random-action rate (ε) at the time of the batch.
    pub epsilon: f32,
    /// Sampling temperature at the time of the batch.
    pub temperature: f32,
    /// Mean policy probability of each action over the batch's inputs.
    pub policy: Vec<f32>,
    /// Mean policy entropy over the batch's inputs, in nats
    /// (uniform over `n` actions = `ln n`).
    pub entropy: f32,
    /// Fraction of the batch's stored actions per action.
    pub taken: Vec<f32>,
}

impl TrainStats {
    /// Mean and population standard deviation of `values`.
    pub(crate) fn mean_std(values: &[f32]) -> (f32, f32) {
        if values.is_empty() {
            return (0.0, 0.0);
        }
        let n = values.len() as f32;
        let mean = values.iter().sum::<f32>() / n;
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        (mean, var.sqrt())
    }

    /// Entropy (nats) of one probability vector.
    pub(crate) fn entropy_of(probs: &[f32]) -> f32 {
        probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum()
    }
}

fn fmt_vec(v: &[f32]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{x:.3}")).collect();
    format!("[{}]", parts.join(", "))
}

impl fmt::Display for TrainStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "batch {}: reward mean={:.5} std={:.5} | loss={:.5} grad_norm={:.5} | \
             eps={:.3} temp={:.3} | policy={} entropy={:.4} | taken={}",
            self.batch,
            self.reward_mean,
            self.reward_std,
            self.loss,
            self.grad_norm,
            self.epsilon,
            self.temperature,
            fmt_vec(&self.policy),
            self.entropy,
            fmt_vec(&self.taken),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_std_and_entropy() {
        let (m, s) = TrainStats::mean_std(&[1.0, 3.0]);
        assert!((m - 2.0).abs() < 1e-6 && (s - 1.0).abs() < 1e-6);
        assert_eq!(TrainStats::mean_std(&[]), (0.0, 0.0));

        let uniform = TrainStats::entropy_of(&[1.0 / 3.0; 3]);
        assert!((uniform - 3f32.ln()).abs() < 1e-5);
        assert_eq!(TrainStats::entropy_of(&[1.0, 0.0, 0.0]), 0.0);
    }
}
