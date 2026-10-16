use serde::Deserialize;

use crate::node::RunningMode;

/// Compute backend used for the neural network math.
/// `Cpu` (default) runs everything through `ndarray`;
/// `Gpu` dispatches to wgpu compute shaders (falls back to CPU when no
/// compatible adapter is found).
#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputeBackend {
    #[default]
    Cpu,
    Gpu,
}

#[derive(Deserialize, Debug, Clone)]
pub struct Configurations {
    pub interval_secs: u64,
    pub batch_size: u32,
    pub total_batches: usize,
    pub input_number: usize,
    pub output_number: usize,
    pub hidden_layers: usize,
    pub reply_capacity: usize,
    pub model_name: String,
    pub log_interval: u64,
    pub mode: RunningMode,
    #[serde(default)]
    pub backend: ComputeBackend,
    /// Seed for every source of randomness (weight init, exploration,
    /// action sampling, replay sampling). `None` (default) uses fresh OS
    /// entropy, so runs differ.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Rewrite rewards with the built-in shaping (bonus for rare successful
    /// actions, penalty of up to ~0.5 for failed or over-used ones). Off by
    /// default: the learner trains on the raw reward. Only enable it when
    /// your rewards are on the order of ±1.
    #[serde(default)]
    pub reward_shaping: bool,
    /// SGD step size, applied once per batch to the batch-averaged
    /// gradient. Can be changed at runtime through `Node::lr`.
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f32,
    /// Clip the gradient so its global L2 norm is at most this value.
    /// `0` disables clipping.
    #[serde(default = "default_max_grad_norm")]
    pub max_grad_norm: f32,
    /// Random-action rate (ε) at the start of training.
    #[serde(default = "default_epsilon_start")]
    pub epsilon_start: f32,
    /// Random-action rate (ε) after `exploration_batches` batches.
    #[serde(default = "default_epsilon_end")]
    pub epsilon_end: f32,
    /// Sampling temperature at the start of training.
    #[serde(default = "default_temperature_start")]
    pub temperature_start: f32,
    /// Sampling temperature after `exploration_batches` batches.
    #[serde(default = "default_temperature_end")]
    pub temperature_end: f32,
    /// Batches over which ε and the temperature move linearly from their
    /// start to their end values. They stay at the end values afterwards.
    #[serde(default = "default_exploration_batches")]
    pub exploration_batches: u32,
    /// Temperature used by `infer_action` and in `Infer` mode. There is no
    /// random-action rate at inference. `0` = always the highest-scoring
    /// action (greedy).
    #[serde(default = "default_infer_temperature")]
    pub infer_temperature: f32,
}

fn default_epsilon_start() -> f32 {
    0.3
}

fn default_epsilon_end() -> f32 {
    0.05
}

fn default_temperature_start() -> f32 {
    1.5
}

fn default_temperature_end() -> f32 {
    1.0
}

fn default_exploration_batches() -> u32 {
    500
}

fn default_infer_temperature() -> f32 {
    0.5
}

fn default_learning_rate() -> f32 {
    0.05
}

fn default_max_grad_norm() -> f32 {
    1.0
}

impl Configurations {
    /// (ε, temperature) of the exploration schedule after `batch` batches.
    pub fn exploration_at(&self, batch: u32) -> (f32, f32) {
        let progress = if self.exploration_batches == 0 {
            1.0
        } else {
            (batch as f32 / self.exploration_batches as f32).min(1.0)
        };
        let lerp = |start: f32, end: f32| start + (end - start) * progress;
        (
            lerp(self.epsilon_start, self.epsilon_end),
            lerp(self.temperature_start, self.temperature_end),
        )
    }
}

impl Default for Configurations {
    /// Defaults for the optional fields. `input_number` and `output_number`
    /// must still be set to match your features and actions, e.g.
    /// `Configurations { input_number: 8, output_number: 3, ..Default::default() }`.
    fn default() -> Self {
        Self {
            interval_secs: crate::constants::TRAIN_INTERVAL_SECS,
            batch_size: crate::constants::BATCH_SIZE,
            total_batches: 1_000,
            input_number: 0,
            output_number: 0,
            hidden_layers: 16,
            reply_capacity: 10_000,
            model_name: "model.json".to_owned(),
            log_interval: 0,
            mode: RunningMode::Training,
            backend: ComputeBackend::Cpu,
            seed: None,
            reward_shaping: false,
            learning_rate: default_learning_rate(),
            max_grad_norm: default_max_grad_norm(),
            epsilon_start: default_epsilon_start(),
            epsilon_end: default_epsilon_end(),
            temperature_start: default_temperature_start(),
            temperature_end: default_temperature_end(),
            exploration_batches: default_exploration_batches(),
            infer_temperature: default_infer_temperature(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exploration_schedule_is_linear_then_flat() {
        let c = Configurations {
            epsilon_start: 0.3,
            epsilon_end: 0.05,
            temperature_start: 2.0,
            temperature_end: 1.0,
            exploration_batches: 100,
            ..Default::default()
        };
        let close = |(e, t): (f32, f32), (ee, te): (f32, f32)| {
            assert!((e - ee).abs() < 1e-6 && (t - te).abs() < 1e-6, "({e}, {t})")
        };
        close(c.exploration_at(0), (0.3, 2.0));
        close(c.exploration_at(50), (0.175, 1.5));
        close(c.exploration_at(100), (0.05, 1.0));
        close(c.exploration_at(10_000), (0.05, 1.0));
        let instant = Configurations { exploration_batches: 0, ..c };
        close(instant.exploration_at(0), (0.05, 1.0));
    }

    #[test]
    fn defaults_end_with_low_exploration() {
        let c = Configurations::default();
        assert!(c.epsilon_end <= 0.05 && c.temperature_end <= 1.0);
        assert!(!c.reward_shaping);
        assert_eq!(c.seed, None);
    }
}

