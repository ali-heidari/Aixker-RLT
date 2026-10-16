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
        }
    }
}
