use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Experience {
    pub features: Vec<f32>,
    pub action: u8,
    pub latency_ms: f32,
    /// Reward used for training (after shaping, when enabled).
    pub reward: f32,
    /// Reward exactly as returned by the reward callback.
    #[serde(default)]
    pub raw_reward: f32,
    /// Probability the acting policy gave `action` when it was taken
    /// (0 = unknown). Used to correct for replaying old experiences.
    #[serde(default)]
    pub behavior_prob: f32,
    pub success: bool,
    pub timestamp_ms: u128,
}



