use crate::model::Model;

use super::node::Node;
use ndarray::{Array, Array2};
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;

/// One chosen action and how it was chosen.
pub(crate) struct Selection {
    pub action: usize,
    /// Logits shifted so the largest is 0.
    pub logits: Array2<f32>,
    /// Softmax of the logits at the sampling temperature.
    pub probs: Vec<f32>,
    /// Probability that this selection rule picks `action`, including the
    /// ε-random part. Stored with the experience for off-policy correction.
    pub behavior_prob: f32,
}

/// Pick the action to take for `features`, without exploration: no random
/// actions, sampling at `Configurations::infer_temperature` (`0` = greedy).
/// Returns the action, the logits (shifted so the largest is 0) and the
/// probabilities at that temperature.
pub fn infer_action(node: &Node, features: &Vec<f32>) -> (usize, Array2<f32>, Vec<f32>) {
    let s = select_action(node, features, 0.0, node.config().infer_temperature);
    (s.action, s.logits, s.probs)
}

/// Choose an action: with probability `epsilon` a uniform random one,
/// otherwise a sample from `softmax(logits / temperature)`, or the argmax
/// when `temperature` is 0.
pub(crate) fn select_action(
    node: &Node,
    features: &[f32],
    epsilon: f32,
    temperature: f32,
) -> Selection {
    let x = Array2::from_shape_vec((1, node.config().input_number), features.to_vec()).unwrap();
    let model = node.model.read().unwrap();
    let logits = model.forward(&x);
    let num_actions = logits.len();

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let logits = logits
        .into_iter()
        .map(|x| (x - max_logit).clamp(-50.0, 50.0))
        .collect();
    let logits = Array::from_shape_vec((1,num_actions), logits).unwrap();

    let mut probs = if temperature > 0.0 {
        Model::softmax(&logits, temperature)
    } else {
        // Greedy: all probability on the first highest logit.
        let best = logits
            .iter()
            .position(|&z| z == 0.0)
            .unwrap_or(0);
        (0..num_actions).map(|i| if i == best { 1.0 } else { 0.0 }).collect()
    };
    // Numerical safety: clamp and normalize
    for p in probs.iter_mut() {
        if p.is_nan() || *p < 0.0 {
            *p = 0.0
        }
    }
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    } else {
        probs = vec![1.0 / probs.len() as f32; probs.len()];
    }

    let mut rng = node.rng.lock().unwrap();
    let action = if epsilon > 0.0 && rng.gen::<f32>() < epsilon {
        rng.gen_range(0..probs.len())
    } else {
        let dist = WeightedIndex::new(&probs).unwrap();
        dist.sample(&mut *rng)
    };

    let n = probs.len() as f32;
    let behavior_prob = epsilon / n + (1.0 - epsilon) * probs[action];

    Selection {
        action,
        logits,
        probs,
        behavior_prob,
    }
}
