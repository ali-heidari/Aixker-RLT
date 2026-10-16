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

/// Pick an action for `features` and return it with its logits and
/// probabilities.
pub fn infer_action(node: &Node, features: &Vec<f32>) -> (usize, Array2<f32>, Vec<f32>) {
    let s = select_action(node, features);
    (s.action, s.logits, s.probs)
}

pub(crate) fn select_action(node: &Node, features: &[f32]) -> Selection {
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

    let temperature = *node.temperature.read().unwrap();
    let mut probs = Model::softmax(&logits, temperature);
    // 4) Numerical safety: clamp and normalize
    for p in probs.iter_mut() {
        if *p < 0.0 {
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

    // 5) Epsilon-greedy exploration
    let eps: f32 = *node.epsilon.read().unwrap();
    let mut rng = node.rng.lock().unwrap();
    let action = if rng.gen::<f32>() < eps {
        rng.gen_range(0..probs.len())
    } else {
        let dist = WeightedIndex::new(&probs).unwrap();
        dist.sample(&mut *rng)
    };

    let n = probs.len() as f32;
    let behavior_prob = eps / n + (1.0 - eps) * probs[action];

    Selection {
        action,
        logits,
        probs,
        behavior_prob,
    }
}
