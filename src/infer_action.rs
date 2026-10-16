use crate::model::Model;

use super::node::Node;
use ndarray::{Array, Array2};
use rand::distributions::{Distribution, WeightedIndex};
use rand::Rng;

pub fn infer_action(node: &Node, features: &Vec<f32>) -> (usize, Array2<f32>, Vec<f32>) {
    let mut rng = rand::thread_rng();

    let x = Array2::from_shape_vec((1, node.config().input_number), features.clone()).unwrap();
    let model = node.model.read().unwrap();
    let logits = model.forward(&x);
    let num_actions = logits.len();

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let logits = logits
        .into_iter()
        .map(|x| (x - max_logit).clamp(-50.0, 50.0))
        .collect();
    let logits = Array::from_shape_vec((1,num_actions), logits).unwrap();

    let temperature = node.temperature.write().unwrap().clone();
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
    let eps: f32 = node.epsilon.write().unwrap().clone();
    let action = if rng.gen::<f32>() < eps {
        rng.gen_range(0..probs.len())
    } else {
        let dist = WeightedIndex::new(&probs).unwrap();
        dist.sample(&mut rng)
    };

    (action, logits, probs)
}
