use crate::diagnostics::TrainStats;
use crate::configurations::Configurations;
use crate::rng;
use crate::models::gpu_model::{GpuModel, Grads};
use anyhow::Ok;
use log::{info, warn};
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::rand::distributions::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::u8;

#[derive(Clone, Serialize, Deserialize)]
pub struct Model {
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>,
    pub snapshot: String,
    id: String,
    /// Checkpoint file path. Not serialized.
    #[serde(skip)]
    path: String,
    /// GPU backend, enabled via `Configurations::backend = Gpu`.
    /// Not serialized; checkpoints stay backend-agnostic.
    #[serde(skip)]
    gpu: Option<Arc<GpuModel>>,
}

impl Model {
    /// A freshly initialized model for checkpoint `id`.
    pub fn new(id: &str, config: &Configurations) -> Self {
        let mut rng = rng::stream_rng(config.seed, rng::STREAM_INIT);
        let uniform = Uniform::new(-0.1, 0.1);
        Self {
            w1: Array2::random_using(
                (config.input_number, config.hidden_layers),
                uniform,
                &mut rng,
            ),
            b1: Array1::zeros(config.hidden_layers),
            w2: Array2::random_using(
                (config.hidden_layers, config.output_number),
                uniform,
                &mut rng,
            ),
            b2: Array1::zeros(config.output_number),
            snapshot: String::new(),
            id: id.to_owned(),
            path: Self::model_path(id, config),
            gpu: None,
        }
    }

    /// Load checkpoint `id` if it exists and parses; otherwise start fresh.
    pub fn load_or_new(id: &str, config: &Configurations) -> Self {
        let path = Self::model_path(id, config);
        let loaded = std::fs::read_to_string(&path)
            .ok()
            .and_then(|data| serde_json::from_str::<Model>(&data.replace("null", "0.0")).ok());
        match loaded {
            Some(mut model) => {
                model.path = path;
                model
            }
            None => Self::new(id, config),
        }
    }

    /// Enable the GPU compute backend. Falls back to CPU (with a warning)
    /// when no compatible adapter is available.
    pub fn enable_gpu(&mut self, max_batch: usize) {
        match GpuModel::new(&self.w1, &self.b1, &self.w2, &self.b2, max_batch.max(1)) {
            std::result::Result::Ok(gpu) => {
                info!("model '{}' running on the GPU backend", self.id);
                self.gpu = Some(Arc::new(gpu));
            }
            Err(e) => {
                warn!("GPU backend unavailable, falling back to CPU: {e}");
                self.gpu = None;
            }
        }
    }

    /// All parameters flattened in a fixed order: w1, b1, w2, b2.
    pub fn parameters(&self) -> Vec<f32> {
        self.w1
            .iter()
            .chain(self.b1.iter())
            .chain(self.w2.iter())
            .chain(self.b2.iter())
            .copied()
            .collect()
    }

    fn relu(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| v.max(0.0))
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        if let Some(gpu) = &self.gpu {
            match gpu.forward(x) {
                std::result::Result::Ok(logits) => return logits,
                Err(e) => warn!("GPU forward failed, using CPU: {e}"),
            }
        }
        let h = Self::relu(&(x.dot(&self.w1) + &self.b1));
        h.dot(&self.w2) + &self.b2
    }

    pub fn softmax2(x: &Array2<f32>, temperature: f32) -> Array2<f32> {
        let max_per_row = x.map_axis(Axis(1), |r| r.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        let exps = x - &max_per_row.insert_axis(Axis(1));
        let exps = exps.mapv(|v| (v / temperature).exp());
        let sum_per_row = exps.sum_axis(Axis(1)).insert_axis(Axis(1));
        &exps / &sum_per_row
    }

    pub fn softmax(logits: &Array2<f32>, temperature: f32) -> Vec<f32> {
        // temperature > 0.0
        let inv_temp = 1.0 / temperature;
        let mut max = f32::NEG_INFINITY;
        for &v in logits {
            if v > max {
                max = v
            }
        }
        // subtract max for numerical stability
        let mut exps: Vec<f32> = logits
            .iter()
            .map(|&v| ((v - max) * inv_temp).exp())
            .collect();
        let sum: f32 = exps.iter().sum();
        if sum == 0.0 {
            // fallback to uniform
            let n = logits.len() as f32;
            return vec![1.0 / n; logits.len()];
        }
        exps.iter_mut().for_each(|x| *x /= sum);
        exps
    }

    /// One policy-gradient step on a batch. Returns the batch statistics
    /// that depend on the model (loss, gradient norm, policy, entropy).
    pub fn reinforce(
        &mut self,
        x: &Array2<f32>,
        actions: &Array1<usize>,
        rewards: &Array1<f32>,
        lr: f32,
        max_grad_norm: f32,
    ) -> TrainStats {
        let (probs, mut grads) = self.policy_gradient(x, actions, rewards);
        let stats = Self::batch_stats(&probs, actions, rewards, &grads);
        grads.clip_norm(max_grad_norm);
        self.apply_gradients(&grads, lr);

        if self.w1.iter().any(|x| x.is_nan()) {
            warn!("NaN detected in weights!");
        }
        stats
    }

    /// Policy probabilities (temperature 1) and the gradient of
    /// `L = -mean_i(r_i · log π(a_i|s_i))` with respect to every weight.
    fn policy_gradient(
        &self,
        x: &Array2<f32>,
        actions: &Array1<usize>,
        rewards: &Array1<f32>,
    ) -> (Array2<f32>, Grads) {
        let logits = self.forward(x);
        let probs = Self::softmax2(&logits, 1.0);

        // dL/dz_i = -r_i · (onehot(a_i) - p_i) / n, for every action.
        let n = x.nrows().max(1) as f32;
        let mut grad_logits = probs.clone();
        for (i, &a) in actions.iter().enumerate() {
            grad_logits[[i, a]] -= 1.0;
            grad_logits.row_mut(i).mapv_inplace(|g| g * rewards[i] / n);
        }
        let grads = self.backward(x, &grad_logits);
        (probs, grads)
    }

    /// Model-side statistics of one batch. `loss` is the REINFORCE loss
    /// `-mean(r_i · log π(a_i|s_i))`.
    fn batch_stats(
        probs: &Array2<f32>,
        actions: &Array1<usize>,
        rewards: &Array1<f32>,
        grads: &Grads,
    ) -> TrainStats {
        let n = probs.nrows().max(1) as f32;
        let loss = actions
            .iter()
            .enumerate()
            .map(|(i, &a)| -rewards[i] * probs[[i, a]].max(1e-12).ln())
            .sum::<f32>()
            / n;
        let entropy = probs
            .rows()
            .into_iter()
            .map(|row| TrainStats::entropy_of(row.as_slice().unwrap()))
            .sum::<f32>()
            / n;
        let mut taken = vec![0.0; probs.ncols()];
        for &a in actions {
            taken[a] += 1.0 / n;
        }
        TrainStats {
            loss,
            grad_norm: grads.norm(),
            policy: probs.mean_axis(Axis(0)).map(|m| m.to_vec()).unwrap_or_default(),
            entropy,
            taken,
            ..Default::default()
        }
    }

    /// Backpropagate dL/dlogits through the network (GPU when enabled).
    fn backward(&self, x: &Array2<f32>, grad_logits: &Array2<f32>) -> Grads {
        if let Some(gpu) = &self.gpu {
            match gpu.backward(x, grad_logits) {
                std::result::Result::Ok(grads) => return grads,
                Err(e) => warn!("GPU backward failed, using CPU: {e}"),
            }
        }
        let h_pre = x.dot(&self.w1) + &self.b1;
        let h = Self::relu(&h_pre);
        let grad_h = grad_logits.dot(&self.w2.t());
        let grad_h_relu = grad_h * &h_pre.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 });
        Grads {
            w1: x.t().dot(&grad_h_relu),
            b1: grad_h_relu.sum_axis(Axis(0)),
            w2: h.t().dot(grad_logits),
            b2: grad_logits.sum_axis(Axis(0)),
        }
    }

    /// Plain SGD step; keeps the GPU copy of the weights in sync.
    fn apply_gradients(&mut self, grads: &Grads, lr: f32) {
        self.w1.scaled_add(-lr, &grads.w1);
        self.b1.scaled_add(-lr, &grads.b1);
        self.w2.scaled_add(-lr, &grads.w2);
        self.b2.scaled_add(-lr, &grads.b2);
        if let Some(gpu) = &self.gpu {
            gpu.upload_weights(&self.w1, &self.b1, &self.w2, &self.b2);
        }
    }

    fn model_path(id: &str, config: &Configurations) -> String {
        let dir = "./models";
        if let Err(err) = std::fs::create_dir_all(dir) {
            warn!("Failed to create model directory '{}': {}", dir, err);
        }
        format!("{}/{}.{}", dir, id, config.model_name)
    }

    pub fn save_model(&self) -> anyhow::Result<()> {
        let json = serde_json::to_string(self)?;
        let mut file = File::create(&self.path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn sanitize(&mut self) {
        for x in self.w1.iter_mut() {
            if !x.is_finite() || x.abs() > 1e6 {
                *x = 0.0;
            }
        }
        for x in self.b1.iter_mut() {
            if !x.is_finite() || x.abs() > 1e6 {
                *x = 0.0;
            }
        }
        for x in self.w2.iter_mut() {
            if !x.is_finite() || x.abs() > 1e6 {
                *x = 0.0;
            }
        }
        for x in self.b2.iter_mut() {
            if !x.is_finite() || x.abs() > 1e6 {
                *x = 0.0;
            }
        }
        // Keep the GPU copy in sync with the sanitized CPU weights.
        if let Some(gpu) = &self.gpu {
            gpu.upload_weights(&self.w1, &self.b1, &self.w2, &self.b2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_model() -> Model {
        let config = Configurations {
            input_number: 3,
            output_number: 3,
            hidden_layers: 5,
            seed: Some(3),
            ..Default::default()
        };
        let mut model = Model::new("grad-test", &config);
        // Non-zero biases so every ReLU unit is clearly on or off.
        model.b1 = Array1::from_vec(vec![0.05, -0.04, 0.03, 0.06, -0.02]);
        model.b2 = Array1::from_vec(vec![0.1, -0.2, 0.05]);
        model
    }

    fn loss(model: &Model, x: &Array2<f32>, actions: &Array1<usize>, rewards: &Array1<f32>) -> f64 {
        let probs = Model::softmax2(&model.forward(x), 1.0);
        let n = x.nrows() as f64;
        actions
            .iter()
            .enumerate()
            .map(|(i, &a)| -(rewards[i] as f64) * (probs[[i, a]] as f64).ln())
            .sum::<f64>()
            / n
    }

    #[test]
    fn policy_gradient_matches_finite_differences() {
        let mut model = tiny_model();
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![0.5, -0.3, 0.8, -0.6, 0.2, 0.1, 0.9, 0.4, -0.7, 0.3, -0.8, 0.5],
        )
        .unwrap();
        let actions = Array1::from_vec(vec![0, 2, 1, 2]);
        let rewards = Array1::from_vec(vec![1.0, -0.5, 0.3, 0.8]);
        let (_, grads) = model.policy_gradient(&x, &actions, &rewards);

        let eps = 1e-2f32;
        let check = |analytic: f32, model_plus: &Model, model_minus: &Model| {
            let numeric = (loss(model_plus, &x, &actions, &rewards)
                - loss(model_minus, &x, &actions, &rewards))
                / (2.0 * eps as f64);
            assert!(
                (analytic as f64 - numeric).abs() < 2e-3,
                "analytic {analytic} vs numeric {numeric}"
            );
        };
        for j in 0..3 {
            let (mut p, mut m) = (model.clone(), model.clone());
            p.b2[j] += eps;
            m.b2[j] -= eps;
            check(grads.b2[j], &p, &m);
        }
        for (r, c) in [(0, 0), (2, 3), (1, 4)] {
            let (mut p, mut m) = (model.clone(), model.clone());
            p.w1[[r, c]] += eps;
            m.w1[[r, c]] -= eps;
            check(grads.w1[[r, c]], &p, &m);
        }
        for (r, c) in [(0, 1), (3, 2)] {
            let (mut p, mut m) = (model.clone(), model.clone());
            p.w2[[r, c]] += eps;
            m.w2[[r, c]] -= eps;
            check(grads.w2[[r, c]], &p, &m);
        }

        // Averaged, not summed: doubling the batch with copies keeps the gradient.
        let x2 = ndarray::concatenate(Axis(0), &[x.view(), x.view()]).unwrap();
        let a2 = ndarray::concatenate(Axis(0), &[actions.view(), actions.view()]).unwrap();
        let r2 = ndarray::concatenate(Axis(0), &[rewards.view(), rewards.view()]).unwrap();
        let (_, grads2) = model.policy_gradient(&x2, &a2, &r2);
        assert!((grads.norm() - grads2.norm()).abs() < 1e-5);

        // lr = 0 leaves the weights alone.
        let before = model.parameters();
        model.reinforce(&x, &actions, &rewards, 0.0, 1.0);
        assert_eq!(before, model.parameters());
    }

    #[test]
    fn clipping_caps_the_global_norm() {
        let mut grads = Grads {
            w1: Array2::from_elem((2, 2), 3.0),
            b1: Array1::from_elem(2, 3.0),
            w2: Array2::from_elem((2, 2), 3.0),
            b2: Array1::from_elem(2, 3.0),
        };
        let before = grads.norm();
        grads.clip_norm(0.0);
        assert_eq!(grads.norm(), before, "0 disables clipping");
        grads.clip_norm(1.0);
        assert!((grads.norm() - 1.0).abs() < 1e-5);
        grads.clip_norm(5.0);
        assert!((grads.norm() - 1.0).abs() < 1e-5, "below the cap: unchanged");
    }
}
