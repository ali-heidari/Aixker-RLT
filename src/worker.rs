use crate::diagnostics::TrainStats;
use crate::experience::Experience;

use super::model::Model;
use super::node::Node;
use anyhow::{Ok, Result};
use log::{info, warn};
use ndarray::{s, Array1, Array2};
use std::{sync::Arc, time::Duration};
use tokio::time::sleep;

pub struct Worker {
    id: u32,
    learning_rate: f32,
}

impl Worker {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            learning_rate: 0.00001,
        }
    }

    fn train_on_batch(
        &self,
        input_number: usize,
        batch: Vec<Experience>,
        mut model: std::sync::RwLockWriteGuard<'_, Model>,
    ) -> Result<TrainStats> {
        let mut inputs = Array2::<f32>::zeros((batch.len(), input_number));
        let mut predicted_action = Array1::<usize>::zeros(batch.len());
        let mut rewards = Array1::<f32>::zeros(batch.len());
        let mut raw_rewards = Vec::with_capacity(batch.len());

        for (i, experience) in batch.iter().enumerate() {
            inputs.slice_mut(s![i, ..]).assign(&Array1::from(
                experience.features[..input_number].to_vec(),
            ));
            predicted_action[i] = experience.action as usize;
            rewards[i] = experience.reward.clamp(-1.0, 1.0);
            raw_rewards.push(experience.raw_reward);
        }

        //  let loss = model.backward(&x, &y, 0.01);
        let mut stats = model.reinforce(&inputs, &predicted_action, &rewards, self.learning_rate);
        (stats.reward_mean, stats.reward_std) = TrainStats::mean_std(&raw_rewards);

        Ok(stats)
    }

    pub fn start(node: Arc<Node>) {
        tokio::spawn(async move {
            let mut worker = Worker::new(1);
            if let Err(e) = worker.start_to_work(node).await {
                warn!("training worker error: {:?}", e);
            }
        });
    }

    pub fn do_once(&mut self, node: &Node) -> Result<()> {
        self.learning_rate = *node.lr.read().unwrap();

        let buffer = node.buffer.read().unwrap();
        if buffer.len() < node.config().batch_size as usize {
            drop(buffer);
            info!("replay buffer below batch size; skipping training step");
            return Ok(());
        }

        let batch = buffer.sample(node.config().batch_size as usize);
        drop(buffer);

        let model = node.model.write().unwrap();
        let stats = self.train_on_batch(node.config().input_number, batch, model).unwrap();
        let size = node.batch_history.read().unwrap().len();

        node.record_stats(size as u32 + 1, stats);
        Ok(())
    }

    async fn start_to_work(&mut self, node: Arc<Node>) -> Result<()> {
        info!("Worker {} is working!", self.id);

        let mut batch_counter = 0;

        loop {
            sleep(Duration::from_secs(node.config().interval_secs)).await;
            self.learning_rate = *node.lr.read().unwrap();

            let buffer = node.buffer.read().unwrap();
            if buffer.len() < node.config().batch_size as usize {
                drop(buffer);
                continue;
            }

            if batch_counter > node.config().total_batches as u32 {
                drop(buffer);
                info!(
                    "Training complete after {} batches.",
                    node.config().total_batches
                );
                break;
            }
            batch_counter += 1;

            let batch = buffer.sample(node.config().batch_size as usize);
            drop(buffer);

            let model = node.model.write().unwrap();
            let stats = self.train_on_batch(node.config().input_number, batch, model).unwrap();
            node.record_stats(batch_counter, stats);
        }
        Ok(())
    }
}
