//! Training must work for any number of actions, not only 3.

use std::sync::Arc;

use aixker_rlt::configurations::Configurations;
use aixker_rlt::node::Node;
use aixker_rlt::RunningMode;

fn train_with_actions(output_number: usize) {
    let config = Configurations {
        batch_size: 8,
        total_batches: 3,
        input_number: 2,
        output_number,
        hidden_layers: 4,
        reply_capacity: 64,
        model_name: format!("action-count-{output_number}.json"),
        seed: Some(5),
        ..Default::default()
    };
    let path = format!("./models/test.{}", config.model_name);
    let _ = std::fs::remove_file(&path);
    let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
    let node = runtime.block_on(Node::start_with_config(
        Arc::new(config),
        |_| vec![0.5, -0.5],
        |_, action, _| (action as f32 * 0.1, action > 0),
        RunningMode::Training,
        "test",
    ));
    let saved = std::fs::read_to_string(&path).expect("final checkpoint is written");
    let _ = std::fs::remove_file(&path);
    assert!(node.batch_history.read().unwrap().len() > 3);
    // The checkpoint holds the final weights, not the ones from a batch earlier.
    let saved: serde_json::Value = serde_json::from_str(&saved).unwrap();
    let b2: Vec<f32> = serde_json::from_value(saved["b2"]["data"].clone()).unwrap();
    let w = node.weights();
    assert_eq!(&w[w.len() - output_number..], b2.as_slice());
}

#[test]
fn trains_with_two_actions() {
    train_with_actions(2);
}

#[test]
fn trains_with_five_actions() {
    train_with_actions(5);
}

/// The background worker of `TrainingWithInterval` must get to run on a
/// single-threaded runtime too.
#[test]
fn interval_worker_runs_on_current_thread_runtime() {
    let config = Configurations {
        interval_secs: 0,
        batch_size: 8,
        total_batches: 5,
        input_number: 2,
        output_number: 3,
        hidden_layers: 4,
        reply_capacity: 64,
        model_name: "interval-worker.json".to_owned(),
        seed: Some(5),
        ..Default::default()
    };
    let path = format!("./models/test.{}", config.model_name);
    let _ = std::fs::remove_file(&path);
    let steps = std::cell::Cell::new(0);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .unwrap();
    let node = runtime.block_on(Node::start_with_config(
        Arc::new(config),
        |_| {
            // Stop after 2,000 steps so a starved worker fails instead of hanging.
            steps.set(steps.get() + 1);
            if steps.get() > 2_000 {
                vec![]
            } else {
                vec![0.5, -0.5]
            }
        },
        |_, action, _| (action as f32 * 0.1, action > 0),
        RunningMode::TrainingWithInterval,
        "test",
    ));
    let _ = std::fs::remove_file(&path);
    assert!(!node.batch_history.read().unwrap().is_empty(), "the worker never trained");
}
