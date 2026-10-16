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
    let _ = std::fs::remove_file(&path);
    assert!(node.batch_history.read().unwrap().len() > 3);
}

#[test]
fn trains_with_two_actions() {
    train_with_actions(2);
}

#[test]
fn trains_with_five_actions() {
    train_with_actions(5);
}
