//! Two nodes with different configurations in one process, without the
//! global `initialize()`.

use std::sync::Arc;

use aixker_rlt::configurations::Configurations;
use aixker_rlt::infer_action::infer_action;
use aixker_rlt::node::Node;
use aixker_rlt::RunningMode;

fn config(input_number: usize, output_number: usize, model_name: &str) -> Arc<Configurations> {
    Arc::new(Configurations {
        interval_secs: 0,
        batch_size: 8,
        total_batches: 1,
        input_number,
        output_number,
        hidden_layers: 4,
        reply_capacity: 64,
        model_name: model_name.to_owned(),
        log_interval: 0,
        mode: RunningMode::Infer,
        backend: Default::default(),
    })
}

#[test]
fn nodes_with_different_configs_coexist() {
    let a = Node::new_with_config(config(2, 3, "per-node-a.json"), 0.0, 1.0, RunningMode::Infer, "test");
    let b = Node::new_with_config(config(5, 4, "per-node-b.json"), 0.0, 1.0, RunningMode::Infer, "test");

    let (action_a, _, probs_a) = infer_action(&a, &vec![0.1; 2]);
    let (action_b, _, probs_b) = infer_action(&b, &vec![0.1; 5]);

    assert_eq!(probs_a.len(), 3);
    assert_eq!(probs_b.len(), 4);
    assert!(action_a < 3 && action_b < 4);
    assert_eq!(a.config().input_number, 2);
    assert_eq!(b.config().input_number, 5);
}
