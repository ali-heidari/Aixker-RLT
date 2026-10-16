use std::sync::{Arc, OnceLock};

use crate::configurations::Configurations;

pub mod configurations;
mod constants;
pub mod diagnostics;
mod experience;
pub mod infer_action;
mod model;
mod models;
pub mod node;
mod reply_buffer;
mod rng;
mod worker;
mod footstep;
pub use crate::configurations::ComputeBackend;
pub use crate::diagnostics::TrainStats;
pub use crate::node::RunningMode;

pub static CONFIG: OnceLock<Arc<Configurations>> = OnceLock::new();

pub fn initialize(config: Arc<Configurations>) {
    CONFIG.set(config).unwrap()
}

pub fn get_config() -> Arc<Configurations> {
    let config = CONFIG.get().unwrap();
    config.clone()

    
}
