use super::experience::Experience;
use rand::prelude::*;

pub struct ReplayBuffer {
    buf: Vec<Experience>,
    cap: usize,
    idx: usize,
}

impl ReplayBuffer {
    pub fn new(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            cap,
            idx: 0,
        }
    }

    pub fn push(&mut self, ex: Experience) {
        if self.buf.len() < self.cap {
            self.buf.push(ex);
        } else {
            self.buf[self.idx] = ex;
            self.idx = (self.idx + 1) % self.cap;
        }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Draw `n` experiences uniformly, with replacement.
    pub fn sample<R: Rng>(&self, n: usize, rng: &mut R) -> Vec<Experience> {
        (0..n)
            .filter_map(|_| self.buf.get(rng.gen_range(0..self.buf.len())).cloned())
            .collect()
    }
}
