//! GPU compute backend for the two-layer policy network, built on wgpu.
//!
//! The GPU runs the heavy matrix work: the forward pass and backpropagation
//! of a given loss gradient. The loss gradient and the SGD step
//! are computed on the CPU (they are O(batch × actions) and O(weights)), and
//! the updated weights are re-uploaded with `upload_weights`. The CPU copy of
//! the weights stays authoritative for JSON checkpointing.

use std::sync::mpsc;

use anyhow::{anyhow, Result};
use log::info;
use ndarray::{Array1, Array2};
use wgpu::util::DeviceExt;

const WG_1D: u32 = 64;
const WG_2D: u32 = 8;

fn ceil_div(v: u32, d: u32) -> u32 {
    v.div_ceil(d)
}

/// 16-byte uniform payload shared by every shader (`u32`/`f32` lanes).
fn params_bytes(p: [u32; 4]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (i, v) in p.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes());
    }
    out
}

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub struct GpuModel {
    device: wgpu::Device,
    queue: wgpu::Queue,

    input: usize,
    hidden: usize,
    output: usize,
    max_batch: usize,

    // Persistent weight buffers.
    w1: wgpu::Buffer,
    b1: wgpu::Buffer,
    w2: wgpu::Buffer,
    b2: wgpu::Buffer,

    // Per-batch scratch buffers (sized for `max_batch`).
    x: wgpu::Buffer,
    logits: wgpu::Buffer,
    grad_logits: wgpu::Buffer,

    // Weight-gradient outputs.
    grad_w1: wgpu::Buffer,
    grad_b1: wgpu::Buffer,
    grad_w2: wgpu::Buffer,
    grad_b2: wgpu::Buffer,

    staging_logits: wgpu::Buffer,
    staging_w1: wgpu::Buffer,
    staging_b1: wgpu::Buffer,
    staging_w2: wgpu::Buffer,
    staging_b2: wgpu::Buffer,

    p_linear: wgpu::ComputePipeline,
    p_grad_weights: wgpu::ComputePipeline,
    p_grad_bias: wgpu::ComputePipeline,
    p_grad_hidden: wgpu::ComputePipeline,

    // Uniform buffers, one per dispatch site (all 16 bytes).
    u_l1: wgpu::Buffer,
    u_l2: wgpu::Buffer,
    u_grad_w2: wgpu::Buffer,
    u_grad_b2: wgpu::Buffer,
    u_grad_h: wgpu::Buffer,
    u_grad_w1: wgpu::Buffer,
    u_grad_b1: wgpu::Buffer,

    // Pre-built bind groups (buffers never change identity).
    bg_l1: wgpu::BindGroup,
    bg_l2: wgpu::BindGroup,
    bg_grad_w2: wgpu::BindGroup,
    bg_grad_b2: wgpu::BindGroup,
    bg_grad_h: wgpu::BindGroup,
    bg_grad_w1: wgpu::BindGroup,
    bg_grad_b1: wgpu::BindGroup,
}

/// Weight gradients produced by a backward pass, shaped like the weights.
pub struct Grads {
    pub w1: Array2<f32>,
    pub b1: Array1<f32>,
    pub w2: Array2<f32>,
    pub b2: Array1<f32>,
}

impl Grads {
    /// Global L2 norm over all four gradient tensors.
    pub fn norm(&self) -> f32 {
        let sq = |v: f32, x: &f32| v + x * x;
        (self.w1.iter().fold(0.0, sq)
            + self.b1.iter().fold(0.0, sq)
            + self.w2.iter().fold(0.0, sq)
            + self.b2.iter().fold(0.0, sq))
        .sqrt()
    }

    /// Scale all gradients down so the global norm is at most `max_norm`.
    /// `max_norm <= 0` disables clipping.
    pub fn clip_norm(&mut self, max_norm: f32) {
        let norm = self.norm();
        if max_norm > 0.0 && norm > max_norm {
            let scale = max_norm / norm;
            self.w1 *= scale;
            self.b1 *= scale;
            self.w2 *= scale;
            self.b2 *= scale;
        }
    }
}

fn pipeline(device: &wgpu::Device, label: &str, src: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn bind_group(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

impl GpuModel {
    pub fn new(
        w1: &Array2<f32>,
        b1: &Array1<f32>,
        w2: &Array2<f32>,
        b2: &Array1<f32>,
        max_batch: usize,
    ) -> Result<Self> {
        let (input, hidden) = w1.dim();
        let output = w2.dim().1;
        let max_batch = max_batch.max(1);

        // PRIMARY = Vulkan / Metal / DX12 only. Skipping the OpenGL/EGL
        // backend avoids noisy Mesa probe warnings (libEGL / ZINK) on Linux.
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("no compatible GPU adapter: {e}"))?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .map_err(|e| anyhow!("failed to acquire GPU device: {e}"))?;
        info!("GPU backend initialized on {}", adapter.get_info().name);

        let weight = |label: &str, data: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &f32_bytes(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
        };
        let scratch = |label: &str, len: usize, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | extra,
                mapped_at_creation: false,
            })
        };
        let staging = |label: &str, len: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (len * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let uniform = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let w1_buf = weight("w1", w1.as_slice().unwrap());
        let b1_buf = weight("b1", b1.as_slice().unwrap());
        let w2_buf = weight("w2", w2.as_slice().unwrap());
        let b2_buf = weight("b2", b2.as_slice().unwrap());

        let copy_dst = wgpu::BufferUsages::COPY_DST;
        let copy_src = wgpu::BufferUsages::COPY_SRC;
        let none = wgpu::BufferUsages::empty();

        let x = scratch("x", max_batch * input, copy_dst);
        let h = scratch("h", max_batch * hidden, none);
        let logits = scratch("logits", max_batch * output, copy_src);
        let grad_logits = scratch("grad_logits", max_batch * output, copy_dst);
        let grad_h = scratch("grad_h", max_batch * hidden, none);
        let grad_w1 = scratch("grad_w1", input * hidden, copy_src);
        let grad_b1 = scratch("grad_b1", hidden, copy_src);
        let grad_w2 = scratch("grad_w2", hidden * output, copy_src);
        let grad_b2 = scratch("grad_b2", output, copy_src);

        let staging_logits = staging("staging_logits", max_batch * output);
        let staging_w1 = staging("staging_w1", input * hidden);
        let staging_b1 = staging("staging_b1", hidden);
        let staging_w2 = staging("staging_w2", hidden * output);
        let staging_b2 = staging("staging_b2", output);

        let p_linear = pipeline(&device, "linear", include_str!("../forward.wgsl"));
        let p_grad_weights =
            pipeline(&device, "grad_weights", include_str!("../grad-weights.wgsl"));
        let p_grad_bias = pipeline(&device, "grad_bias", include_str!("../grad-bias.wgsl"));
        let p_grad_hidden =
            pipeline(&device, "grad_hidden", include_str!("../grad-hidden.wgsl"));

        let u_l1 = uniform("u_l1");
        let u_l2 = uniform("u_l2");
        let u_grad_w2 = uniform("u_grad_w2");
        let u_grad_b2 = uniform("u_grad_b2");
        let u_grad_h = uniform("u_grad_h");
        let u_grad_w1 = uniform("u_grad_w1");
        let u_grad_b1 = uniform("u_grad_b1");

        let bg_l1 = bind_group(&device, &p_linear, &[&u_l1, &x, &w1_buf, &b1_buf, &h]);
        let bg_l2 = bind_group(&device, &p_linear, &[&u_l2, &h, &w2_buf, &b2_buf, &logits]);
        let bg_grad_w2 = bind_group(
            &device,
            &p_grad_weights,
            &[&u_grad_w2, &h, &grad_logits, &grad_w2],
        );
        let bg_grad_b2 = bind_group(&device, &p_grad_bias, &[&u_grad_b2, &grad_logits, &grad_b2]);
        let bg_grad_h = bind_group(
            &device,
            &p_grad_hidden,
            &[&u_grad_h, &grad_logits, &w2_buf, &h, &grad_h],
        );
        let bg_grad_w1 = bind_group(&device, &p_grad_weights, &[&u_grad_w1, &x, &grad_h, &grad_w1]);
        let bg_grad_b1 = bind_group(&device, &p_grad_bias, &[&u_grad_b1, &grad_h, &grad_b1]);

        Ok(Self {
            device,
            queue,
            input,
            hidden,
            output,
            max_batch,
            w1: w1_buf,
            b1: b1_buf,
            w2: w2_buf,
            b2: b2_buf,
            x,
            logits,
            grad_logits,
            grad_w1,
            grad_b1,
            grad_w2,
            grad_b2,
            staging_logits,
            staging_w1,
            staging_b1,
            staging_w2,
            staging_b2,
            p_linear,
            p_grad_weights,
            p_grad_bias,
            p_grad_hidden,
            u_l1,
            u_l2,
            u_grad_w2,
            u_grad_b2,
            u_grad_h,
            u_grad_w1,
            u_grad_b1,
            bg_l1,
            bg_l2,
            bg_grad_w2,
            bg_grad_b2,
            bg_grad_h,
            bg_grad_w1,
            bg_grad_b1,
        })
    }

    fn write_params(&self, buf: &wgpu::Buffer, p: [u32; 4]) {
        self.queue.write_buffer(buf, 0, &params_bytes(p));
    }

    /// Re-upload CPU weights (used after `sanitize` or external edits).
    pub fn upload_weights(
        &self,
        w1: &Array2<f32>,
        b1: &Array1<f32>,
        w2: &Array2<f32>,
        b2: &Array1<f32>,
    ) {
        self.queue
            .write_buffer(&self.w1, 0, &f32_bytes(w1.as_slice().unwrap()));
        self.queue
            .write_buffer(&self.b1, 0, &f32_bytes(b1.as_slice().unwrap()));
        self.queue
            .write_buffer(&self.w2, 0, &f32_bytes(w2.as_slice().unwrap()));
        self.queue
            .write_buffer(&self.b2, 0, &f32_bytes(b2.as_slice().unwrap()));
    }

    fn read_staging(&self, buf: &wgpu::Buffer, count: usize) -> Result<Vec<f32>> {
        let slice = buf.slice(0..(count * 4) as u64);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| anyhow!("GPU poll failed: {e:?}"))?;
        rx.recv()
            .map_err(|e| anyhow!("GPU map channel closed: {e}"))?
            .map_err(|e| anyhow!("GPU buffer map failed: {e:?}"))?;
        let data = slice.get_mapped_range();
        let out = bytes_to_f32(&data);
        drop(data);
        buf.unmap();
        Ok(out)
    }

    /// Encode both linear layers (layer 1 with ReLU, layer 2 raw logits).
    fn encode_forward(&self, pass: &mut wgpu::ComputePass<'_>, batch: u32) {
        pass.set_pipeline(&self.p_linear);
        pass.set_bind_group(0, &self.bg_l1, &[]);
        pass.dispatch_workgroups(
            ceil_div(batch, WG_2D),
            ceil_div(self.hidden as u32, WG_2D),
            1,
        );
        pass.set_bind_group(0, &self.bg_l2, &[]);
        pass.dispatch_workgroups(
            ceil_div(batch, WG_2D),
            ceil_div(self.output as u32, WG_2D),
            1,
        );
    }

    pub fn forward(&self, x: &Array2<f32>) -> Result<Array2<f32>> {
        let batch = x.nrows();
        if batch > self.max_batch {
            return Err(anyhow!(
                "batch {batch} exceeds GPU capacity {}",
                self.max_batch
            ));
        }
        let xv: Vec<f32> = x.iter().cloned().collect();
        self.queue.write_buffer(&self.x, 0, &f32_bytes(&xv));
        let (b, i, h, o) = (
            batch as u32,
            self.input as u32,
            self.hidden as u32,
            self.output as u32,
        );
        self.write_params(&self.u_l1, [b, i, h, 1]);
        self.write_params(&self.u_l2, [b, h, o, 0]);

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            self.encode_forward(&mut pass, b);
        }
        enc.copy_buffer_to_buffer(
            &self.logits,
            0,
            &self.staging_logits,
            0,
            (batch * self.output * 4) as u64,
        );
        self.queue.submit([enc.finish()]);

        let data = self.read_staging(&self.staging_logits, batch * self.output)?;
        Ok(Array2::from_shape_vec((batch, self.output), data)?)
    }

    /// Backpropagate `grad_logits` (dL/dlogits, shape `batch × output`)
    /// through the network and return the weight gradients. The weights on
    /// the GPU are not changed; the caller applies the update on the CPU and
    /// re-uploads with `upload_weights`.
    pub fn backward(&self, x: &Array2<f32>, grad_logits: &Array2<f32>) -> Result<Grads> {
        let batch = x.nrows();
        if batch > self.max_batch {
            return Err(anyhow!(
                "batch {batch} exceeds GPU capacity {}",
                self.max_batch
            ));
        }
        let (b, i, h, o) = (
            batch as u32,
            self.input as u32,
            self.hidden as u32,
            self.output as u32,
        );

        let xv: Vec<f32> = x.iter().cloned().collect();
        let gv: Vec<f32> = grad_logits.iter().cloned().collect();
        self.queue.write_buffer(&self.x, 0, &f32_bytes(&xv));
        self.queue.write_buffer(&self.grad_logits, 0, &f32_bytes(&gv));

        self.write_params(&self.u_l1, [b, i, h, 1]);
        self.write_params(&self.u_l2, [b, h, o, 0]);
        self.write_params(&self.u_grad_w2, [b, h, o, 0]);
        self.write_params(&self.u_grad_b2, [b, o, 0, 0]);
        self.write_params(&self.u_grad_h, [b, h, o, 0]);
        self.write_params(&self.u_grad_w1, [b, i, h, 0]);
        self.write_params(&self.u_grad_b1, [b, h, 0, 0]);

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            // Recompute the hidden activations for this batch.
            pass.set_pipeline(&self.p_linear);
            pass.set_bind_group(0, &self.bg_l1, &[]);
            pass.dispatch_workgroups(ceil_div(b, WG_2D), ceil_div(h, WG_2D), 1);

            pass.set_pipeline(&self.p_grad_weights);
            pass.set_bind_group(0, &self.bg_grad_w2, &[]);
            pass.dispatch_workgroups(ceil_div(h, WG_2D), ceil_div(o, WG_2D), 1);

            pass.set_pipeline(&self.p_grad_bias);
            pass.set_bind_group(0, &self.bg_grad_b2, &[]);
            pass.dispatch_workgroups(ceil_div(o, WG_1D), 1, 1);

            pass.set_pipeline(&self.p_grad_hidden);
            pass.set_bind_group(0, &self.bg_grad_h, &[]);
            pass.dispatch_workgroups(ceil_div(b, WG_2D), ceil_div(h, WG_2D), 1);

            pass.set_pipeline(&self.p_grad_weights);
            pass.set_bind_group(0, &self.bg_grad_w1, &[]);
            pass.dispatch_workgroups(ceil_div(i, WG_2D), ceil_div(h, WG_2D), 1);

            pass.set_pipeline(&self.p_grad_bias);
            pass.set_bind_group(0, &self.bg_grad_b1, &[]);
            pass.dispatch_workgroups(ceil_div(h, WG_1D), 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.grad_w1, 0, &self.staging_w1, 0, (i * h * 4) as u64);
        enc.copy_buffer_to_buffer(&self.grad_b1, 0, &self.staging_b1, 0, (h * 4) as u64);
        enc.copy_buffer_to_buffer(&self.grad_w2, 0, &self.staging_w2, 0, (h * o * 4) as u64);
        enc.copy_buffer_to_buffer(&self.grad_b2, 0, &self.staging_b2, 0, (o * 4) as u64);
        self.queue.submit([enc.finish()]);

        let gw1 = self.read_staging(&self.staging_w1, self.input * self.hidden)?;
        let gb1 = self.read_staging(&self.staging_b1, self.hidden)?;
        let gw2 = self.read_staging(&self.staging_w2, self.hidden * self.output)?;
        let gb2 = self.read_staging(&self.staging_b2, self.output)?;
        Ok(Grads {
            w1: Array2::from_shape_vec((self.input, self.hidden), gw1)?,
            b1: Array1::from_vec(gb1),
            w2: Array2::from_shape_vec((self.hidden, self.output), gw2)?,
            b2: Array1::from_vec(gb2),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Axis;
    use ndarray_rand::rand::distributions::Uniform;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::RandomExt;

    #[test]
    fn gpu_matches_cpu() {
        let (input, hidden, output, batch) = (8usize, 16usize, 3usize, 4usize);
        let mut rng = ndarray_rand::rand::rngs::StdRng::seed_from_u64(42);
        let dist = Uniform::new(-0.5, 0.5);
        let w1 = Array2::random_using((input, hidden), dist, &mut rng);
        let b1 = Array1::random_using(hidden, dist, &mut rng);
        let w2 = Array2::random_using((hidden, output), dist, &mut rng);
        let b2 = Array1::random_using(output, dist, &mut rng);
        let x = Array2::random_using((batch, input), dist, &mut rng);
        let grad_logits = Array2::random_using((batch, output), dist, &mut rng);

        let gpu = match GpuModel::new(&w1, &b1, &w2, &b2, batch) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU test (no adapter): {e}");
                return;
            }
        };

        // Forward parity.
        let h_pre = x.dot(&w1) + &b1;
        let h = h_pre.mapv(|v| v.max(0.0));
        let logits_cpu = h.dot(&w2) + &b2;
        let logits_gpu = gpu.forward(&x).unwrap();
        for (a, b) in logits_cpu.iter().zip(logits_gpu.iter()) {
            assert!((a - b).abs() < 1e-4, "forward mismatch: {a} vs {b}");
        }

        // Backward parity.
        let grad_h = grad_logits.dot(&w2.t()) * &h_pre.mapv(|v| if v > 0.0 { 1.0f32 } else { 0.0 });
        let expected = Grads {
            w1: x.t().dot(&grad_h),
            b1: grad_h.sum_axis(Axis(0)),
            w2: h.t().dot(&grad_logits),
            b2: grad_logits.sum_axis(Axis(0)),
        };
        let got = gpu.backward(&x, &grad_logits).unwrap();

        let pairs: [(&str, Vec<f32>, Vec<f32>); 4] = [
            ("w1", got.w1.iter().cloned().collect(), expected.w1.iter().cloned().collect()),
            ("b1", got.b1.iter().cloned().collect(), expected.b1.iter().cloned().collect()),
            ("w2", got.w2.iter().cloned().collect(), expected.w2.iter().cloned().collect()),
            ("b2", got.b2.iter().cloned().collect(), expected.b2.iter().cloned().collect()),
        ];
        for (name, got, expected) in pairs {
            for (g, e) in got.iter().zip(expected.iter()) {
                assert!((g - e).abs() < 1e-4, "{name} mismatch: {g} vs {e}");
            }
        }
    }
}
