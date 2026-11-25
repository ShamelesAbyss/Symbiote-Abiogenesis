//! GPU-accelerated compute engine (wgpu 24 + WGSL) for Project Convergence.
//!
//! Runs the continuous 4-channel reaction-diffusion EcoField on the GPU
//! (targeting NVIDIA RTX 4070 Laptop GPU) with automatic CPU fallback.

use crate::app::Environment;
use crate::convergence::eco_field::{EcoField, ECO_CHANNELS};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EcoUniforms {
    width: u32,
    height: u32,
    regen_rate: f32,
    _pad: f32,
}

pub struct ConvergenceGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    eco_pipeline: wgpu::ComputePipeline,
    eco_layout: wgpu::BindGroupLayout,
    eco_src: wgpu::Buffer,
    eco_dst: wgpu::Buffer,
    eco_readback: wgpu::Buffer,
    eco_uniforms: wgpu::Buffer,
    width: u32,
    height: u32,
    pub adapter_name: String,
    pub backend: String,
}

impl ConvergenceGpu {
    pub fn try_new(width: u32, height: u32, _max_organisms: u32) -> Result<Self, String> {
        pollster::block_on(Self::new_async(width, height))
    }

    async fn new_async(width: u32, height: u32) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "no GPU adapter found".to_string())?;

        let info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Symbiote Convergence GPU Compute"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| format!("request_device failed: {e:?}"))?;

        let byte_size = (width as usize * height as usize * ECO_CHANNELS * std::mem::size_of::<f32>()) as u64;

        let eco_src = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("EcoField Src"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let eco_dst = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("EcoField Dst"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let eco_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("EcoField Readback"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let eco_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("EcoField Uniforms"),
            size: std::mem::size_of::<EcoUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("EcoField Reaction-Diffusion WGSL"),
            source: wgpu::ShaderSource::Wgsl(ECO_FIELD_WGSL.into()),
        });

        let eco_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("EcoField Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("EcoField Pipeline Layout"),
            bind_group_layouts: &[&eco_layout],
            push_constant_ranges: &[],
        });

        let eco_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("EcoField Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            eco_pipeline,
            eco_layout,
            eco_src,
            eco_dst,
            eco_readback,
            eco_uniforms,
            width,
            height,
            adapter_name: info.name,
            backend: format!("{:?}", info.backend),
        })
    }

    pub fn step_eco_field(&self, field: &mut EcoField, env: Environment) -> Result<(), String> {
        let regen_rate = match env {
            Environment::Bloom => 0.0065,
            Environment::Calm => 0.0038,
            Environment::Drift => 0.0032,
            Environment::Storm => 0.0028,
            Environment::Hunger => 0.0014,
        };
        let uniforms = EcoUniforms {
            width: self.width,
            height: self.height,
            regen_rate,
            _pad: 0.0,
        };

        self.queue
            .write_buffer(&self.eco_src, 0, bytemuck::cast_slice(&field.cells));
        self.queue
            .write_buffer(&self.eco_uniforms, 0, bytemuck::bytes_of(&uniforms));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("EcoField Bind Group"),
            layout: &self.eco_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.eco_src.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.eco_dst.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.eco_uniforms.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("EcoField Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("EcoField Compute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.eco_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, 1);
        }

        let byte_size = (field.cells.len() * std::mem::size_of::<f32>()) as u64;
        encoder.copy_buffer_to_buffer(&self.eco_dst, 0, &self.eco_readback, 0, byte_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.eco_readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|e| format!("recv error: {e:?}"))?
            .map_err(|e| format!("map error: {e:?}"))?;

        {
            let mapped = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&mapped);
            field.cells.copy_from_slice(floats);
        }
        self.eco_readback.unmap();
        field.tick += 1;
        Ok(())
    }
}

const ECO_FIELD_WGSL: &str = r#"
struct EcoUniforms {
    width: u32,
    height: u32,
    regen_rate: f32,
    pad: f32,
};

@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> u: EcoUniforms;

fn idx(x: u32, y: u32) -> u32 {
    return y * u.width + x;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= u.width || y >= u.height) {
        return;
    }

    let xl = select(x - 1u, 0u, x == 0u);
    let xr = min(x + 1u, u.width - 1u);
    let yu = select(y - 1u, 0u, y == 0u);
    let yd = min(y + 1u, u.height - 1u);

    let c = src[idx(x, y)];
    let up = src[idx(x, yu)];
    let dn = src[idx(x, yd)];
    let lt = src[idx(xl, y)];
    let rt = src[idx(xr, y)];

    let lap = (up + dn + lt + rt) * 0.25 - c;

    var root_grow = 0.0;
    if (dn.w > 0.35 && c.x > 0.22 && c.z < 0.35) {
        root_grow = 0.0025 * dn.w * c.x;
    }
    let next_root = clamp(c.w * 0.9992 + root_grow, 0.0, 1.0);
    let next_n = clamp(c.x + lap.x * 0.18 + u.regen_rate * (1.0 - c.x) + c.w * 0.008 * (1.0 - c.x) - c.z * 0.004, 0.0, 1.0);
    let next_p = clamp((c.y + lap.y * 0.22) * 0.975, 0.0, 1.0);
    let next_t = clamp((c.z + lap.z * 0.14) * 0.968, 0.0, 1.0);

    dst[idx(x, y)] = vec4<f32>(next_n, next_p, next_t, next_root);
}
"#;
