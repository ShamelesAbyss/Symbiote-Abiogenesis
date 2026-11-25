//! Interactive GPU Graphical Observatory for Symbiote: Project Abiogenesis — Convergence.
//!
//! Renders the continuous planetary EcoField (nutrients, living roots, pheromones, toxins)
//! and every evolving organism's localized 24x24 continuous 2-channel Lenia cellular body
//! with smooth bioluminescent membranes, internal cytoplasm gradients, and seamless zooming
//! between ecosystem-wide and microscopic cellular scales.

use crate::convergence::eco_field::{ECO_HEIGHT, ECO_WIDTH};
use crate::convergence::engine::{ConvergenceWorld, MAX_ORGANISMS};
use crate::convergence::lenia_body::{BODY_CHANNELS, BODY_RES};
use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniforms {
    center_x: f32,
    center_y: f32,
    zoom: f32,
    aspect: f32,
    time: f32,
    org_count: u32,
    eco_w: u32,
    eco_h: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuOrganismMeta {
    x: f32,
    y: f32,
    radius: f32,
    hue: f32,
    energy_norm: f32,
    pulse: f32,
    _pad0: f32,
    _pad1: f32,
}

pub fn run_observatory(prefer_gpu_compute: bool) -> Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = ObservatoryApp::new(prefer_gpu_compute);
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct ObservatoryGpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    cam_buf: wgpu::Buffer,
    eco_buf: wgpu::Buffer,
    org_meta_buf: wgpu::Buffer,
    org_bodies_buf: wgpu::Buffer,
}

struct ObservatoryApp {
    world: ConvergenceWorld,
    window: Option<Arc<Window>>,
    gpu: Option<ObservatoryGpuState>,
    cam_x: f32,
    cam_y: f32,
    cam_zoom: f32,
    paused: bool,
    start_time: Instant,
    last_title_update: Instant,
    frame_count: u32,
}

impl ObservatoryApp {
    fn new(prefer_gpu_compute: bool) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xAB10_6E4E);
        Self {
            world: ConvergenceWorld::new(seed, prefer_gpu_compute),
            window: None,
            gpu: None,
            cam_x: 0.0,
            cam_y: 0.0,
            cam_zoom: 1.0,
            paused: false,
            start_time: Instant::now(),
            last_title_update: Instant::now(),
            frame_count: 0,
        }
    }

    fn init_gpu(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("no compatible GPU adapter for observatory");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Symbiote Observatory Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("request_device failed");

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let cam_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniforms"),
            size: std::mem::size_of::<CameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let eco_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("EcoField Render Buffer"),
            size: (ECO_WIDTH * ECO_HEIGHT * 4 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let org_meta_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Meta Buffer"),
            size: (MAX_ORGANISMS * std::mem::size_of::<GpuOrganismMeta>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let org_bodies_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Lenia Bodies Buffer"),
            size: (MAX_ORGANISMS * BODY_RES * BODY_RES * BODY_CHANNELS * std::mem::size_of::<f32>())
                as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Observatory Bind Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_frag(1),
                storage_frag(2),
                storage_frag(3),
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Observatory Bind Group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cam_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: eco_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: org_meta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: org_bodies_buf.as_entire_binding(),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Observatory Shader"),
            source: wgpu::ShaderSource::Wgsl(OBSERVATORY_WGSL.into()),
        });

        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Observatory Pipe Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Observatory Pipeline"),
            layout: Some(&pipe_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        self.gpu = Some(ObservatoryGpuState {
            surface,
            device,
            queue,
            config,
            pipeline,
            bind_group,
            cam_buf,
            eco_buf,
            org_meta_buf,
            org_bodies_buf,
        });
    }

    fn render_frame(&mut self) {
        if !self.paused {
            self.world.step();
        }

        let Some(gpu) = &self.gpu else { return };
        let org_count = self.world.organisms.len().min(MAX_ORGANISMS);

        let cam = CameraUniforms {
            center_x: self.cam_x,
            center_y: self.cam_y,
            zoom: self.cam_zoom,
            aspect: gpu.config.width as f32 / gpu.config.height.max(1) as f32,
            time: self.start_time.elapsed().as_secs_f32(),
            org_count: org_count as u32,
            eco_w: ECO_WIDTH as u32,
            eco_h: ECO_HEIGHT as u32,
        };
        gpu.queue
            .write_buffer(&gpu.cam_buf, 0, bytemuck::bytes_of(&cam));
        gpu.queue.write_buffer(
            &gpu.eco_buf,
            0,
            bytemuck::cast_slice(&self.world.eco_field.cells),
        );

        let mut metas = Vec::with_capacity(org_count);
        let mut bodies = Vec::with_capacity(org_count * BODY_RES * BODY_RES * BODY_CHANNELS);
        for org in self.world.organisms.iter().take(org_count) {
            metas.push(GpuOrganismMeta {
                x: org.x,
                y: org.y,
                radius: 0.055 + org.body.metrics.radius_norm * 0.055,
                hue: org.morph_genome.biolum_hue,
                energy_norm: (org.energy / 100.0).clamp(0.1, 1.4),
                pulse: org.body.pulse_phase,
                _pad0: 0.0,
                _pad1: 0.0,
            });
            bodies.extend_from_slice(&org.body.cells);
        }
        if !metas.is_empty() {
            gpu.queue
                .write_buffer(&gpu.org_meta_buf, 0, bytemuck::cast_slice(&metas));
            gpu.queue
                .write_buffer(&gpu.org_bodies_buf, 0, bytemuck::cast_slice(&bodies));
        }

        let Ok(frame) = gpu.surface.get_current_texture() else {
            return;
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Observatory Frame Encoder"),
            });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Observatory Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            rpass.set_pipeline(&gpu.pipeline);
            rpass.set_bind_group(0, &gpu.bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
        gpu.queue.submit(Some(encoder.finish()));
        frame.present();

        self.frame_count += 1;
        if self.last_title_update.elapsed().as_secs_f32() >= 0.5 {
            let s = self.world.stats();
            let fps = self.frame_count as f32 / self.last_title_update.elapsed().as_secs_f32();
            if let Some(win) = &self.window {
                win.set_title(&format!(
                    "SYMBIOTE: ABIOGENESIS CONVERGENCE | {:.0} FPS | Orgs: {} | Lineages: {} | Gen: {} | Zoom: {:.1}x | Env: {} | [Space:Pause Scroll:Zoom Arrows:Pan 0:Reset]",
                    fps,
                    s.organism_count,
                    s.active_lineages,
                    s.max_generation,
                    self.cam_zoom,
                    self.world.symbiote.environment.name(),
                ));
            }
            self.frame_count = 0;
            self.last_title_update = Instant::now();
        }
    }
}

impl ApplicationHandler for ObservatoryApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("SYMBIOTE: PROJECT ABIOGENESIS — CONVERGENCE OBSERVATORY")
                .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 800.0));
            let win = Arc::new(event_loop.create_window(attrs).expect("create window"));
            self.init_gpu(win.clone());
            self.window = Some(win);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = new_size.width.max(1);
                    gpu.config.height = new_size.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y / 40.0) as f32,
                };
                let factor = if scroll > 0.0 { 1.15 } else { 1.0 / 1.15 };
                self.cam_zoom = (self.cam_zoom * factor).clamp(0.6, 18.0);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                let pan = 0.08 / self.cam_zoom;
                match code {
                    KeyCode::Escape | KeyCode::KeyQ => event_loop.exit(),
                    KeyCode::Space => self.paused = !self.paused,
                    KeyCode::ArrowLeft | KeyCode::KeyA => self.cam_x -= pan,
                    KeyCode::ArrowRight | KeyCode::KeyD => self.cam_x += pan,
                    KeyCode::ArrowUp | KeyCode::KeyW => self.cam_y -= pan,
                    KeyCode::ArrowDown | KeyCode::KeyS => self.cam_y += pan,
                    KeyCode::Equal => self.cam_zoom = (self.cam_zoom * 1.2).clamp(0.6, 18.0),
                    KeyCode::Minus => self.cam_zoom = (self.cam_zoom / 1.2).clamp(0.6, 18.0),
                    KeyCode::Digit0 => {
                        self.cam_x = 0.0;
                        self.cam_y = 0.0;
                        self.cam_zoom = 1.0;
                    }
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => {
                self.render_frame();
                if let Some(win) = &self.window {
                    win.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(win) = &self.window {
            win.request_redraw();
        }
    }
}

fn storage_frag(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

const OBSERVATORY_WGSL: &str = r#"
struct Camera {
    center_x: f32,
    center_y: f32,
    zoom: f32,
    aspect: f32,
    time: f32,
    org_count: u32,
    eco_w: u32,
    eco_h: u32,
};

struct OrgMeta {
    x: f32,
    y: f32,
    radius: f32,
    hue: f32,
    energy_norm: f32,
    pulse: f32,
    pad0: f32,
    pad1: f32,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> eco: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> orgs: array<OrgMeta>;
@group(0) @binding(3) var<storage, read> bodies: array<vec2<f32>>;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var pts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    var out: VsOut;
    let p = pts[vi];
    out.pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let hp = fract(h) * 6.0;
    let x = c * (1.0 - abs(hp % 2.0 - 1.0));
    var rgb = vec3<f32>(0.0);
    if (hp < 1.0) { rgb = vec3<f32>(c, x, 0.0); }
    else if (hp < 2.0) { rgb = vec3<f32>(x, c, 0.0); }
    else if (hp < 3.0) { rgb = vec3<f32>(0.0, c, x); }
    else if (hp < 4.0) { rgb = vec3<f32>(0.0, x, c); }
    else if (hp < 5.0) { rgb = vec3<f32>(x, 0.0, c); }
    else { rgb = vec3<f32>(c, 0.0, x); }
    return rgb + vec3<f32>(v - c);
}

fn sample_eco(wx: f32, wy: f32) -> vec4<f32> {
    let u = clamp((wx + 1.0) * 0.5, 0.0, 0.999) * f32(cam.eco_w - 1u);
    let v = clamp((wy + 1.0) * 0.5, 0.0, 0.999) * f32(cam.eco_h - 1u);
    let x0 = u32(floor(u));
    let y0 = u32(floor(v));
    let x1 = min(x0 + 1u, cam.eco_w - 1u);
    let y1 = min(y0 + 1u, cam.eco_h - 1u);
    let fx = fract(u);
    let fy = fract(v);

    let c00 = eco[y0 * cam.eco_w + x0];
    let c10 = eco[y0 * cam.eco_w + x1];
    let c01 = eco[y1 * cam.eco_w + x0];
    let c11 = eco[y1 * cam.eco_w + x1];
    return mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);
}

fn sample_body(org_idx: u32, lu: f32, lv: f32) -> vec2<f32> {
    let res = 24u;
    let fx_pos = clamp(lu, 0.0, 0.999) * f32(res - 1u);
    let fy_pos = clamp(lv, 0.0, 0.999) * f32(res - 1u);
    let x0 = u32(floor(fx_pos));
    let y0 = u32(floor(fy_pos));
    let x1 = min(x0 + 1u, res - 1u);
    let y1 = min(y0 + 1u, res - 1u);
    let fx = fract(fx_pos);
    let fy = fract(fy_pos);

    let base = org_idx * res * res;
    let b00 = bodies[base + y0 * res + x0];
    let b10 = bodies[base + y0 * res + x1];
    let b01 = bodies[base + y1 * res + x0];
    let b11 = bodies[base + y1 * res + x1];
    return mix(mix(b00, b10, fx), mix(b01, b11, fx), fy);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let ndc = (in.uv - vec2<f32>(0.5, 0.5)) * 2.0;
    let wx = cam.center_x + (ndc.x * cam.aspect) / cam.zoom;
    let wy = cam.center_y + ndc.y / cam.zoom;

    // Deep abyssal background with continuous reaction-diffusion fields
    var col = vec3<f32>(0.012, 0.018, 0.032);
    if (abs(wx) <= 1.0 && abs(wy) <= 1.0) {
        let e = sample_eco(wx, wy);
        // Nutrient: soft bioluminescent teal-emerald abyssal current
        col += vec3<f32>(0.03, 0.16, 0.18) * e.x;
        // Pheromone: ethereal violet-cyan mist
        col += vec3<f32>(0.14, 0.06, 0.24) * e.y;
        // Toxin: warning crimson-amber haze
        col += vec3<f32>(0.32, 0.05, 0.08) * e.z;
        // Living root network: luminous sapphire-gold filaments
        col += vec3<f32>(0.10, 0.34, 0.68) * pow(e.w, 1.3);
    }

    // Composite localized continuous Lenia cellular organisms
    for (var i: u32 = 0u; i < cam.org_count; i = i + 1u) {
        let o = orgs[i];
        let dx = wx - o.x;
        let dy = wy - o.y;
        if (abs(dx) > o.radius || abs(dy) > o.radius) {
            continue;
        }
        let lu = (dx / o.radius) * 0.5 + 0.5;
        let lv = (dy / o.radius) * 0.5 + 0.5;
        let cell = sample_body(i, lu, lv);
        let cyto = cell.x;
        let mem = cell.y;

        if (cyto > 0.01 || mem > 0.01) {
            let breath = 0.88 + 0.12 * sin(o.pulse + cam.time * 2.0);
            let core_rgb = hsv_to_rgb(o.hue, 0.72, 0.95 * breath * o.energy_norm);
            let mem_rgb = hsv_to_rgb(fract(o.hue + 0.14), 0.45, 1.25);

            // Translucent cytoplasm + bright refractive membrane rim
            let cyto_alpha = clamp(cyto * 0.92, 0.0, 0.95);
            col = mix(col, core_rgb * (0.45 + 0.85 * cyto), cyto_alpha);
            col += mem_rgb * mem * 0.65;
        }
    }

    // Subtle tone-mapping for rich bioluminescent glow
    col = col / (vec3<f32>(1.0) + col * 0.35);
    return vec4<f32>(pow(col, vec3<f32>(0.92)), 1.0);
}
"#;
