//! Symbiote: Abiogenesis v0.4.0 — Project GENOMORPH Research Laboratory.
//!
//! Renders BOTH the smooth 40x40x3 continuous Lenia cellular bodies AND each organism's
//! live articulated procedural anatomy (segments, fins, tails, jaws, antennae, carapaces,
//! filter veils) in WGSL, paired with a custom obsidian-glass `egui` 0.31 workspace:
//! - Live interactive NEAT `NeuralGraph` visualizer
//! - Parent-offspring / ancestor anatomical homology diff (`diff_genomes`)
//! - Developmental stage, digestive biology & functional organ inspector
//! - MAP-Elites Discovery Archive with explicit `[USER INTERVENTION]` resurrection
//! - Fullscreen (`F11` / `Alt+Enter`), minimize guards, zero-churn buffers & crash logging

use crate::convergence::crash_log;
use crate::convergence::eco_field::{ECO_HEIGHT, ECO_WIDTH};
use crate::convergence::engine::{toroidal_delta, ConvergenceWorld, MAX_ORGANISMS_BUFFER};
use crate::convergence::genomorph::{diff_genomes, MAX_PHENOTYPE_ORGANS};
use crate::convergence::lenia_body::{BODY_CHANNELS, BODY_RES};
use crate::convergence::neural_graph::{NeuralGraph, NodeKind};
use crate::convergence::persistence::{load_world, save_world_atomic};
use crate::convergence::phylogeny::DiscoveryEntry;
use crate::convergence::snapshot::render_png;
use crate::convergence::universes::UniversePreset;
use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
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
    selected_idx: i32,
    _pad0: i32,
    _pad1: i32,
    _pad2: i32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuOrganismMeta {
    x: f32,
    y: f32,
    radius: f32,
    hue: f32,
    accent_shift: f32,
    energy_norm: f32,
    pulse: f32,
    mass: f32,
    heading: f32,
    organ_start: u32,
    organ_count: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuOrgan {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    width: f32,
    hue_shift: f32,
    kind_code: u32,
    _pad: u32,
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
    organs_buf: wgpu::Buffer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

struct ObservatoryApp {
    world: ConvergenceWorld,
    prefer_gpu_compute: bool,
    window: Option<Arc<Window>>,
    gpu: Option<ObservatoryGpuState>,
    cam_x: f32,
    cam_y: f32,
    cam_zoom: f32,
    paused: bool,
    single_step: bool,
    steps_per_frame: u32,
    cinematic_mode: bool,
    selected_org_id: Option<u64>,
    track_selected: bool,
    selected_preset: UniversePreset,
    status_toast: String,
    cursor_pos: (f32, f32),
    pop_history: VecDeque<f32>,
    nutr_history: VecDeque<f32>,
    is_minimized: bool,
    alt_held: bool,
    metas_scratch: Vec<GpuOrganismMeta>,
    bodies_scratch: Vec<f32>,
    organs_scratch: Vec<GpuOrgan>,
    start_time: Instant,
    last_fps_update: Instant,
    frame_count: u32,
    current_fps: f32,
}

impl ObservatoryApp {
    fn new(prefer_gpu_compute: bool) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xAB10_6E4E);
        Self {
            world: ConvergenceWorld::new(seed, prefer_gpu_compute),
            prefer_gpu_compute,
            window: None,
            gpu: None,
            cam_x: 0.0,
            cam_y: 0.0,
            cam_zoom: 1.80,
            paused: false,
            single_step: false,
            steps_per_frame: 1,
            cinematic_mode: false,
            selected_org_id: None,
            track_selected: false,
            selected_preset: UniversePreset::PrimordialOcean,
            status_toast: "Click any organism to inspect anatomy & NeuralGraph · F11 Fullscreen"
                .to_string(),
            cursor_pos: (0.0, 0.0),
            pop_history: VecDeque::with_capacity(120),
            nutr_history: VecDeque::with_capacity(120),
            is_minimized: false,
            alt_held: false,
            metas_scratch: Vec::with_capacity(MAX_ORGANISMS_BUFFER),
            bodies_scratch: Vec::with_capacity(MAX_ORGANISMS_BUFFER * BODY_RES * BODY_RES * 4),
            organs_scratch: Vec::with_capacity(MAX_ORGANISMS_BUFFER * MAX_PHENOTYPE_ORGANS),
            start_time: Instant::now(),
            last_fps_update: Instant::now(),
            frame_count: 0,
            current_fps: 0.0,
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
                label: Some("Symbiote Abiogenesis GENOMORPH Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("request_device failed");

        let max_dim = device.limits().max_texture_dimension_2d;
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.clamp(1, max_dim),
            height: size.height.clamp(1, max_dim),
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
            size: (MAX_ORGANISMS_BUFFER * std::mem::size_of::<GpuOrganismMeta>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let org_bodies_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism 40x40x3 Lenia Bodies Buffer"),
            size: (MAX_ORGANISMS_BUFFER * BODY_RES * BODY_RES * 4 * std::mem::size_of::<f32>())
                as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let organs_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism Procedural Anatomy Buffer"),
            size: (MAX_ORGANISMS_BUFFER * MAX_PHENOTYPE_ORGANS * std::mem::size_of::<GpuOrgan>())
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
                storage_frag(4),
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
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: organs_buf.as_entire_binding(),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Observatory GENOMORPH Shader"),
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

        let egui_ctx = egui::Context::default();
        configure_laboratory_theme(&egui_ctx);
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, None, 1, false);

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
            organs_buf,
            egui_ctx,
            egui_state,
            egui_renderer,
        });
    }

    fn toggle_borderless_fullscreen(&self) {
        if let Some(win) = &self.window {
            if win.fullscreen().is_some() {
                win.set_fullscreen(None);
                crash_log::log_event("exited borderless fullscreen");
            } else {
                win.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                crash_log::log_event("entered borderless fullscreen");
            }
        }
    }

    fn select_organism_at_cursor(&mut self) {
        let Some(gpu) = &self.gpu else { return };
        let w = gpu.config.width.max(1) as f32;
        let h = gpu.config.height.max(1) as f32;
        let aspect = w / h;
        let u = self.cursor_pos.0 / w;
        let v = self.cursor_pos.1 / h;
        let ndc_x = (u - 0.5) * 2.0;
        let ndc_y = (v - 0.5) * 2.0;
        let wx = self.cam_x + (ndc_x * aspect) / self.cam_zoom;
        let wy = self.cam_y + ndc_y / self.cam_zoom;

        let mut best: Option<(u64, f32)> = None;
        for org in &self.world.organisms {
            let dx = toroidal_delta(org.x - wx);
            let dy = toroidal_delta(org.y - wy);
            let d = (dx * dx + dy * dy).sqrt();
            let r = 0.085 + org.body.metrics.radius_norm * 0.075;
            if d <= r && best.map(|(_, bd)| d < bd).unwrap_or(true) {
                best = Some((org.id, d));
            }
        }
        self.selected_org_id = best.map(|(id, _)| id);
        if let Some(id) = self.selected_org_id {
            self.status_toast = format!("Inspecting Organism #{id}");
        }
    }

    fn render_frame(&mut self) {
        if !self.paused {
            for _ in 0..self.steps_per_frame {
                self.world.step();
            }
        } else if self.single_step {
            self.world.step();
            self.single_step = false;
        }

        if self.world.tick % 4 == 0 {
            if self.pop_history.len() >= 120 {
                self.pop_history.pop_front();
                self.nutr_history.pop_front();
            }
            self.pop_history.push_back(self.world.organisms.len() as f32);
            self.nutr_history.push_back(self.world.eco_field.mean_nutrient());
        }

        if self.track_selected {
            if let Some(id) = self.selected_org_id {
                if let Some(org) = self.world.organisms.iter().find(|o| o.id == id) {
                    self.cam_x += toroidal_delta(org.x - self.cam_x) * 0.16;
                    self.cam_y += toroidal_delta(org.y - self.cam_y) * 0.16;
                }
            }
        }

        let Some(win) = self.window.clone() else {
            return;
        };
        if self.is_minimized || win.is_minimized() == Some(true) {
            return;
        }
        let inner = win.inner_size();
        if inner.width == 0 || inner.height == 0 {
            self.is_minimized = true;
            return;
        }
        let Some(gpu) = &mut self.gpu else { return };

        let max_dim = gpu.device.limits().max_texture_dimension_2d;
        let target_w = inner.width.clamp(1, max_dim);
        let target_h = inner.height.clamp(1, max_dim);
        if gpu.config.width != target_w || gpu.config.height != target_h {
            gpu.config.width = target_w;
            gpu.config.height = target_h;
            gpu.surface.configure(&gpu.device, &gpu.config);
            crash_log::log_event(&format!(
                "surface synchronized to {}x{} (scale={:.2})",
                target_w,
                target_h,
                win.scale_factor()
            ));
        }

        let org_count = self.world.organisms.len().min(MAX_ORGANISMS_BUFFER);
        let selected_idx = self
            .selected_org_id
            .and_then(|id| {
                self.world
                    .organisms
                    .iter()
                    .take(org_count)
                    .position(|o| o.id == id)
            })
            .map(|i| i as i32)
            .unwrap_or(-1);

        let cam = CameraUniforms {
            center_x: self.cam_x,
            center_y: self.cam_y,
            zoom: self.cam_zoom,
            aspect: gpu.config.width as f32 / gpu.config.height.max(1) as f32,
            time: self.start_time.elapsed().as_secs_f32(),
            org_count: org_count as u32,
            eco_w: ECO_WIDTH as u32,
            eco_h: ECO_HEIGHT as u32,
            selected_idx,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        gpu.queue
            .write_buffer(&gpu.cam_buf, 0, bytemuck::bytes_of(&cam));
        gpu.queue.write_buffer(
            &gpu.eco_buf,
            0,
            bytemuck::cast_slice(&self.world.eco_field.cells),
        );

        self.metas_scratch.clear();
        self.bodies_scratch.clear();
        self.organs_scratch.clear();
        for org in self.world.organisms.iter().take(org_count) {
            let organ_start = self.organs_scratch.len() as u32;
            let organ_count = org.phenotype.organs.len().min(MAX_PHENOTYPE_ORGANS) as u32;
            for organ in org.phenotype.organs.iter().take(MAX_PHENOTYPE_ORGANS) {
                self.organs_scratch.push(GpuOrgan {
                    x0: organ.x0,
                    y0: organ.y0,
                    x1: organ.x1,
                    y1: organ.y1,
                    width: organ.width,
                    hue_shift: organ.hue_shift,
                    kind_code: organ.kind.code(),
                    _pad: 0,
                });
            }
            let radius = (0.055 + org.body.metrics.radius_norm * 0.058)
                * (0.45 + org.maturity * 0.55)
                * org.genomorph.base_scale;
            self.metas_scratch.push(GpuOrganismMeta {
                x: org.x,
                y: org.y,
                radius,
                hue: org.morph_genome.biolum_hue,
                accent_shift: org.morph_genome.accent_shift,
                energy_norm: (org.energy / 95.0).clamp(0.30, 1.18),
                pulse: org.body.pulse_phase,
                mass: org.body.metrics.mass,
                heading: org.body.heading,
                organ_start,
                organ_count,
                _pad: 0,
            });
            for chunk in org.body.cells.chunks_exact(BODY_CHANNELS) {
                self.bodies_scratch.push(chunk[0]);
                self.bodies_scratch.push(chunk[1]);
                self.bodies_scratch.push(chunk[2]);
                self.bodies_scratch.push(0.0);
            }
        }
        if !self.metas_scratch.is_empty() {
            gpu.queue.write_buffer(
                &gpu.org_meta_buf,
                0,
                bytemuck::cast_slice(&self.metas_scratch),
            );
            gpu.queue.write_buffer(
                &gpu.org_bodies_buf,
                0,
                bytemuck::cast_slice(&self.bodies_scratch),
            );
        }
        if !self.organs_scratch.is_empty() {
            gpu.queue.write_buffer(
                &gpu.organs_buf,
                0,
                bytemuck::cast_slice(&self.organs_scratch),
            );
        }

        let raw_input = gpu.egui_state.take_egui_input(&win);
        let stats = self.world.stats();
        let mut action_new_universe: Option<UniversePreset> = None;
        let mut action_save = false;
        let mut action_load = false;
        let mut action_screenshot = false;
        let mut action_resurrect: Option<DiscoveryEntry> = None;

        let full_output = gpu.egui_ctx.run(raw_input, |ctx| {
            if self.cinematic_mode {
                egui::Area::new(egui::Id::new("cinematic_pill"))
                    .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(18.0, -18.0))
                    .show(ctx, |ui| {
                        egui::Frame::NONE
                            .fill(egui::Color32::from_rgba_unmultiplied(6, 10, 18, 170))
                            .stroke(egui::Stroke::new(
                                1.0,
                                egui::Color32::from_rgba_unmultiplied(56, 249, 215, 55),
                            ))
                            .corner_radius(egui::CornerRadius::same(14))
                            .inner_margin(egui::Margin::symmetric(14, 7))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new("◉ GENOMORPH")
                                            .strong()
                                            .color(egui::Color32::from_rgb(56, 249, 215)),
                                    );
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "· {} orgs · Gen {} · Avg {:.1} genes · H: Lab UI · F11: Fullscreen",
                                            stats.organism_count, stats.max_generation, stats.mean_genes
                                        ))
                                        .small()
                                        .color(egui::Color32::from_rgb(160, 185, 210)),
                                    );
                                });
                            });
                    });
                return;
            }

            let panel_frame = egui::Frame::side_top_panel(&ctx.style())
                .fill(egui::Color32::from_rgba_unmultiplied(7, 11, 19, 228))
                .stroke(egui::Stroke::new(
                    1.0,
                    egui::Color32::from_rgba_unmultiplied(56, 249, 215, 36),
                ))
                .inner_margin(egui::Margin::same(13));

            // Left Panel: GENOMORPH Universe Laboratory & Biosphere Telemetry
            egui::SidePanel::left("lab_controls")
                .default_width(276.0)
                .resizable(true)
                .frame(panel_frame)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("◉ ABIOGENESIS · GENOMORPH")
                                .size(14.5)
                                .strong()
                                .color(egui::Color32::from_rgb(56, 249, 215)),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "{:.0} FPS · {} · Tick {}",
                                self.current_fps,
                                if stats.gpu_active { "RTX GPU + Rayon" } else { "CPU-Rayon" },
                                stats.tick
                            ))
                            .small()
                            .color(egui::Color32::from_rgb(130, 165, 195)),
                        );
                        ui.add_space(6.0);

                        ui.horizontal(|ui| {
                            if ui
                                .button(if self.paused { "▶ Play" } else { "⏸ Pause" })
                                .clicked()
                            {
                                self.paused = !self.paused;
                            }
                            if ui.button("⏭ Step").clicked() {
                                self.paused = true;
                                self.single_step = true;
                            }
                            if ui.button("🎬 Cinema").clicked() {
                                self.cinematic_mode = true;
                            }
                        });

                        ui.add_space(4.0);
                        ui.add(
                            egui::Slider::new(&mut self.steps_per_frame, 1..=4).text("Sim Speed"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.cam_zoom, 0.7..=16.0)
                                .logarithmic(true)
                                .text("Microscope"),
                        );
                        if ui.button("⌖ Reset View (0)").clicked() {
                            self.cam_x = 0.0;
                            self.cam_y = 0.0;
                            self.cam_zoom = 1.80;
                            self.track_selected = false;
                        }

                        ui.add_space(6.0);
                        ui.separator();
                        ui.label(
                            egui::RichText::new("EXPERIMENTAL UNIVERSE (12)")
                                .small()
                                .strong()
                                .color(egui::Color32::from_rgb(148, 163, 184)),
                        );
                        egui::ComboBox::from_id_salt("universe_preset")
                            .width(235.0)
                            .selected_text(self.selected_preset.name())
                            .show_ui(ui, |ui| {
                                for preset in UniversePreset::ALL {
                                    ui.selectable_value(
                                        &mut self.selected_preset,
                                        preset,
                                        preset.name(),
                                    );
                                }
                            });
                        ui.label(
                            egui::RichText::new(self.selected_preset.description())
                                .small()
                                .color(egui::Color32::from_rgb(140, 160, 185)),
                        );
                        if ui.button("🌌 Seed Selected Universe").clicked() {
                            action_new_universe = Some(self.selected_preset);
                        }
                        ui.checkbox(
                            &mut self.world.universe.allow_abiogenesis,
                            "Allow Abiogenesis (off = perm. extinction)",
                        );

                        ui.add_space(6.0);
                        ui.separator();
                        ui.label(
                            egui::RichText::new("EVOLUTIONARY TELEMETRY")
                                .small()
                                .strong()
                                .color(egui::Color32::from_rgb(148, 163, 184)),
                        );
                        ui.horizontal(|ui| {
                            metric_pill(
                                ui,
                                "ORGS",
                                &format!("{}", stats.organism_count),
                                egui::Color32::from_rgb(56, 249, 215),
                            );
                            metric_pill(
                                ui,
                                "GENES",
                                &format!("{:.1}", stats.mean_genes),
                                egui::Color32::from_rgb(167, 139, 250),
                            );
                            metric_pill(
                                ui,
                                "GEN",
                                &format!("{}", stats.max_generation),
                                egui::Color32::from_rgb(244, 114, 182),
                            );
                        });
                        ui.add_space(4.0);
                        ui.label(
                            egui::RichText::new(format!(
                                "Lineages: {} (Extinct: {}) · Synapses: {:.1}",
                                stats.active_lineages, stats.extinction_events, stats.mean_synapses
                            ))
                            .small()
                            .color(egui::Color32::from_rgb(165, 185, 205)),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "Births {} · Deaths {} · Predations {}",
                                stats.total_births, stats.total_deaths, stats.total_predations
                            ))
                            .small()
                            .color(egui::Color32::from_rgb(165, 185, 205)),
                        );

                        ui.add_space(4.0);
                        draw_sparkline(
                            ui,
                            "Population",
                            &self.pop_history,
                            egui::Color32::from_rgb(56, 249, 215),
                        );
                        draw_sparkline(
                            ui,
                            "Mean Ambient Nutrient",
                            &self.nutr_history,
                            egui::Color32::from_rgb(52, 211, 153),
                        );

                        ui.add_space(6.0);
                        ui.separator();
                        ui.horizontal(|ui| {
                            if ui.button("💾 Save v3").clicked() {
                                action_save = true;
                            }
                            if ui.button("📂 Load v3").clicked() {
                                action_load = true;
                            }
                            if ui.button("📸 PNG").clicked() {
                                action_screenshot = true;
                            }
                        });
                        ui.label(
                            egui::RichText::new(&self.status_toast)
                                .small()
                                .italics()
                                .color(egui::Color32::from_rgb(125, 155, 185)),
                        );
                    });
                });

            // Right Panel: Anatomical Microscope, Live NeuralGraph, Lineage Diff & Discovery Archive
            egui::SidePanel::right("inspector_panel")
                .default_width(300.0)
                .resizable(true)
                .frame(panel_frame)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.label(
                            egui::RichText::new("🔬 ANATOMICAL & NEURAL MICROSCOPE")
                                .size(13.5)
                                .strong()
                                .color(egui::Color32::from_rgb(56, 249, 215)),
                        );
                        ui.add_space(4.0);

                        if let Some(id) = self.selected_org_id {
                            if let Some(org) = self.world.organisms.iter().find(|o| o.id == id) {
                                ui.horizontal(|ui| {
                                    ui.strong(format!(
                                        "Org #{} · {} ({})",
                                        org.id,
                                        org.life_stage.name(),
                                        org.tribe.name()
                                    ));
                                    ui.checkbox(&mut self.track_selected, "Track");
                                });
                                if ui.button("🔎 Focus Microscope (6×)").clicked() {
                                    self.cam_x = org.x;
                                    self.cam_y = org.y;
                                    self.cam_zoom = 6.0;
                                    self.track_selected = true;
                                }
                                ui.label(
                                    egui::RichText::new(format!(
                                        "Lineage #{} · Gen {} · Age {} · Offspring {}",
                                        org.lineage_id, org.generation, org.age, org.offspring_count
                                    ))
                                    .small(),
                                );
                                ui.label(
                                    egui::RichText::new(format!(
                                        "Energy {:.1} · Maturity {:.0}% · Diet {:.0}% Carnivore",
                                        org.energy,
                                        org.maturity * 100.0,
                                        org.genomorph.diet_carnivory * 100.0
                                    ))
                                    .small()
                                    .color(egui::Color32::from_rgb(251, 191, 36)),
                                );

                                ui.add_space(4.0);
                                ui.separator();
                                ui.label(
                                    egui::RichText::new("FUNCTIONAL ANATOMY & DIGESTION")
                                        .small()
                                        .strong()
                                        .color(egui::Color32::from_rgb(148, 163, 184)),
                                );
                                density_bar(ui, "Propulsion", org.phenotype.propulsion_power / 2.2, egui::Color32::from_rgb(56, 249, 215));
                                density_bar(ui, "Bite Power", org.phenotype.bite_power / 1.2, egui::Color32::from_rgb(244, 114, 182));
                                density_bar(ui, "Armor", org.phenotype.armor_rating, egui::Color32::from_rgb(167, 139, 250));
                                density_bar(ui, "Filter", org.phenotype.filter_efficiency / 2.0, egui::Color32::from_rgb(52, 211, 153));
                                density_bar(
                                    ui,
                                    "Stomach",
                                    (org.stomach_plant + org.stomach_meat)
                                        / org.phenotype.stomach_capacity.max(1.0),
                                    egui::Color32::from_rgb(251, 191, 36),
                                );

                                ui.add_space(4.0);
                                ui.separator();
                                ui.label(
                                    egui::RichText::new(format!(
                                        "LIVE NEURALGRAPH ({} hidden, {} synapses)",
                                        org.brain.hidden_count(),
                                        org.brain.active_connection_count()
                                    ))
                                    .small()
                                    .strong()
                                    .color(egui::Color32::from_rgb(167, 139, 250)),
                                );
                                draw_neural_graph(ui, &org.brain);

                                ui.add_space(4.0);
                                ui.separator();
                                ui.label(
                                    egui::RichText::new("GENERATIONAL ANATOMY DIFF")
                                        .small()
                                        .strong()
                                        .color(egui::Color32::from_rgb(244, 114, 182)),
                                );
                                ui.label(
                                    egui::RichText::new(format!("Last: {}", org.genomorph.last_mutation))
                                        .small()
                                        .italics(),
                                );
                                let anc_rec = org
                                    .parent_id
                                    .and_then(|pid| self.world.phylogeny.history.get(&pid))
                                    .or_else(|| self.world.phylogeny.history.get(&org.lineage_id));
                                if let Some(anc) = anc_rec {
                                    let diff = diff_genomes(&anc.genomorph, &org.genomorph);
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "vs Ancestor #{} (Gen {}): {} retained · {} mod · +{} added · -{} lost",
                                            anc.id,
                                            anc.generation,
                                            diff.retained.len(),
                                            diff.modified.len(),
                                            diff.added.len(),
                                            diff.removed.len()
                                        ))
                                        .small()
                                        .color(egui::Color32::from_rgb(190, 225, 245)),
                                    );
                                    for (gid, kind) in diff.added.iter().take(3) {
                                        ui.label(
                                            egui::RichText::new(format!("  + #{} {}", gid, kind.name()))
                                                .small()
                                                .color(egui::Color32::from_rgb(52, 211, 153)),
                                        );
                                    }
                                    for (gid, kind, note) in diff.modified.iter().take(3) {
                                        ui.label(
                                            egui::RichText::new(format!("  ~ #{} {} ({})", gid, kind.name(), note))
                                                .small()
                                                .color(egui::Color32::from_rgb(251, 191, 36)),
                                        );
                                    }
                                } else {
                                    ui.label(egui::RichText::new("Founding generation (Gen 0)").small());
                                }
                            } else {
                                ui.label("Selected organism has perished.");
                                self.selected_org_id = None;
                                self.track_selected = false;
                            }
                        } else {
                            ui.label(
                                egui::RichText::new(
                                    "Click any organism to inspect its procedural anatomy, live NeuralGraph activations, and parent-offspring homology diff.",
                                )
                                .small()
                                .color(egui::Color32::from_rgb(145, 165, 190)),
                            );
                        }

                        ui.add_space(8.0);
                        ui.separator();
                        ui.label(
                            egui::RichText::new(format!(
                                "🧬 MAP-ELITES DISCOVERY ARCHIVE ({}/36)",
                                stats.archive_discoveries
                            ))
                            .small()
                            .strong()
                            .color(egui::Color32::from_rgb(148, 163, 184)),
                        );
                        let mut top: Vec<_> = self
                            .world
                            .phylogeny
                            .map_elites
                            .iter()
                            .flatten()
                            .collect();
                        top.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
                        for entry in top.into_iter().take(6) {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "◆ {} (G{})",
                                        entry.label, entry.generation
                                    ))
                                    .small()
                                    .color(egui::Color32::from_rgb(190, 225, 245)),
                                );
                                if ui.small_button("Spawn").clicked() {
                                    action_resurrect = Some(entry.clone());
                                }
                            });
                        }
                    });
                });
        });

        gpu.egui_state
            .handle_platform_output(&win, full_output.platform_output);
        let tris = gpu
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, image_delta) in &full_output.textures_delta.set {
            gpu.egui_renderer
                .update_texture(&gpu.device, &gpu.queue, *id, image_delta);
        }
        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [gpu.config.width, gpu.config.height],
            pixels_per_point: full_output.pixels_per_point,
        };

        let frame = match gpu.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                crash_log::log_event("surface Lost/Outdated -> reconfiguring");
                gpu.surface.configure(&gpu.device, &gpu.config);
                return;
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                crash_log::log_event("ERROR: wgpu SurfaceError::OutOfMemory");
                return;
            }
            Err(_) => return,
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Observatory Frame Encoder"),
            });

        gpu.egui_renderer.update_buffers(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &tris,
            &screen_desc,
        );

        {
            let mut rpass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
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
                })
                .forget_lifetime();
            rpass.set_pipeline(&gpu.pipeline);
            rpass.set_bind_group(0, &gpu.bind_group, &[]);
            rpass.draw(0..3, 0..1);
            gpu.egui_renderer.render(&mut rpass, &tris, &screen_desc);
        }

        gpu.queue.submit(Some(encoder.finish()));
        frame.present();

        for id in &full_output.textures_delta.free {
            gpu.egui_renderer.free_texture(id);
        }
        gpu.device.poll(wgpu::Maintain::Poll);

        if let Some(preset) = action_new_universe {
            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0xAB10_6E4E);
            self.world =
                ConvergenceWorld::from_universe(seed, preset.config(), self.prefer_gpu_compute);
            self.selected_org_id = None;
            self.track_selected = false;
            self.pop_history.clear();
            self.nutr_history.clear();
            self.status_toast = format!("Seeded {}", preset.name());
        }
        if let Some(entry) = action_resurrect {
            if let Some(new_id) = self.world.spawn_from_discovery(&entry) {
                self.selected_org_id = Some(new_id);
                self.status_toast = format!("[USER INTERVENTION] Spawned #{} ({})", new_id, entry.label);
            }
        }
        if action_save {
            match save_world_atomic(&self.world, "autosave") {
                Ok(p) => self.status_toast = format!("Saved v3 {}", p.display()),
                Err(e) => self.status_toast = format!("Save error: {e}"),
            }
        }
        if action_load {
            match load_world(
                Path::new("saves/worlds/autosave.abio.json"),
                self.prefer_gpu_compute,
            ) {
                Ok(w) => {
                    self.world = w;
                    self.selected_org_id = None;
                    self.status_toast = "Loaded v3 autosave world".to_string();
                }
                Err(e) => self.status_toast = format!("Load error: {e}"),
            }
        }
        if action_screenshot {
            let _ = std::fs::create_dir_all("assets/screenshots");
            match render_png(
                &self.world,
                1440,
                810,
                self.cam_x,
                self.cam_y,
                self.cam_zoom,
                Path::new("assets/screenshots/observatory_capture.png"),
            ) {
                Ok(()) => {
                    self.status_toast =
                        "Saved assets/screenshots/observatory_capture.png".to_string()
                }
                Err(e) => self.status_toast = format!("Capture error: {e}"),
            }
        }

        self.frame_count += 1;
        if self.last_fps_update.elapsed().as_secs_f32() >= 0.5 {
            self.current_fps =
                self.frame_count as f32 / self.last_fps_update.elapsed().as_secs_f32();
            self.frame_count = 0;
            self.last_fps_update = Instant::now();
        }
    }
}

fn draw_neural_graph(ui: &mut egui::Ui, brain: &NeuralGraph) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(268.0, 108.0), egui::Sense::hover());
    ui.painter().rect_filled(
        rect,
        6.0,
        egui::Color32::from_rgba_unmultiplied(8, 12, 24, 230),
    );

    let mut pos: HashMap<u32, egui::Pos2> = HashMap::new();
    let inputs: Vec<_> = brain.nodes.iter().filter(|n| n.kind == NodeKind::Input).collect();
    let hiddens: Vec<_> = brain.nodes.iter().filter(|n| n.kind == NodeKind::Hidden).collect();
    let outputs: Vec<_> = brain.nodes.iter().filter(|n| n.kind == NodeKind::Output).collect();

    for (i, n) in inputs.iter().enumerate() {
        let y = rect.top() + 8.0 + (i as f32 / inputs.len().max(2) as f32) * (rect.height() - 16.0);
        pos.insert(n.id, egui::pos2(rect.left() + 18.0, y));
    }
    for (i, n) in hiddens.iter().enumerate() {
        let y = rect.top()
            + 14.0
            + ((i as f32 + 0.5) / hiddens.len().max(1) as f32) * (rect.height() - 28.0);
        pos.insert(n.id, egui::pos2(rect.center().x, y));
    }
    for (i, n) in outputs.iter().enumerate() {
        let y = rect.top()
            + 16.0
            + ((i as f32 + 0.5) / outputs.len().max(1) as f32) * (rect.height() - 32.0);
        pos.insert(n.id, egui::pos2(rect.right() - 18.0, y));
    }

    for c in &brain.connections {
        if !c.enabled {
            continue;
        }
        if let (Some(&p0), Some(&p1)) = (pos.get(&c.from), pos.get(&c.to)) {
            let alpha = ((c.weight.abs() * 140.0) as u8).clamp(45, 220);
            let col = if c.weight >= 0.0 {
                egui::Color32::from_rgba_unmultiplied(56, 249, 215, alpha)
            } else {
                egui::Color32::from_rgba_unmultiplied(244, 114, 182, alpha)
            };
            ui.painter()
                .line_segment([p0, p1], egui::Stroke::new(1.1, col));
        }
    }

    for n in &brain.nodes {
        if let Some(&p) = pos.get(&n.id) {
            let act = n.value.clamp(-1.0, 1.0);
            let col = if act >= 0.0 {
                egui::Color32::from_rgb(56, (140.0 + act * 115.0) as u8, 220)
            } else {
                egui::Color32::from_rgb((150.0 + (-act) * 100.0) as u8, 80, 175)
            };
            ui.painter().circle_filled(p, 3.6, col);
        }
    }
}

fn configure_laboratory_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.window_fill = egui::Color32::from_rgba_unmultiplied(8, 12, 22, 235);
    visuals.panel_fill = egui::Color32::from_rgba_unmultiplied(7, 11, 19, 228);
    visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(13, 20, 34);
    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(17, 26, 44);
    visuals.widgets.inactive.corner_radius = egui::CornerRadius::same(6);
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(24, 42, 68);
    visuals.widgets.hovered.fg_stroke =
        egui::Stroke::new(1.2, egui::Color32::from_rgb(56, 249, 215));
    visuals.widgets.hovered.corner_radius = egui::CornerRadius::same(6);
    visuals.widgets.active.bg_fill = egui::Color32::from_rgb(14, 116, 144);
    visuals.selection.bg_fill = egui::Color32::from_rgba_unmultiplied(14, 165, 233, 90);
    visuals.selection.stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(56, 249, 215));
    ctx.set_visuals(visuals);
}

fn metric_pill(ui: &mut egui::Ui, label: &str, value: &str, color: egui::Color32) {
    egui::Frame::NONE
        .fill(egui::Color32::from_rgb(12, 19, 32))
        .stroke(egui::Stroke::new(1.0, color.linear_multiply(0.35)))
        .corner_radius(egui::CornerRadius::same(6))
        .inner_margin(egui::Margin::symmetric(8, 5))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.label(
                    egui::RichText::new(label)
                        .size(9.5)
                        .color(egui::Color32::from_rgb(140, 160, 185)),
                );
                ui.label(egui::RichText::new(value).size(14.0).strong().color(color));
            });
        });
}

fn density_bar(ui: &mut egui::Ui, label: &str, frac: f32, color: egui::Color32) {
    let f = frac.clamp(0.0, 1.0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(format!("{label:<10}")).small().monospace());
        let (rect, _) = ui.allocate_exact_size(egui::vec2(155.0, 7.5), egui::Sense::hover());
        ui.painter()
            .rect_filled(rect, 4.0, egui::Color32::from_rgb(15, 23, 38));
        let mut fill_rect = rect;
        fill_rect.set_width(rect.width() * f);
        ui.painter().rect_filled(fill_rect, 4.0, color);
    });
}

fn draw_sparkline(ui: &mut egui::Ui, title: &str, data: &VecDeque<f32>, color: egui::Color32) {
    ui.label(
        egui::RichText::new(title)
            .size(10.5)
            .color(egui::Color32::from_rgb(140, 165, 190)),
    );
    let (rect, _) = ui.allocate_exact_size(egui::vec2(244.0, 34.0), egui::Sense::hover());
    ui.painter().rect_filled(
        rect,
        6.0,
        egui::Color32::from_rgba_unmultiplied(10, 16, 28, 220),
    );
    if data.len() < 2 {
        return;
    }
    let min_v = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max_v = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = (max_v - min_v).max(0.001);
    let n = (data.len() - 1) as f32;

    let points: Vec<egui::Pos2> = data
        .iter()
        .enumerate()
        .map(|(i, &val)| {
            let x = rect.left() + 4.0 + (i as f32 / n) * (rect.width() - 8.0);
            let y = rect.bottom() - 4.0 - ((val - min_v) / span) * (rect.height() - 8.0);
            egui::pos2(x, y)
        })
        .collect();
    ui.painter()
        .add(egui::Shape::line(points, egui::Stroke::new(1.6, color)));
}

impl ApplicationHandler for ObservatoryApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("SYMBIOTE: ABIOGENESIS v0.4.0 — PROJECT GENOMORPH")
                .with_inner_size(winit::dpi::LogicalSize::new(1440.0, 900.0));
            let win = Arc::new(event_loop.create_window(attrs).expect("create window"));
            self.init_gpu(win.clone());
            self.window = Some(win);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let mut egui_consumed = false;
        if let (Some(win), Some(gpu)) = (&self.window, &mut self.gpu) {
            let resp = gpu.egui_state.on_window_event(win, &event);
            egui_consumed = resp.consumed;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new_size) => {
                if new_size.width == 0 || new_size.height == 0 {
                    self.is_minimized = true;
                    crash_log::log_event("window minimized (0x0)");
                } else {
                    self.is_minimized = false;
                    if let Some(gpu) = &mut self.gpu {
                        let max_dim = gpu.device.limits().max_texture_dimension_2d;
                        gpu.config.width = new_size.width.min(max_dim);
                        gpu.config.height = new_size.height.min(max_dim);
                        gpu.surface.configure(&gpu.device, &gpu.config);
                        crash_log::log_event(&format!(
                            "window resized -> {}x{}",
                            gpu.config.width, gpu.config.height
                        ));
                    }
                }
            }
            WindowEvent::ModifiersChanged(mods) => {
                self.alt_held = mods.state().alt_key();
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = (position.x as f32, position.y as f32);
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } if !egui_consumed => {
                self.select_organism_at_cursor();
            }
            WindowEvent::MouseWheel { delta, .. } if !egui_consumed => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => (p.y / 40.0) as f32,
                };
                let factor = if scroll > 0.0 { 1.15 } else { 1.0 / 1.15 };
                self.cam_zoom = (self.cam_zoom * factor).clamp(0.7, 18.0);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } if !egui_consumed => {
                let pan = 0.08 / self.cam_zoom;
                match code {
                    KeyCode::Escape | KeyCode::KeyQ => event_loop.exit(),
                    KeyCode::Space => self.paused = !self.paused,
                    KeyCode::KeyH => self.cinematic_mode = !self.cinematic_mode,
                    KeyCode::F11 => self.toggle_borderless_fullscreen(),
                    KeyCode::Enter if self.alt_held => self.toggle_borderless_fullscreen(),
                    KeyCode::ArrowLeft | KeyCode::KeyA => {
                        self.track_selected = false;
                        self.cam_x -= pan;
                    }
                    KeyCode::ArrowRight | KeyCode::KeyD => {
                        self.track_selected = false;
                        self.cam_x += pan;
                    }
                    KeyCode::ArrowUp | KeyCode::KeyW => {
                        self.track_selected = false;
                        self.cam_y -= pan;
                    }
                    KeyCode::ArrowDown | KeyCode::KeyS => {
                        self.track_selected = false;
                        self.cam_y += pan;
                    }
                    KeyCode::Equal => self.cam_zoom = (self.cam_zoom * 1.2).clamp(0.7, 18.0),
                    KeyCode::Minus => self.cam_zoom = (self.cam_zoom / 1.2).clamp(0.7, 18.0),
                    KeyCode::Digit0 => {
                        self.cam_x = 0.0;
                        self.cam_y = 0.0;
                        self.cam_zoom = 1.80;
                        self.track_selected = false;
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
    selected_idx: i32,
    pad0: i32,
    pad1: i32,
    pad2: i32,
};

struct OrgMeta {
    x: f32,
    y: f32,
    radius: f32,
    hue: f32,
    accent_shift: f32,
    energy_norm: f32,
    pulse: f32,
    mass: f32,
    heading: f32,
    organ_start: u32,
    organ_count: u32,
    pad: u32,
};

struct Organ {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    width: f32,
    hue_shift: f32,
    kind_code: u32,
    pad: u32,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> eco: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> orgs: array<OrgMeta>;
@group(0) @binding(3) var<storage, read> bodies: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> organs: array<Organ>;

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

fn toroidal_delta(d: f32) -> f32 {
    return d - 2.0 * floor((d + 1.0) * 0.5);
}

fn sd_capsule_t(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let pa = p - a;
    let ba = b - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 0.00001), 0.0, 1.0);
    return vec2<f32>(length(pa - ba * h), h);
}

fn sample_eco(wx: f32, wy: f32) -> vec4<f32> {
    let u = fract((wx + 1.0) * 0.5) * f32(cam.eco_w);
    let v = fract((wy + 1.0) * 0.5) * f32(cam.eco_h);
    let x0 = u32(floor(u)) % cam.eco_w;
    let y0 = u32(floor(v)) % cam.eco_h;
    let x1 = (x0 + 1u) % cam.eco_w;
    let y1 = (y0 + 1u) % cam.eco_h;
    let fx = smoothstep(0.0, 1.0, fract(u));
    let fy = smoothstep(0.0, 1.0, fract(v));

    let c00 = eco[y0 * cam.eco_w + x0];
    let c10 = eco[y0 * cam.eco_w + x1];
    let c01 = eco[y1 * cam.eco_w + x0];
    let c11 = eco[y1 * cam.eco_w + x1];
    return mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);
}

fn sample_body(org_idx: u32, lu: f32, lv: f32) -> vec3<f32> {
    let res = 40u;
    let fx_pos = clamp(lu, 0.0, 0.999) * f32(res - 1u);
    let fy_pos = clamp(lv, 0.0, 0.999) * f32(res - 1u);
    let x0 = u32(floor(fx_pos));
    let y0 = u32(floor(fy_pos));
    let x1 = min(x0 + 1u, res - 1u);
    let y1 = min(y0 + 1u, res - 1u);
    let fx = smoothstep(0.0, 1.0, fract(fx_pos));
    let fy = smoothstep(0.0, 1.0, fract(fy_pos));

    let base = org_idx * res * res;
    let b00 = bodies[base + y0 * res + x0].xyz;
    let b10 = bodies[base + y0 * res + x1].xyz;
    let b01 = bodies[base + y1 * res + x0].xyz;
    let b11 = bodies[base + y1 * res + x1].xyz;
    return mix(mix(b00, b10, fx), mix(b01, b11, fx), fy);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let ndc = (in.uv - vec2<f32>(0.5, 0.5)) * 2.0;
    let wx = cam.center_x + (ndc.x * cam.aspect) / cam.zoom;
    let wy = cam.center_y + ndc.y / cam.zoom;

    let vig = clamp(1.0 - 0.24 * (ndc.x * ndc.x * 0.45 + ndc.y * ndc.y), 0.55, 1.0);
    var col = vec3<f32>(0.003, 0.005, 0.011) * vig;

    let e = sample_eco(wx, wy);
    let nutr_wisp = pow(max(e.x - 0.36, 0.0), 2.2);
    let pher_wisp = pow(e.y, 1.5);
    let tox_wisp = pow(e.z, 1.6);
    let root_fil = pow(e.w, 2.4);

    col += vec3<f32>(0.008, 0.055, 0.085) * nutr_wisp;
    col += vec3<f32>(0.14, 0.05, 0.26) * pher_wisp;
    col += vec3<f32>(0.42, 0.05, 0.09) * tox_wisp;
    col += vec3<f32>(0.03, 0.14, 0.28) * root_fil;

    for (var i: u32 = 0u; i < cam.org_count; i = i + 1u) {
        let o = orgs[i];
        let bound_r = o.radius * 1.75;
        let dx = toroidal_delta(wx - o.x);
        let dy = toroidal_delta(wy - o.y);
        if (abs(dx) > bound_r || abs(dy) > bound_r) {
            continue;
        }

        let breath = 0.90 + 0.10 * sin(o.pulse + cam.time * 1.8);
        let core_rgb = hsv_to_rgb(o.hue, 0.90, 0.72 * breath * o.energy_norm);
        let mem_rgb = hsv_to_rgb(fract(o.hue + 0.08), 0.70, 1.15);
        let org_rgb = hsv_to_rgb(fract(o.hue + o.accent_shift), 0.95, 1.45);

        let dist_norm = length(vec2<f32>(dx, dy)) / bound_r;
        if (dist_norm < 1.0) {
            let halo = pow(1.0 - dist_norm, 3.0) * 0.075 * o.mass * 3.0 * o.energy_norm;
            col += core_rgb * halo;
        }

        if (abs(dx) <= o.radius && abs(dy) <= o.radius) {
            let lu = (dx / o.radius) * 0.5 + 0.5;
            let lv = (dy / o.radius) * 0.5 + 0.5;
            let cell = sample_body(i, lu, lv);
            let cyto = cell.x;
            let mem = cell.y;
            let organ = cell.z;

            if (cyto > 0.015 || mem > 0.015 || organ > 0.015) {
                let cyto_soft = smoothstep(0.03, 0.72, cyto);
                let inner_glow = pow(cyto, 1.6);
                let alpha = clamp(cyto_soft * 0.72, 0.0, 0.82);
                let rim = pow(mem, 1.45) * 0.65;
                let nucleus = pow(organ, 1.3) * 1.05;

                col = mix(col, core_rgb * (0.20 + 0.85 * inner_glow), alpha);
                col += mem_rgb * rim;
                col += org_rgb * nucleus;
            }
        }

        // Articulated procedural anatomy organs
        let ch = cos(o.heading);
        let sh = sin(o.heading);
        let lp = vec2<f32>(dx * ch + dy * sh, -dx * sh + dy * ch) / o.radius;

        for (var k: u32 = 0u; k < o.organ_count; k = k + 1u) {
            let og = organs[o.organ_start + k];
            let dt = sd_capsule_t(lp, vec2<f32>(og.x0, og.y0), vec2<f32>(og.x1, og.y1));
            let w_taper = og.width * max(1.0 - dt.y * 0.42, 0.25);
            if (dt.x > w_taper * 1.85) {
                continue;
            }
            let fill = 1.0 - smoothstep(w_taper * 0.35, w_taper, dt.x);
            let rim = 1.0 - smoothstep(0.0, w_taper * 0.55, abs(dt.x - w_taper * 0.75));

            var o_sat = 0.85;
            var o_val = 0.95 * o.energy_norm;
            var o_alpha = 0.68;
            if (og.kind_code == 4u) { o_sat = 0.95; o_val = 1.35 * o.energy_norm; o_alpha = 0.82; }
            else if (og.kind_code == 6u) { o_sat = 0.75; o_val = 1.45 * o.energy_norm; o_alpha = 0.72; }
            else if (og.kind_code == 7u) { o_sat = 0.45; o_val = 1.05 * o.energy_norm; o_alpha = 0.78; }
            else if (og.kind_code == 5u) { o_sat = 0.85; o_val = 0.85 * o.energy_norm; o_alpha = 0.48; }
            let o_col = hsv_to_rgb(fract(o.hue + og.hue_shift), o_sat, o_val);
            let a = fill * o_alpha;
            col = col * (1.0 - a * 0.75) + o_col * a * (0.45 + 0.55 * (1.0 - dt.y)) + o_col * rim * 0.45;
        }

        if (i32(i) == cam.selected_idx) {
            let dist = length(vec2<f32>(dx, dy)) / (o.radius * 1.35);
            if (abs(dist - 0.96) < 0.025) {
                col += vec3<f32>(0.22, 0.97, 0.84) * (0.65 + 0.25 * sin(cam.time * 4.0));
            }
        }
    }

    let x = max(col, vec3<f32>(0.0));
    let mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    return vec4<f32>(pow(clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(0.92)), 1.0);
}
"#;
