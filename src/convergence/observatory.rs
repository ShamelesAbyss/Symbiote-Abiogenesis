//! Symbiote: Abiogenesis — Professional Graphical Research Laboratory.
//!
//! Combines a 3-channel fluorescence-microscopy WGSL renderer with an interactive
//! `egui` scientific interface: Universe presets, live telemetry, click-to-inspect
//! microscopic organism tracking, MAP-Elites discovery archive, atomic world
//! persistence, and distraction-free Cinematic Mode (`H`).

use crate::convergence::eco_field::{ECO_HEIGHT, ECO_WIDTH};
use crate::convergence::engine::{ConvergenceWorld, MAX_ORGANISMS_BUFFER};
use crate::convergence::lenia_body::{BODY_CHANNELS, BODY_RES};
use crate::convergence::persistence::{load_world, save_world_atomic};
use crate::convergence::snapshot::render_png;
use crate::convergence::universes::UniversePreset;
use anyhow::Result;
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
    _pad: f32,
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
            cam_zoom: 1.0,
            paused: false,
            single_step: false,
            steps_per_frame: 1,
            cinematic_mode: false,
            selected_org_id: None,
            track_selected: false,
            selected_preset: UniversePreset::PrimordialOcean,
            status_toast: "Ready — Click any organism to inspect".to_string(),
            cursor_pos: (0.0, 0.0),
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
                label: Some("Symbiote Abiogenesis Observatory Device"),
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
            size: (MAX_ORGANISMS_BUFFER * std::mem::size_of::<GpuOrganismMeta>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // 3 channels packed into vec4<f32> (cyto, mem, organ, 0.0) for 16-byte WGSL alignment
        let org_bodies_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Organism 3ch Lenia Bodies Buffer"),
            size: (MAX_ORGANISMS_BUFFER * BODY_RES * BODY_RES * 4 * std::mem::size_of::<f32>())
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
            label: Some("Observatory 3ch Bioluminescent Shader"),
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
            egui_ctx,
            egui_state,
            egui_renderer,
        });
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
            let d = ((org.x - wx).powi(2) + (org.y - wy).powi(2)).sqrt();
            let r = 0.065 + org.body.metrics.radius_norm * 0.055;
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

        // Smooth camera tracking when inspecting an organism
        if self.track_selected {
            if let Some(id) = self.selected_org_id {
                if let Some(org) = self.world.organisms.iter().find(|o| o.id == id) {
                    self.cam_x += (org.x - self.cam_x) * 0.18;
                    self.cam_y += (org.y - self.cam_y) * 0.18;
                }
            }
        }

        let Some(win) = self.window.clone() else {
            return;
        };
        let Some(gpu) = &mut self.gpu else { return };

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

        let mut metas = Vec::with_capacity(org_count);
        let mut bodies_rgba = Vec::with_capacity(org_count * BODY_RES * BODY_RES * 4);
        for org in self.world.organisms.iter().take(org_count) {
            metas.push(GpuOrganismMeta {
                x: org.x,
                y: org.y,
                radius: 0.055 + org.body.metrics.radius_norm * 0.055,
                hue: org.morph_genome.biolum_hue,
                accent_shift: org.morph_genome.accent_shift,
                energy_norm: (org.energy / 100.0).clamp(0.15, 1.35),
                pulse: org.body.pulse_phase,
                _pad: 0.0,
            });
            for chunk in org.body.cells.chunks_exact(BODY_CHANNELS) {
                bodies_rgba.push(chunk[0]);
                bodies_rgba.push(chunk[1]);
                bodies_rgba.push(chunk[2]);
                bodies_rgba.push(0.0);
            }
        }
        if !metas.is_empty() {
            gpu.queue
                .write_buffer(&gpu.org_meta_buf, 0, bytemuck::cast_slice(&metas));
            gpu.queue
                .write_buffer(&gpu.org_bodies_buf, 0, bytemuck::cast_slice(&bodies_rgba));
        }

        // Build egui Research Laboratory UI
        let raw_input = gpu.egui_state.take_egui_input(&win);
        let stats = self.world.stats();
        let mut action_new_universe: Option<UniversePreset> = None;
        let mut action_save = false;
        let mut action_load = false;
        let mut action_screenshot = false;

        let full_output = gpu.egui_ctx.run(raw_input, |ctx| {
            if self.cinematic_mode {
                return;
            }

            // Left Panel: Universe Laboratory & Simulation Controls
            egui::SidePanel::left("lab_controls")
                .default_width(250.0)
                .resizable(true)
                .show(ctx, |ui| {
                    ui.heading("◉ ABIOGENESIS LAB");
                    ui.label(
                        egui::RichText::new(format!(
                            "{:.0} FPS · {} · Tick {}",
                            self.current_fps,
                            if stats.gpu_active { "GPU+Rayon" } else { "CPU-Rayon" },
                            stats.tick
                        ))
                        .small()
                        .color(egui::Color32::from_rgb(120, 210, 220)),
                    );
                    ui.separator();

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
                        if ui.button("🎬 Hide UI").clicked() {
                            self.cinematic_mode = true;
                        }
                    });

                    ui.add(
                        egui::Slider::new(&mut self.steps_per_frame, 1..=4).text("Steps/Frame"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.cam_zoom, 0.6..=16.0)
                            .logarithmic(true)
                            .text("Zoom"),
                    );
                    if ui.button("Reset Camera (0)").clicked() {
                        self.cam_x = 0.0;
                        self.cam_y = 0.0;
                        self.cam_zoom = 1.0;
                        self.track_selected = false;
                    }

                    ui.separator();
                    ui.strong("Experimental Universe");
                    egui::ComboBox::from_id_salt("universe_preset")
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
                            .weak(),
                    );
                    if ui.button("🌌 Generate Universe").clicked() {
                        action_new_universe = Some(self.selected_preset);
                    }

                    ui.checkbox(
                        &mut self.world.universe.allow_abiogenesis,
                        "Allow Abiogenesis (off = permanent extinction)",
                    );

                    ui.separator();
                    ui.strong("Biosphere Telemetry");
                    ui.label(format!("Organisms: {}", stats.organism_count));
                    ui.label(format!(
                        "Active Lineages: {} (Extinct: {})",
                        stats.active_lineages, stats.extinction_events
                    ));
                    ui.label(format!("Max Generation: {}", stats.max_generation));
                    ui.label(format!(
                        "Births: {} · Deaths: {} · Predations: {}",
                        stats.total_births, stats.total_deaths, stats.total_predations
                    ));
                    ui.label(format!("Mean Nutrient: {:.3}", stats.mean_nutrient));
                    ui.label(format!(
                        "MAP-Elites Archive: {}/36 niches",
                        stats.archive_discoveries
                    ));

                    ui.separator();
                    ui.strong("Persistence & Capture");
                    ui.horizontal(|ui| {
                        if ui.button("💾 Save World").clicked() {
                            action_save = true;
                        }
                        if ui.button("📂 Load World").clicked() {
                            action_load = true;
                        }
                    });
                    if ui.button("📸 Capture PNG Screenshot").clicked() {
                        action_screenshot = true;
                    }
                    ui.separator();
                    ui.label(egui::RichText::new(&self.status_toast).small().italics());
                });

            // Right Panel: Microscopic Organism Inspector & Discovery Archive
            egui::SidePanel::right("inspector_panel")
                .default_width(250.0)
                .resizable(true)
                .show(ctx, |ui| {
                    ui.heading("🔬 Microscopic Inspector");
                    if let Some(id) = self.selected_org_id {
                        if let Some(org) = self.world.organisms.iter().find(|o| o.id == id) {
                            ui.horizontal(|ui| {
                                ui.strong(format!("Organism #{} ({})", org.id, org.tribe.name()));
                                ui.checkbox(&mut self.track_selected, "Track");
                            });
                            if ui.button("Zoom Microscopic (6×)").clicked() {
                                self.cam_x = org.x;
                                self.cam_y = org.y;
                                self.cam_zoom = 6.0;
                                self.track_selected = true;
                            }
                            ui.separator();
                            ui.label(format!(
                                "Lineage #{} · Gen {} · Age {}",
                                org.lineage_id, org.generation, org.age
                            ));
                            ui.label(format!(
                                "Energy: {:.1} · Offspring: {}",
                                org.energy, org.offspring_count
                            ));
                            ui.separator();
                            ui.strong("3-Channel Cellular State");
                            ui.label(format!("Cytoplasm Mass: {:.3}", org.body.metrics.mass));
                            ui.label(format!(
                                "Membrane Integrity: {:.3}",
                                org.body.metrics.membrane_mass
                            ));
                            ui.label(format!(
                                "Organelle Density: {:.3}",
                                org.body.metrics.organelle_mass
                            ));
                            ui.label(format!("Elongation: {:.3}", org.body.metrics.elongation));
                            ui.separator();
                            ui.strong("Heritable MorphGenome");
                            ui.label(format!(
                                "μ={:.4}  σ={:.4}  R={}",
                                org.morph_genome.mu,
                                org.morph_genome.sigma,
                                org.morph_genome.kernel_radius
                            ));
                            ui.label(format!(
                                "Lobes={:.2}  Twist={:+.2}  Asym={:.2}",
                                org.morph_genome.angular_lobes,
                                org.morph_genome.spiral_twist,
                                org.morph_genome.asymmetry
                            ));
                            ui.separator();
                            ui.strong("NeuralBrain Motor Out");
                            ui.label(format!(
                                "Thrust=({:+.2}, {:+.2})  Secrete={:+.2}",
                                org.brain.last_out[0],
                                org.brain.last_out[1],
                                org.brain.last_out[2]
                            ));
                        } else {
                            ui.label("Selected organism has perished.");
                            self.selected_org_id = None;
                            self.track_selected = false;
                        }
                    } else {
                        ui.label("Click any organism in the viewport to inspect its continuous cellular body, genome, and neural activity.");
                    }

                    ui.separator();
                    ui.strong("🧬 Discovery Archive (Top)");
                    let mut top: Vec<_> = self
                        .world
                        .phylogeny
                        .map_elites
                        .iter()
                        .flatten()
                        .collect();
                    top.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
                    for entry in top.into_iter().take(6) {
                        ui.label(
                            egui::RichText::new(format!(
                                "• {} (Gen {}, fit {:.0})",
                                entry.label, entry.generation, entry.fitness
                            ))
                            .small(),
                        );
                    }
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

        // Apply deferred UI actions
        if let Some(preset) = action_new_universe {
            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0xAB10_6E4E);
            self.world = ConvergenceWorld::from_universe(seed, preset.config(), self.prefer_gpu_compute);
            self.selected_org_id = None;
            self.track_selected = false;
            self.status_toast = format!("Generated {}", preset.name());
        }
        if action_save {
            match save_world_atomic(&self.world, "autosave") {
                Ok(p) => self.status_toast = format!("Saved {}", p.display()),
                Err(e) => self.status_toast = format!("Save error: {e}"),
            }
        }
        if action_load {
            match load_world(Path::new("saves/worlds/autosave.abio.json"), self.prefer_gpu_compute) {
                Ok(w) => {
                    self.world = w;
                    self.selected_org_id = None;
                    self.status_toast = "Loaded autosave world".to_string();
                }
                Err(e) => self.status_toast = format!("Load error: {e}"),
            }
        }
        if action_screenshot {
            let _ = std::fs::create_dir_all("assets/screenshots");
            match render_png(
                &self.world,
                1280,
                720,
                self.cam_x,
                self.cam_y,
                self.cam_zoom,
                Path::new("assets/screenshots/observatory_capture.png"),
            ) {
                Ok(()) => self.status_toast = "Saved assets/screenshots/observatory_capture.png".to_string(),
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

impl ApplicationHandler for ObservatoryApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let attrs = Window::default_attributes()
                .with_title("SYMBIOTE: ABIOGENESIS — EVOLUTION UNBOUND")
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
                if let Some(gpu) = &mut self.gpu {
                    gpu.config.width = new_size.width.max(1);
                    gpu.config.height = new_size.height.max(1);
                    gpu.surface.configure(&gpu.device, &gpu.config);
                }
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
            } if !egui_consumed => {
                let pan = 0.08 / self.cam_zoom;
                match code {
                    KeyCode::Escape | KeyCode::KeyQ => event_loop.exit(),
                    KeyCode::Space => self.paused = !self.paused,
                    KeyCode::KeyH => self.cinematic_mode = !self.cinematic_mode,
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
                    KeyCode::Equal => self.cam_zoom = (self.cam_zoom * 1.2).clamp(0.6, 18.0),
                    KeyCode::Minus => self.cam_zoom = (self.cam_zoom / 1.2).clamp(0.6, 18.0),
                    KeyCode::Digit0 => {
                        self.cam_x = 0.0;
                        self.cam_y = 0.0;
                        self.cam_zoom = 1.0;
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
    pad: f32,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(0) @binding(1) var<storage, read> eco: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> orgs: array<OrgMeta>;
@group(0) @binding(3) var<storage, read> bodies: array<vec4<f32>>;

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

fn sample_body(org_idx: u32, lu: f32, lv: f32) -> vec3<f32> {
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

    var col = vec3<f32>(0.008, 0.014, 0.026);
    if (abs(wx) <= 1.0 && abs(wy) <= 1.0) {
        let e = sample_eco(wx, wy);
        col += vec3<f32>(0.02, 0.14, 0.17) * e.x;
        col += vec3<f32>(0.14, 0.06, 0.26) * e.y;
        col += vec3<f32>(0.36, 0.05, 0.09) * e.z;
        col += vec3<f32>(0.08, 0.34, 0.72) * pow(e.w, 1.3);
    }

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
        let organ = cell.z;

        if (cyto > 0.01 || mem > 0.01 || organ > 0.01) {
            let breath = 0.88 + 0.12 * sin(o.pulse + cam.time * 2.0);
            let core_rgb = hsv_to_rgb(o.hue, 0.72, 0.90 * breath * o.energy_norm);
            let mem_rgb = hsv_to_rgb(fract(o.hue + 0.12), 0.42, 1.30);
            let org_rgb = hsv_to_rgb(fract(o.hue + o.accent_shift), 0.85, 1.45);

            let cyto_alpha = clamp(cyto * 0.90, 0.0, 0.94);
            col = mix(col, core_rgb * (0.40 + 0.85 * cyto), cyto_alpha);
            col += mem_rgb * mem * 0.72;
            col += org_rgb * organ * 0.95;
        }

        if (i32(i) == cam.selected_idx) {
            let dist = length(vec2<f32>(dx, dy)) / o.radius;
            if (abs(dist - 0.92) < 0.035) {
                col += vec3<f32>(0.35, 0.95, 1.0) * 0.85;
            }
        }
    }

    col = col / (vec3<f32>(1.0) + col * 0.32);
    return vec4<f32>(pow(col, vec3<f32>(0.92)), 1.0);
}
"#;
