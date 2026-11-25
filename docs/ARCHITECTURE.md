# Architecture — Symbiote: Abiogenesis (v0.5.0 — Project INFINITY)

## Execution Boundaries (CPU+Rayon vs. GPU WGSL)

Abiogenesis uses a pragmatic hybrid architecture designed for consumer GPUs (tested on NVIDIA RTX 4070 Laptop GPU, 8 GB VRAM) while guaranteeing full CPU fallback (`--cpu` and `--no-default-features`).

| Subsystem | Runtime Execution | Resolution / Scale | Module |
| :--- | :--- | :--- | :--- |
| **World Bounds & Camera** | CPU (`SimCamera`) + GPU Uniforms | `16×9`, `20×11.25`, `24×13.5` non-repeating widescreen worlds | `src/convergence/camera.rs` |
| **GENOMORPH Anatomy** | CPU (Rayon parallel) | Up to 14 homology-tracked `AnatomyGene`s / 10 `OrganKind`s | `src/convergence/genomorph.rs` |
| **NEAT `NeuralGraph`** | CPU (Rayon parallel) | `10` grounded inputs → evolvable hidden nodes → `4` outputs | `src/convergence/neural_graph.rs` |
| **Localized Lenia Bodies** | CPU (Rayon parallel) | `40×40×3` per organism (`19.2 KB` double-buffered) | `src/convergence/lenia_body.rs` |
| **Multi-Biome `EcoField`** | **GPU WGSL Compute** (`wgpu` 24) with CPU-Rayon fallback | `256×144×4` (`576 KB` per buffer) | `src/convergence/eco_field.rs`, `gpu_compute.rs` |
| **Contact & Predation** | CPU (`O(N)` Spatial Hash Grid, cell `0.85`) | Up to `320` simultaneous organisms | `src/convergence/engine.rs` |
| **Phylogeny & MAP-Elites** | CPU | Individual ancestry archive + `6×6` Quality-Diversity grid | `src/convergence/phylogeny.rs` |
| **Atomic Persistence** | CPU (write-to-`.tmp` + rename) | `saves/worlds/*.abio.json` (versioned schema `v4`) | `src/convergence/persistence.rs` |
| **Observatory Rendering** | **GPU Fragment Shader** (`wgpu` 24) + `egui` 0.31 | Strict `[0, W] × [0, H]` clipping, CPU frustum culling, Minimap | `src/convergence/observatory.rs` |
| **Headless Snapshots** | CPU (Rayon parallel, shader-parity math) | `1440×810` RGB PNG (`--snapshot`) | `src/convergence/snapshot.rs` |
| **Legacy Symbiote Core** | CPU (`--legacy-tui` or coupled every 2 ticks) | `576–1760` particles + `64×36` cellular substrate | `src/app.rs`, `src/sim.rs`, `src/automata.rs` |

## Topology Separation (Project INFINITY)

- **Simulation Topology**: Organisms and reaction-diffusion chemicals wrap seamlessly across `[0, world_w) × [0, world_h)` via `WorldBounds::wrap_x/y` and shortest signed `delta_x/y`.
- **Rendering & Camera Topology**: `OBSERVATORY_WGSL` and `snapshot.rs` strictly clip any pixel outside `[0, world_w] × [0, world_h]` to deep outer void (with an optional subtle boundary frame). Each organism and selection ring appears strictly once in full-world overview, allowing only clipped seam fragments when crossing a toroidal boundary.
