# Architecture — Symbiote: Abiogenesis (v0.3.0)

## Execution Boundaries (CPU+Rayon vs. GPU WGSL)

Abiogenesis uses a pragmatic hybrid architecture designed for consumer GPUs (tested on NVIDIA RTX 4070 Laptop GPU, 8 GB VRAM) while guaranteeing full CPU fallback (`--cpu` and `--no-default-features`).

| Subsystem | Runtime Execution | Resolution / Scale | Module |
| :--- | :--- | :--- | :--- |
| **Localized Lenia Bodies** | CPU (Rayon parallel) | `40×40×3` per organism (`19.2 KB` double-buffered) | `src/convergence/lenia_body.rs` |
| **NeuralBrain Controller** | CPU (Rayon parallel) | `6 → 6 (recurrent) → 3` `tanh` network per organism | `src/convergence/lenia_body.rs` |
| **Toroidal EcoField** | **GPU WGSL Compute** (`wgpu` 24) with CPU-Rayon fallback | `128×128×4` (`256 KB` per buffer) | `src/convergence/eco_field.rs`, `gpu_compute.rs` |
| **Contact & Predation** | CPU ($O(N)$ Spatial Hash Grid, cell `0.16`) | Up to `320` simultaneous organisms | `src/convergence/engine.rs` |
| **Phylogeny & MAP-Elites** | CPU | `6×6` behavior-morphology Quality-Diversity archive | `src/convergence/phylogeny.rs` |
| **Atomic Persistence** | CPU (write-to-`.tmp` + rename) | `saves/worlds/*.abio.json` (versioned schema `v2`) | `src/convergence/persistence.rs` |
| **Observatory Rendering** | **GPU Fragment Shader** (`wgpu` 24) + `egui` 0.31 | Native window (`1440×900` default, resizable to 4K) | `src/convergence/observatory.rs` |
| **Headless Snapshots** | CPU (Rayon parallel, shader-parity math) | `1440×810` RGB PNG (`--snapshot`) | `src/convergence/snapshot.rs` |
| **Legacy Symbiote Core** | CPU (`--legacy-tui` or coupled every 2 ticks) | `576–1760` particles + `64×36` cellular substrate | `src/app.rs`, `src/sim.rs`, `src/automata.rs` |

## Engineered Mechanisms vs. Emergent Dynamics

To maintain scientific clarity, Abiogenesis distinguishes explicitly engineered rules from emergent population-level outcomes:

- **Engineered**: 3-channel Lenia update equations, 9-tap Gaussian diffusion smoothing, genome-driven morphogenetic envelopes, `NeuralBrain` topology, coherent predation damage/energy transfer, zero-sum symbiosis, mitotic splitting thresholds, toroidal wrapping, and MAP-Elites binning.
- **Emergent**: Lineage survival vs. extinction under shifting environments (`Calm → Hunger → Bloom`), population-wide drift in heritable `mu` and `asymmetry`, spatial clustering around hydrothermal nutrient plumes, predator-prey turnover, and multi-niche morphological specialization.
