# HANDOFF.md — Project State & Cross-Agent Continuity (v0.5.0)

## Current Status (v0.5.0 — Project INFINITY)
- **Repository**: `https://github.com/ShamelesAbyss/Symbiote-Abiogenesis` (`main` branch exclusively)
- **Upstream Originals Preserved**: `Symbiote`, `Primordia`, and `Hydra` remain completely untouched.
- **Default Entrypoint**: Double-clicking `symbiote-abiogenesis.exe` launches the native `winit` + `wgpu` 24 + `egui` 0.31 **INFINITY Research Laboratory** (`--legacy-tui` for classic Symbiote terminal mode).
- **Verified Quality Gates**:
  - `cargo test --no-default-features`: **7/7 automated scientific & camera assertions passing** (Experiments A–K, including `SimCamera` projection, cursor-centered zoom invariance, aspect-ratio `fit_world`, frustum culling, and `v4` persistence roundtrip).
  - `--experiment`: **79.2 ticks/sec** (`GPU-Compute + Rayon`) with `87` organisms, `256×144×4` multi-biome `EcoField`, `40×40×3` Lenia bodies, procedural anatomy, and NEAT `NeuralGraph`.
  - `--snapshot`: Visually verified `ecosystem_overview.png` and `microscopic_cells.png` — **zero world repetition**, crisp world boundary framing, distinct multi-biome filaments, and articulated organism anatomy.
  - Live Windows executable verified across 3 cycles of `2560×1440` Maximize → Minimize (`0×0`) → Restore with `ALIVE_AFTER_STRESS=True` and `CRASH_COUNT=0`.

## Key Modules (`src/convergence/`)
- `camera.rs`: `WorldBounds` (`[0, width) × [0, height)`) and aspect-aware `SimCamera` (`fit_world`, cursor-centered zoom, frustum culling, minimap projection)
- `genomorph.rs`: Hierarchical `GenomorphGenome`, persistent homology `gene_id`s, 10 `OrganKind`s, structural mutations, `diff_genomes`, `LifeStage`, `AnatomyPhenotype`
- `neural_graph.rs`: NEAT-inspired `NeuralGraph` (`10` inputs → evolvable hidden nodes → `4` outputs)
- `lenia_body.rs`: `40×40×3` smooth diffusive Lenia cellular bodies & `MorphGenome`
- `eco_field.rs` & `gpu_compute.rs`: Toroidal `256×144×4` multi-biome reaction-diffusion field (WGSL compute + Rayon CPU fallback)
- `engine.rs`: Unified multiscale `ConvergenceWorld` (widescreen `WorldBounds`, embryogenesis, digestive biology, directional jaw predation, armor defense, spatial hashing)
- `universes.rs`: 12 Experimental Universes with configurable world sizes (`16×9`, `20×11.25`, `24×13.5`)
- `phylogeny.rs`: Individual `OrganismRecord` ancestry history + `6×6` MAP-Elites archive
- `persistence.rs`: Versioned `v4` atomic world persistence (`saves/worlds/*.abio.json`)
- `observatory.rs` & `snapshot.rs`: Non-tiling WGSL procedural anatomy + Lenia renderer, CPU frustum culling, interactive World Minimap, live `NeuralGraph` visualizer, Homology Diff UI, and headless `1440×810` PNG renderer
- `crash_log.rs`: Global panic hook & runtime event logging

## Recommended Next Steps (v0.6.0)
1. Move `40×40×3` Lenia body stepping into a batched WGSL compute shader so both `EcoField` and organism bodies remain GPU-resident for 500–1,000+ simultaneous organisms.
2. Wire contact-triggered sexual recombination (`crossover_homologous`) as an optional reproductive strategy alongside mitotic division.
3. Add a dedicated graphical branching Phylogenetic Tree canvas & side-by-side multi-generation specimen gallery window inside `observatory.rs`.
