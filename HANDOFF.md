# HANDOFF.md — Project State & Cross-Agent Continuity (v0.4.0)

## Current Status (v0.4.0 — Project GENOMORPH)
- **Repository**: `https://github.com/ShamelesAbyss/Symbiote-Abiogenesis` (`main` branch exclusively)
- **Upstream Originals Preserved**: `Symbiote`, `Primordia`, and `Hydra` remain completely untouched.
- **Default Entrypoint**: Double-clicking `symbiote-abiogenesis.exe` launches the native `winit` + `wgpu` 24 + `egui` 0.31 **GENOMORPH Research Laboratory** (`--legacy-tui` for classic Symbiote terminal mode).
- **Verified Quality Gates**:
  - `cargo test --no-default-features`: **6/6 automated Experiments A–J assertions passing** (now explicitly executed in GitHub Actions CI on Ubuntu & Windows).
  - `--experiment`: **100.8 ticks/sec** (`GPU-Compute + Rayon`) with `40×40×3` Lenia bodies + procedural anatomy + NEAT `NeuralGraph` + toroidal `EcoField`; *Predator's Paradise* validated 992 anatomy-driven predations, 61 births, 12 deaths, 10 lineage extinctions, and 12/36 MAP-Elites niches.
  - Live Windows executable verified across multi-cycle `2560×1440` Maximize/Minimize/Restore stress testing with 0 crashes; persistent crash logging writes to `logs/crash_<ts>.log` and `logs/runtime.log`.

## Key Modules (`src/convergence/`)
- `genomorph.rs`: Hierarchical `GenomorphGenome`, persistent homology `gene_id`s, 10 `OrganKind`s, structural mutations, `diff_genomes`, `LifeStage`, `AnatomyPhenotype`
- `neural_graph.rs`: NEAT-inspired `NeuralGraph` (`10` inputs → evolvable hidden nodes → `4` outputs)
- `lenia_body.rs`: `40×40×3` smooth diffusive Lenia cellular bodies & `MorphGenome`
- `eco_field.rs` & `gpu_compute.rs`: Toroidal `128×128×4` reaction-diffusion field (WGSL compute + Rayon CPU fallback)
- `engine.rs`: Unified multiscale `ConvergenceWorld` (embryogenesis, digestive biology, directional jaw predation, armor defense, kin symbiosis, spatial hashing)
- `universes.rs`: 12 Experimental Universes (including *Genesis Zero*, *Predator's Paradise*, *The Living Reef*, *Evolutionary Crucible*)
- `phylogeny.rs`: Individual `OrganismRecord` ancestry history + `6×6` MAP-Elites archive
- `persistence.rs`: Versioned `v3` atomic world persistence (`saves/worlds/*.abio.json`)
- `observatory.rs` & `snapshot.rs`: WGSL procedural anatomy + Lenia renderer, live `NeuralGraph` visualizer, Parent-Offspring Homology Diff UI, and headless `1440×810` PNG renderer
- `crash_log.rs`: Global panic hook & runtime event logging

## Recommended Next Steps (v0.5.0)
1. Move `40×40×3` Lenia body stepping into a batched WGSL compute shader so both `EcoField` and organism bodies remain GPU-resident for 500–1,000+ simultaneous organisms.
2. Wire contact-triggered sexual recombination (`crossover_homologous`) as an optional reproductive strategy alongside mitotic division.
3. Add a dedicated graphical branching Phylogenetic Tree canvas & side-by-side multi-generation specimen gallery window inside `observatory.rs`.
