# HANDOFF.md — Project State & Continuation Notes

## Current Status (v0.2.0 — Evolution Unbound)
- **Repository**: `https://github.com/ShamelesAbyss/Symbiote-Abiogenesis` (`main` branch)
- **Originals Preserved**: Both `Symbiote` and `Primordia` remain unmodified upstream references.
- **Default Entrypoint**: Double-clicking `symbiote-abiogenesis.exe` launches the native `winit` + `wgpu` + `egui` **Abiogenesis Research Laboratory** (`--legacy-tui` for classic Symbiote terminal mode).
- **Validated Backends & Tests**:
  - `cargo test --no-default-features`: 3/3 automated assertions passing
  - `GPU-Compute + Rayon` (`--experiment`): **184.8 ticks/sec** with 3-channel Lenia + `NeuralBrain` + Spatial Hashing + `EcoField` WGSL compute
  - Live Windows executable verified running (`Title='SYMBIOTE: ABIOGENESIS — EVOLUTION UNBOUND'`)

## Key Modules (`src/convergence/`)
- `lenia_body.rs`: 3-channel `24×24×3` Lenia bodies, `MorphGenome`, `NeuralBrain`
- `eco_field.rs`: `128×128×4` continuous reaction-diffusion ecological field
- `engine.rs`: Unified 6-scale `ConvergenceWorld` with $O(N)$ spatial hashing & coherent resource accounting
- `universes.rs`: 8 Experimental Universes + configurable permanent extinction
- `phylogeny.rs`: Evolutionary lineage tracker + `6×6` MAP-Elites Discovery Archive
- `persistence.rs`: Versioned atomic world save/load (`saves/worlds/*.abio.json`)
- `gpu_compute.rs`: `wgpu` 24 WGSL compute pipeline for `EcoField`
- `observatory.rs`: Native `egui` + `wgpu` Research Laboratory & 3-channel fluorescence shader
- `snapshot.rs`: Headless `1280×720` PNG screenshot renderer

## Recommended Next Steps (v0.3.0)
1. Move `LeniaBody::step_cpu` into a batched WGSL compute shader so both `EcoField` and all `LeniaBody` instances remain GPU-resident with zero per-frame readback.
2. Add an interactive graphical Evolutionary Tree of Life canvas inside `observatory.rs` to browse ancestral lineages and resurrect archived MAP-Elites organisms into the live world.
3. Add multi-cellular colony adhesion bonds and sexual crossover on compatible organism contact.
