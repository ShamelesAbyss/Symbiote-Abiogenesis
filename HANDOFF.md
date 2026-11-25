# HANDOFF.md — Project State & Continuation Notes

## Current Status (v0.1.0)
- **Repository**: `https://github.com/ShamelesAbyss/Symbiote-Abiogenesis` (`main` branch)
- **Originals Preserved**: Both `Symbiote` and `Primordia` remain unmodified upstream references.
- **Validated Backends**:
  - `GPU-Compute + Rayon` (`wgpu` 24 + `ECO_FIELD_WGSL`): ~193 ticks/sec @ 96 organisms
  - `CPU-Rayon` (`--cpu`): ~159 ticks/sec @ 96 organisms
  - Pure CPU (`--no-default-features`): compiles cleanly with zero GPU/windowing dependencies

## Key Entry Points
- `src/main.rs`: CLI dispatch (`--observatory`, `--experiment`, `--snapshot`, `--cpu`, or default TUI)
- `src/convergence/lenia_body.rs`: Localized `24x24x2` continuous Lenia bodies & `MorphGenome`
- `src/convergence/eco_field.rs`: `128x128x4` continuous reaction-diffusion ecological field
- `src/convergence/engine.rs`: Unified 6-scale `ConvergenceWorld`
- `src/convergence/gpu_compute.rs`: `wgpu` 24 compute pipeline for `EcoField`
- `src/convergence/observatory.rs`: Interactive `winit` + `wgpu` graphical observatory
- `src/convergence/snapshot.rs`: Headless `1280x720` PNG screenshot renderer

## Recommended Next Steps
1. Move `LeniaBody::step_cpu` into a batched WGSL compute shader in `gpu_compute.rs` so both `EcoField` and all `LeniaBody` instances remain GPU-resident with zero per-frame readback in `--observatory` mode.
2. Add spatial hashing in `ConvergenceWorld::step` to scale `MAX_ORGANISMS` from 96 to 512+.
3. Add interactive click-to-inspect organism HUD in `observatory.rs`.
