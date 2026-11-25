# Changelog

All notable changes to **Symbiote: Project Abiogenesis** are documented in this file.

## [0.1.0] - 2026-11-24
### Added
- **Project Convergence Architecture** (`src/convergence/`):
  - `lenia_body.rs`: Localized `24x24x2` continuous Lenia cellular bodies (Cytoplasm + Membrane channels), heritable `MorphGenome`, radial/angular kernels, Gaussian growth, chemotaxis, regeneration, and mitotic division.
  - `eco_field.rs`: Continuous `128x128x4` planetary reaction-diffusion ecological field (Nutrient, Pheromone, Toxin, Living Root Network) coupled bidirectionally with Symbiote's `Ecology` zones and `CellularAutomata`.
  - `engine.rs`: Unified 6-scale `ConvergenceWorld` integrating Symbiote particle chemistry, localized Lenia bodies, physical membrane collisions, predation, symbiosis, and nutrient recycling on death.
  - `gpu_compute.rs`: `wgpu` 24 WGSL compute pipeline (`ECO_FIELD_WGSL`) with automatic Rayon CPU fallback.
  - `observatory.rs`: Interactive `winit` 0.30 + `wgpu` 24 bioluminescent graphical observatory (`--observatory`) with seamless planetary-to-microscopic zooming.
  - `snapshot.rs`: Headless `1280x720` PNG screenshot renderer (`--snapshot`).
  - `experiments.rs`: 4-part quantitative experimental validation suite (`--experiment`).
- **Governance & CI**:
  - `THIRD_PARTY_LICENSES.md` preserving Bert Chan (2018) / Lenia and Primordia MIT attribution.
  - GitHub Actions CI workflow (`.github/workflows/ci.yml`).
  - `README.md`, `CHANGELOG.md`, `ROADMAP.md`, `AGENTS.md`, and `HANDOFF.md`.
