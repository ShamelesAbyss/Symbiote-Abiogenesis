# Changelog

All notable changes to **Symbiote: Abiogenesis** are documented in this file.

## [0.2.0] - 2026-11-24 — Evolution Unbound
### Added
- **Default Native Graphical Research Laboratory** (`src/convergence/observatory.rs`):
  - Launching `symbiote-abiogenesis.exe` directly opens the `winit` + `wgpu` + `egui` Research Laboratory (classic Symbiote TUI preserved under `--legacy-tui`).
  - Collapsible scientific panels: Universe Preset selector, Play/Pause/Single-Step/Speed controls, Biosphere Telemetry, Click-to-Inspect Microscopic Organism Inspector with smooth camera tracking, MAP-Elites Discovery Archive, Atomic Save/Load World, and Distraction-Free Cinematic Mode (`H`).
- **3-Channel Multichannel Lenia Morphogenesis & Primitive Adaptive Intelligence** (`src/convergence/lenia_body.rs`):
  - Upgraded localized cellular bodies to `24×24×3` (**Cytoplasm**, **Organic Membrane**, **Internal Organelles / Motility**) with spiral-twist anisotropic kernels and saltational macro-mutations + crossover.
  - Added heritable recurrent `NeuralBrain` (6 sensory inputs → 6 recurrent hidden neurons → 3 motor/secretion outputs) driving foraging and chemical signaling.
- **Genuine Ecological Selection & Spatial Hashing** (`src/convergence/engine.rs`):
  - $O(N)$ Spatial Hash Grid scaling to 320+ continuous organisms.
  - Coherent resource accounting: predation inflicts real physical cellular damage (`apply_damage`) and transfers energy proportional to damaged biomass; symbiosis is zero-sum energy sharing; locomotion and membrane maintenance cost real metabolic energy.
- **8 Experimental Universes** (`src/convergence/universes.rs`):
  - *Primordial Ocean*, *Abyssal Biosphere*, *Genesis*, *Alien Garden*, *Evolutionary Arena*, *Extinction Protocol*, *Mathematical Anomaly*, and *Sandbox Laboratory*, including configurable permanent extinction (`allow_abiogenesis = false`).
- **Evolutionary Tree of Life & MAP-Elites Discovery Archive** (`src/convergence/phylogeny.rs`):
  - Tracks founding lineages, parent-child ancestry, extinction events, and a `6×6` MAP-Elites Quality-Diversity grid.
- **Atomic World Persistence** (`src/convergence/persistence.rs`):
  - Versioned atomic write-to-temp-then-rename persistence (`saves/worlds/*.abio.json`).
- **Automated Scientific Test Suite** (`cargo test`):
  - Assertions for numerical stability, physical damage accounting, mutation bounds & crossover, nutrient-morphology coupling, and atomic persistence round-trip.

## [0.1.0] - 2026-11-24
### Added
- Initial Project Convergence foundation (`24×24×2` Lenia bodies, `128×128×4` EcoField, WGSL GPU compute, basic observatory, CI, and upstream attribution).
