# Changelog

All notable changes to **Symbiote: Abiogenesis** are documented in this file.

## [0.3.0] - 2026-11-24 — Project Bioluminescence
### Changed & Added
- **Radical Visual & Morphological Overhaul** (`src/convergence/lenia_body.rs`, `observatory.rs`, `snapshot.rs`):
  - Upgraded localized continuous cellular bodies from `24×24×3` to **`40×40×3`** with 9-tap Gaussian smoothing to eliminate Nyquist striping and pixelation.
  - Added genome-driven morphogenetic envelopes (`axial_stretch`, `tendril_affinity`, `angular_lobes`, `spiral_twist`) producing genuinely distinct, organic alien silhouettes: asymmetric swimmers, undulating tendrils, spiral vortices, stellate blooms, and crescent gliders.
  - Replaced flat turquoise background with a **near-black abyssal void**, subtle optical vignette, drifting marine-snow bokeh, sinuous mycelial filaments, Hermite-smoothed interpolation, translucent refractive membranes, luminous internal organelles, and ACES filmic tone mapping.
- **Toroidal World Topology** (`src/convergence/eco_field.rs`, `engine.rs`, `gpu_compute.rs`):
  - Eliminated organism boundary congregation completely by upgrading both `EcoField` (CPU & WGSL) and `ConvergenceWorld` to seamless toroidal wrapping with velocity-aligned organism headings.
- **Custom Obsidian-Glass Research Laboratory UI** (`src/convergence/observatory.rs`):
  - Bespoke dark obsidian & bioluminescent-cyan `egui` theme, metric pill cards, 3-channel cellular density bars, live Population & Nutrient sparkline charts, and a sleek Cinematic Observation Mode (`H`).

## [0.2.0] - 2026-11-24 — Evolution Unbound
### Added
- Default native `egui` + `wgpu` Research Laboratory, 3-channel Lenia morphogenesis, heritable `NeuralBrain`, $O(N)$ spatial hashing, coherent predation & resource accounting, 8 Experimental Universes, MAP-Elites Discovery Archive, atomic world persistence, and automated `cargo test` suite.

## [0.1.0] - 2026-11-24
### Added
- Initial Project Convergence foundation (`24×24×2` Lenia bodies, `128×128×4` EcoField, WGSL GPU compute, basic observatory, CI, and upstream attribution).
