# Changelog

All notable changes to **Symbiote: Abiogenesis** are documented in this file.

## [0.5.0] - 2026-11-25 — Project INFINITY
### Fixed & Changed
- **Eliminated Visible World Repetition & Duplicated Selection Rings** (`src/convergence/camera.rs`, `observatory.rs`, `snapshot.rs`):
  - Separated simulation topology (toroidal wrapping on `[0, world_w) × [0, world_h)`) from camera/rendering topology.
  - Both `OBSERVATORY_WGSL` and `snapshot.rs` strictly clip outside `[0, world_w] × [0, world_h]` to deep outer void with an optional subtle world-boundary frame, guaranteeing each organism and selection indicator appears strictly once in full-world overview (allowing only clipped seam fragments on toroidal edges).
  - Updated mouse hit-testing (`select_organism_at_cursor`) to reject clicks in the outer void and map screen pixels through `SimCamera::screen_to_world`.
- **Expansive Widescreen Multi-Biome Worlds** (`src/convergence/eco_field.rs`, `universes.rs`, `engine.rs`):
  - Expanded world coordinate space from `[-1, 1] × [-1, 1]` to configurable widescreen `WorldBounds` (`16×9`, `20×11.25`, `24×13.5` across the 12 experimental universes).
  - Upgraded `EcoField` from `128×128×4` to `256×144×4` with seed-dependent multi-biome structure (hydrothermal vents, toxic trenches, sinuous mycelial ridges).
- **Professional `SimCamera`, Frustum Culling & Navigable Minimap** (`src/convergence/camera.rs`, `observatory.rs`):
  - Added aspect-ratio-aware `SimCamera` (`fit_world`, cursor-centered wheel zoom, smooth keyboard panning, bounds clamping, organism tracking).
  - Added CPU frustum culling (`camera.is_visible(...)`) before GPU buffer upload so off-screen organisms are skipped while maintaining exact `selected_idx` mapping.
  - Added an interactive **World Minimap** in the `egui` Research Laboratory showing all organisms and the clamped viewport rectangle (click/drag to navigate), plus **Fit World (0)** and **Bounds** toggles.
- **Versioned v4 Persistence & Camera Unit Tests** (`persistence.rs`, `experiments.rs`):
  - Upgraded atomic world saves to `v4` (`256×144×4` `EcoField` + `WorldBounds`).
  - Added `exp_k_infinity_camera_and_world_bounds` automated test covering `fit_world` across 16:9 / 21:9 / 4:3, `screen_to_world` ↔ `world_to_uv` roundtrip, cursor-centered zoom invariance, toroidal delta symmetry, and frustum culling (7/7 passing in CI).

## [0.4.0] - 2026-11-25 — Project GENOMORPH
### Added
- **GENOMORPH Developmental Genetics & Procedural Anatomy** (`src/convergence/genomorph.rs`):
  - Hierarchical `GenomorphGenome` encoding `AnatomyGene`s with persistent homology `gene_id`s across 10 functional organ kinds (`CoreSegment`, `BodySegment`, `PropulsionFin`, `UndulatingTail`, `FeedingJaw`, `FilterMembrane`, `SensoryAntenna`, `ArmorCarapace`, `DigestiveVacuole`, `ToxinGland`).
  - True structural mutations (`DuplicateGene`, `DeleteGene`, `AddInnovation`, `SpecializeOrgan`, `SymmetryFlip`, `ParametricDrift`) and homology-matched `crossover_homologous`.
  - Artificial Embryogenesis (`LifeStage`: `Embryo → Juvenile → Adult → Elder`) where developmental maturity and childhood nutrition shape adult scale and organ power.
  - Ancestor-descendant homology diffing (`diff_genomes`) reporting retained, modified, added, and lost anatomical genes.
- **NEAT-Inspired Evolving Neural Graph** (`src/convergence/neural_graph.rs`):
  - Replaced fixed `6→6→3` network with innovation-tracked `NeuralGraph`: 10 grounded sensory inputs (range gated by `SensoryAntenna`) → evolvable hidden interneurons (`AddNode`, `AddConnection`, `ToggleConnection`, `Tanh`/`Sigmoid`/`Sin`/`Relu`) → 4 motor/physiological outputs (`thrust`, `turn`, `bite`, `secrete`).
- **Digestive Biology & Anatomy-Driven Predation** (`src/convergence/engine.rs`):
  - Gradual plant/meat stomach digestion governed by heritable `diet_carnivory` and `digestion_rate`; directional jaw bite attacks mitigated by `ArmorCarapace` rating; kin energy sharing.
- **Expanded Experimental Universes** (`src/convergence/universes.rs`):
  - Expanded to 12 universes including *Genesis Zero* (minimal 2-gene founders), *Predator's Paradise*, *The Living Reef*, and *Evolutionary Crucible*.
- **GPU Procedural Anatomy Rendering & Research Microscope** (`src/convergence/observatory.rs`, `snapshot.rs`):
  - Added GPU storage buffer for articulated procedural organs rendered seamlessly alongside `40×40×3` continuous Lenia bodies.
  - Added live interactive `NeuralGraph` visualizer, Parent-Offspring Homology Diff inspector, and MAP-Elites specimen resurrection (`[USER INTERVENTION]`).
- **Versioned v3 Persistence & Automated CI Tests** (`persistence.rs`, `experiments.rs`, `.github/workflows/ci.yml`):
  - Upgraded atomic saves to `v3` with strict size/version validation and RNG state preservation.
  - Implemented Experiments A–J automated `cargo test` suite and enabled `cargo test --no-default-features` directly in GitHub Actions CI.
  - Documented ALife research synthesis in `docs/RESEARCH_GENOMORPH.md`.

## [0.3.2] - 2026-11-24 — Surface Lifecycle & Crash Diagnostics
### Added & Fixed
- Persistent crash & runtime logging (`logs/crash_<ts>.log`, `logs/runtime.log`), zero-churn reusable GPU upload buffers, minimize guards, pre-render surface sync, and `F11`/`Alt+Enter` borderless fullscreen.

## [0.3.1] - 2026-11-24 — Fullscreen & High-DPI Fix
### Fixed
- Resolved `2048px` `downlevel_defaults` crash on `2560×1440` / 4K displays by adopting `adapter.limits()`.

## [0.3.0] - 2026-11-24 — Project Bioluminescence
### Changed & Added
- Upgraded to `40×40×3` smooth diffusive Lenia bodies, toroidal `EcoField` & movement, near-black ACES bioluminescent shaders, and custom obsidian-glass `egui` Research Laboratory.

## [0.2.0] - 2026-11-24 — Evolution Unbound
### Added
- Default native `egui` + `wgpu` Research Laboratory, 3-channel Lenia, spatial hashing, 8 universes, MAP-Elites archive, and atomic persistence.

## [0.1.0] - 2026-11-24
### Added
- Initial Project Convergence foundation (`24×24×2` Lenia, `128×128×4` EcoField, WGSL compute, CI).
