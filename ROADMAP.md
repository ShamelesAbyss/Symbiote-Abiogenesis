# Roadmap — Symbiote: Abiogenesis

This roadmap distinguishes verified implemented capabilities from experimental, planned, and long-term research directions.

## Implemented (v0.1.0 – v0.3.0)
- **Multichannel Morphogenesis**: Localized `40×40×3` continuous Lenia cellular bodies (Cytoplasm, Membrane, Organelles) with 9-tap Gaussian smoothing and heritable morphogenetic parameters (`mu`, `sigma`, `ring_peaks`, `angular_lobes`, `spiral_twist`, `axial_stretch`, `tendril_affinity`).
- **Primitive Adaptive Intelligence**: Heritable `6 → 6 (recurrent) → 3` `NeuralBrain` sensing local nutrients, gradients, toxins, pheromones, and internal energy.
- **Toroidal Reaction-Diffusion Ecology**: `128×128×4` toroidal `EcoField` accelerated via `wgpu` 24 WGSL compute with Rayon CPU fallback and `--no-default-features` pure CPU build.
- **Coherent Ecological Selection**: $O(N)$ spatial hash grid, physical predation damage & biomass transfer, zero-sum symbiosis, kinetic/membrane maintenance costs, and nutrient recycling on death.
- **8 Experimental Universes**: *Primordial Ocean*, *Abyssal Biosphere*, *Genesis*, *Alien Garden*, *Evolutionary Arena*, *Extinction Protocol*, *Mathematical Anomaly*, and *Sandbox Laboratory* (including permanent extinction mode).
- **Phylogeny & Discovery Archive**: Lineage tracking, extinction counter, and `6×6` MAP-Elites Quality-Diversity grid.
- **Atomic Persistence**: Versioned (`v2`) atomic save/load (`saves/worlds/*.abio.json`).
- **Bioluminescence Research Laboratory**: Default native `winit` + `wgpu` + `egui` 0.31 application with custom obsidian-glass theme, ACES bioluminescent shader, live sparklines, microscopic tracking, and Cinematic Mode (`H`).

## Experimental
- **Genetic Crossover**: `MorphGenome::crossover` is implemented and unit-tested; live contact-triggered sexual recombination is not yet wired into `ConvergenceWorld::step`.
- **Automated Discovery Scoring**: MAP-Elites currently bins on elongation × organelle/membrane complexity; additional behavioral novelty descriptors are under evaluation.

## In Development / Proposed (Next Milestones)
- **GPU-Resident Batched Lenia Compute**: Move `40×40×3` organism stepping into a batched WGSL compute shader to eliminate per-frame body upload and scale beyond 1,000 simultaneous organisms.
- **Interactive Phylogenetic Tree Viewer**: Graphical branching tree canvas inside the Research Laboratory allowing resurrection of archived MAP-Elites specimens.
- **Multi-Slot World Manager**: In-app browser for creating, duplicating, and resuming multiple named persistent worlds.

## Long-Term Research
- **Emergent Multicellularity**: Physical adhesion bonds and resource transport between cooperating cellular bodies forming macro-colonies.
- **Multi-Week Unattended Biosphere Runs**: Automated checkpointing, phylogenetic drift logging, and long-duration stability telemetry.
