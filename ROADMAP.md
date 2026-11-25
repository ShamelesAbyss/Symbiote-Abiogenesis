# Roadmap — Symbiote: Project Abiogenesis

## Phase 1 — Core Convergence (Completed in v0.1.0)
- [x] Localized `24x24x2` continuous Lenia cellular bodies with heritable `MorphGenome`
- [x] Continuous `128x128x4` reaction-diffusion `EcoField` (Nutrient, Pheromone, Toxin, Roots)
- [x] Bidirectional coupling (nutrients -> growth, COM drift -> locomotion, death -> nutrient recycling)
- [x] `wgpu` 24 WGSL GPU compute for `EcoField` + Rayon CPU fallback + `--no-default-features` pure CPU build
- [x] Interactive `winit` + `wgpu` bioluminescent observatory with seamless zooming
- [x] Quantitative experimental validation suite (`--experiment`) and headless PNG capture (`--snapshot`)

## Phase 2 — Full GPU-Resident Organism Simulation (Next)
- [ ] Batched WGSL compute shader stepping all `LeniaBody` instances in parallel on GPU without per-frame CPU readback
- [ ] Spatial hash grid for $O(N)$ physical contact, predation, and colony formation up to 512+ simultaneous continuous organisms
- [ ] Click-to-inspect organism telemetry overlay in the observatory (genome parameters, lineage tree, energy, mass)

## Phase 3 — Deep Evolutionary Morphology
- [ ] Multi-cellular colony adhesion and emergent tissue differentiation (3-channel bodies: Cytoplasm, Membrane, Organelle/Motility)
- [ ] Sexual crossover between compatible colliding organisms' `MorphGenome`s
- [ ] Persistent phylogenetic bestiary and genome vault export/import (JSON)
