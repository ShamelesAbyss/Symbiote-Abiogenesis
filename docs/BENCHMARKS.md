# Benchmarks & Validation Results (v0.3.0)

All measurements below were collected on the reference development machine using `cargo run --release -- --experiment` and `cargo test --no-default-features` at commit `c7df3a8`.

## Reference Hardware
- **CPU**: Intel Core i9-14900HX (24 cores / 32 threads)
- **GPU**: NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM, `wgpu` 24 DX12/Vulkan)
- **RAM**: 32 GB DDR5 · **OS**: Windows 11 x86_64 · **Display**: 2560×1440 @ 144 Hz

## Simulation Throughput & Field-Buffer Memory

> Note: **Simulation ticks/sec** measures full headless multiscale stepping (`EcoField` WGSL compute + Rayon `40×40×3` Lenia bodies + `NeuralBrain` + spatial hash + Symbiote coupling). **Field-buffer memory** measures continuous cellular and ecological `f32` double-buffers only, not total OS process RAM.

| Build / Configuration | Organism Count | Body Resolution | Simulation Throughput | Continuous Field Buffers |
| :--- | :---: | :---: | :---: | :---: |
| **v0.3.0** (`GPU-Compute + Rayon`) | 256 | `40×40×3` | **47.8 ticks/sec** | 10,112 KB (~9.9 MB) |
| **v0.2.0** *(historical, `GPU-Compute + Rayon`)* | 79 | `24×24×3` | 184.8 ticks/sec | 1,578 KB (~1.5 MB) |
| **v0.1.0** *(historical, `GPU-Compute + Rayon`)* | 96 | `24×24×2` | 193.2 ticks/sec | 1,376 KB (~1.3 MB) |
| **v0.1.0** *(historical, `CPU-Rayon`)* | 96 | `24×24×2` | 159.2 ticks/sec | 1,376 KB (~1.3 MB) |

## Experimental Validation (`--experiment`, v0.3.0)

1. **Morphological Stability (Exp 1 — 200 steps/tribe)**: All 6 founding tribes (`BLD`, `MOS`, `DEP`, `SOL`, `DRM`, `STC`) achieved **200/200 steps** persistence with finite cytoplasm (`0.151–0.175`), membrane (`0.216–0.360`), and organelle channels.
2. **Bidirectional Nutrient Coupling (Exp 2 — 120 steps)**: Nutrient-rich (`0.90`) sustained higher mass (`0.178`) and membrane (`0.405`) than nutrient-poor + toxin (`0.02`, mass `0.171`).
3. **Evolution, Predation & MAP-Elites (Exp 3 — 500 ticks, Evolutionary Arena)**:
   - Grew from `64` → `270` organisms across `Calm → Hunger → Bloom`.
   - Recorded **245 births**, **39 deaths**, **2,181 physical predation events**, **3 lineage extinctions**, **Gen 4**, and **17/36 MAP-Elites niches** discovered.

## Automated Tests & GitHub Actions CI
- `cargo test --no-default-features`: **3/3 passed** (`test_numerical_stability_and_damage_accounting`, `test_mutation_bounds_and_crossover`, `test_atomic_persistence_roundtrip`).
- GitHub Actions CI (`.github/workflows/ci.yml`): **Passing** on `ubuntu-latest` (pure CPU), `windows-latest` (pure CPU), and `windows-latest` (full GPU + Observatory release build).
