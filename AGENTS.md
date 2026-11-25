# AGENTS.md — Development Guidelines for Autonomous & Human Contributors

## Repository Rules
- **Primary Language**: Rust (2021 edition). WGSL, CUDA, and C++ are permitted where justified. **No Python.**
- **Branching**: Work on `main` exclusively with clean, incremental, validated commits.
- **Upstream Preservation**: Original `Symbiote` and `Primordia` repositories are read-only upstream references. Preserve MIT attribution in `THIRD_PARTY_LICENSES.md`.
- **Hardware Target**: NVIDIA RTX 4070 Laptop GPU (8 GB VRAM) and 32 GB RAM, while **always maintaining CPU fallback** (`--cpu` and `--no-default-features`).

## Build & Validation Commands
```bash
# Default GPU + Observatory release build & validation
cargo run --release -- --experiment

# CPU fallback validation
cargo run --release -- --experiment --cpu

# Pure CPU build check (must always compile cleanly)
cargo check --no-default-features

# Regenerate real simulation screenshots
cargo run --release -- --snapshot
```

## Architecture Invariants
1. **No Decorative Skins**: Visual appearance in `observatory.rs` and `snapshot.rs` must directly render the actual continuous `LeniaBody` and `EcoField` state buffers.
2. **Bidirectional Coupling**: Changes to `MorphGenome` or `EcoField` must measurably affect both cellular dynamics and ecological survival/reproduction.
3. **Preserve Classic TUI**: Running `cargo run` with no flags must continue launching the original Symbiote terminal simulation.
