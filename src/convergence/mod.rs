//! SYMBIOTE: ABIOGENESIS — PROJECT GENOMORPH (v0.4.0)
//!
//! Unifies procedural developmental anatomy (`genomorph`), NEAT-inspired neural
//! evolution (`neural_graph`), continuous 3-channel Lenia morphology (`lenia_body`),
//! and toroidal reaction-diffusion ecology (`eco_field`).

pub mod camera;
pub mod crash_log;
pub mod eco_field;
pub mod engine;
pub mod experiments;
pub mod genomorph;
#[cfg(feature = "gpu")]
pub mod gpu_compute;
pub mod lenia_body;
pub mod neural_graph;
#[cfg(feature = "observatory")]
pub mod observatory;
pub mod persistence;
pub mod phylogeny;
pub mod snapshot;
pub mod universes;
