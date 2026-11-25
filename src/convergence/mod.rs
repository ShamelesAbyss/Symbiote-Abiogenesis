//! SYMBIOTE: ABIOGENESIS — EVOLUTION UNBOUND
//!
//! Unifies Primordia's continuous cellular automata (Lenia) with Symbiote's
//! evolutionary particle chemistry, ecological fields, and species lineages.
//!
//! Mathematical formulations for Lenia kernel shells and Gaussian growth
//! functions are adapted with attribution from Lenia (Bert Chan, 2018, MIT)
//! and Primordia (ShamelesAbyss).

pub mod eco_field;
pub mod engine;
pub mod experiments;
#[cfg(feature = "gpu")]
pub mod gpu_compute;
pub mod lenia_body;
#[cfg(feature = "observatory")]
pub mod observatory;
pub mod persistence;
pub mod phylogeny;
pub mod snapshot;
pub mod universes;
