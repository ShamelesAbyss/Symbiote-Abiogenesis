//! Experimental Universes for Symbiote: Abiogenesis — Evolution Unbound.
//!
//! Each universe defines genuinely distinct physical, chemical, and evolutionary
//! rules rather than merely cosmetic palette differences.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UniversePreset {
    PrimordialOcean,
    AbyssalBiosphere,
    Genesis,
    AlienGarden,
    EvolutionaryArena,
    ExtinctionProtocol,
    MathematicalAnomaly,
    SandboxLaboratory,
}

impl UniversePreset {
    pub const ALL: [UniversePreset; 8] = [
        Self::PrimordialOcean,
        Self::AbyssalBiosphere,
        Self::Genesis,
        Self::AlienGarden,
        Self::EvolutionaryArena,
        Self::ExtinctionProtocol,
        Self::MathematicalAnomaly,
        Self::SandboxLaboratory,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::PrimordialOcean => "Primordial Ocean",
            Self::AbyssalBiosphere => "Abyssal Biosphere",
            Self::Genesis => "Genesis",
            Self::AlienGarden => "Alien Garden",
            Self::EvolutionaryArena => "Evolutionary Arena",
            Self::ExtinctionProtocol => "Extinction Protocol",
            Self::MathematicalAnomaly => "Mathematical Anomaly",
            Self::SandboxLaboratory => "Sandbox Laboratory",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::PrimordialOcean => "Nutrient-rich primordial sea with rapid cellular diversification.",
            Self::AbyssalBiosphere => "Dark, scarce deep-sea biosphere rewarding metabolic efficiency.",
            Self::Genesis => "Minimal seeding of simple protocells; extinction is permanent.",
            Self::AlienGarden => "High-symbiosis world with lush living roots and branching colonies.",
            Self::EvolutionaryArena => "High-competition arena with intense predation and selection.",
            Self::ExtinctionProtocol => "Periodic environmental catastrophes creating evolutionary bottlenecks.",
            Self::MathematicalAnomaly => "Extreme spiral/multi-lobe kernels and heightened mutation.",
            Self::SandboxLaboratory => "Balanced, configurable research laboratory.",
        }
    }

    pub fn config(self) -> UniverseConfig {
        match self {
            Self::PrimordialOcean => UniverseConfig {
                preset: self,
                initial_organisms: 32,
                max_organisms: 72,
                nutrient_regen_mult: 1.25,
                metabolic_cost_mult: 0.95,
                mutation_rate: 0.055,
                allow_abiogenesis: true,
                catastrophe_interval: 0,
                anomaly_twist_boost: 0.0,
            },
            Self::AbyssalBiosphere => UniverseConfig {
                preset: self,
                initial_organisms: 36,
                max_organisms: 220,
                nutrient_regen_mult: 0.55,
                metabolic_cost_mult: 1.25,
                mutation_rate: 0.050,
                allow_abiogenesis: false,
                catastrophe_interval: 0,
                anomaly_twist_boost: 0.25,
            },
            Self::Genesis => UniverseConfig {
                preset: self,
                initial_organisms: 14,
                max_organisms: 256,
                nutrient_regen_mult: 1.05,
                metabolic_cost_mult: 0.95,
                mutation_rate: 0.075,
                allow_abiogenesis: false,
                catastrophe_interval: 0,
                anomaly_twist_boost: 0.0,
            },
            Self::AlienGarden => UniverseConfig {
                preset: self,
                initial_organisms: 44,
                max_organisms: 280,
                nutrient_regen_mult: 1.45,
                metabolic_cost_mult: 0.85,
                mutation_rate: 0.045,
                allow_abiogenesis: true,
                catastrophe_interval: 0,
                anomaly_twist_boost: 0.35,
            },
            Self::EvolutionaryArena => UniverseConfig {
                preset: self,
                initial_organisms: 64,
                max_organisms: 320,
                nutrient_regen_mult: 0.95,
                metabolic_cost_mult: 1.15,
                mutation_rate: 0.065,
                allow_abiogenesis: false,
                catastrophe_interval: 0,
                anomaly_twist_boost: 0.10,
            },
            Self::ExtinctionProtocol => UniverseConfig {
                preset: self,
                initial_organisms: 56,
                max_organisms: 256,
                nutrient_regen_mult: 1.10,
                metabolic_cost_mult: 1.05,
                mutation_rate: 0.080,
                allow_abiogenesis: false,
                catastrophe_interval: 450,
                anomaly_twist_boost: 0.15,
            },
            Self::MathematicalAnomaly => UniverseConfig {
                preset: self,
                initial_organisms: 42,
                max_organisms: 240,
                nutrient_regen_mult: 1.15,
                metabolic_cost_mult: 0.95,
                mutation_rate: 0.115,
                allow_abiogenesis: true,
                catastrophe_interval: 0,
                anomaly_twist_boost: 1.40,
            },
            Self::SandboxLaboratory => UniverseConfig {
                preset: self,
                initial_organisms: 40,
                max_organisms: 256,
                nutrient_regen_mult: 1.00,
                metabolic_cost_mult: 1.00,
                mutation_rate: 0.055,
                allow_abiogenesis: true,
                catastrophe_interval: 0,
                anomaly_twist_boost: 0.0,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct UniverseConfig {
    pub preset: UniversePreset,
    pub initial_organisms: usize,
    pub max_organisms: usize,
    pub nutrient_regen_mult: f32,
    pub metabolic_cost_mult: f32,
    pub mutation_rate: f32,
    /// When false, extinction is permanent (no artificial protocell injection).
    pub allow_abiogenesis: bool,
    /// Periodic catastrophe interval in ticks (0 = disabled).
    pub catastrophe_interval: u64,
    /// Extra spiral/anisotropy boost for mathematical anomaly worlds.
    pub anomaly_twist_boost: f32,
}
