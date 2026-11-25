//! Evolutionary Tree of Life & Automated MAP-Elites Discovery Archive.
//!
//! Tracks founding lineages, parent-child ancestry, extinction events,
//! and automatically discovers morphologically and behaviorally diverse
//! organisms using a 2D Quality-Diversity (MAP-Elites) grid.

use crate::convergence::lenia_body::{BodyMetrics, MorphGenome};
use crate::particle::Tribe;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const ARCHIVE_BINS: usize = 6;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LineageRecord {
    pub lineage_id: u64,
    pub founder_tribe: Tribe,
    pub founded_tick: u64,
    pub last_seen_tick: u64,
    pub total_members: u64,
    pub living_members: usize,
    pub max_generation: u32,
    pub peak_energy: f32,
    pub extinct: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DiscoveryEntry {
    pub organism_id: u64,
    pub lineage_id: u64,
    pub discovered_tick: u64,
    pub generation: u32,
    pub tribe: Tribe,
    pub label: String,
    pub fitness: f32,
    pub morph_genome: MorphGenome,
    pub metrics: BodyMetrics,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PhylogenyArchive {
    pub lineages: HashMap<u64, LineageRecord>,
    /// 6x6 MAP-Elites grid: [elongation_bin][activity_bin] -> best discovered organism
    pub map_elites: Vec<Option<DiscoveryEntry>>,
    pub total_discoveries: u64,
    pub extinction_events: u64,
}

impl PhylogenyArchive {
    pub fn new() -> Self {
        Self {
            lineages: HashMap::new(),
            map_elites: vec![None; ARCHIVE_BINS * ARCHIVE_BINS],
            total_discoveries: 0,
            extinction_events: 0,
        }
    }

    pub fn record_birth(&mut self, lineage_id: u64, tribe: Tribe, generation: u32, tick: u64) {
        let entry = self.lineages.entry(lineage_id).or_insert(LineageRecord {
            lineage_id,
            founder_tribe: tribe,
            founded_tick: tick,
            last_seen_tick: tick,
            total_members: 0,
            living_members: 0,
            max_generation: 0,
            peak_energy: 0.0,
            extinct: false,
        });
        entry.total_members += 1;
        entry.living_members += 1;
        entry.max_generation = entry.max_generation.max(generation);
        entry.last_seen_tick = tick;
        entry.extinct = false;
    }

    pub fn record_death(&mut self, lineage_id: u64, tick: u64) {
        if let Some(entry) = self.lineages.get_mut(&lineage_id) {
            entry.living_members = entry.living_members.saturating_sub(1);
            entry.last_seen_tick = tick;
            if entry.living_members == 0 && !entry.extinct {
                entry.extinct = true;
                self.extinction_events += 1;
            }
        }
    }

    /// Evaluate an organism for the MAP-Elites Quality-Diversity Discovery Archive.
    pub fn consider_discovery(
        &mut self,
        org_id: u64,
        lineage_id: u64,
        generation: u32,
        tribe: Tribe,
        energy: f32,
        age: u64,
        morph_genome: MorphGenome,
        metrics: BodyMetrics,
        tick: u64,
    ) {
        if self.map_elites.len() != ARCHIVE_BINS * ARCHIVE_BINS {
            self.map_elites = vec![None; ARCHIVE_BINS * ARCHIVE_BINS];
        }
        if age < 40 || metrics.mass < 0.04 {
            return;
        }

        if let Some(lin) = self.lineages.get_mut(&lineage_id) {
            lin.peak_energy = lin.peak_energy.max(energy);
            lin.last_seen_tick = tick;
        }

        // Behavior-Morphology descriptors:
        // Axis 0: Morphological elongation (0.0..0.75)
        // Axis 1: Organelle/membrane complexity & activity
        let b0 = ((metrics.elongation / 0.75) * ARCHIVE_BINS as f32)
            .clamp(0.0, (ARCHIVE_BINS - 1) as f32) as usize;
        let complexity = (metrics.membrane_mass * 4.0 + metrics.organelle_mass * 5.0).clamp(0.0, 1.0);
        let b1 = (complexity * ARCHIVE_BINS as f32).clamp(0.0, (ARCHIVE_BINS - 1) as f32) as usize;
        let idx = b0 * ARCHIVE_BINS + b1;

        let fitness = energy * 0.6 + (age as f32).sqrt() * 2.5 + generation as f32 * 4.0;
        let should_insert = match &self.map_elites[idx] {
            None => true,
            Some(existing) => fitness > existing.fitness * 1.08,
        };

        if should_insert {
            let label = classify_morphology(&morph_genome, &metrics);
            self.map_elites[idx] = Some(DiscoveryEntry {
                organism_id: org_id,
                lineage_id,
                discovered_tick: tick,
                generation,
                tribe,
                label,
                fitness,
                morph_genome,
                metrics,
            });
            self.total_discoveries += 1;
        }
    }

    pub fn active_lineage_count(&self) -> usize {
        self.lineages.values().filter(|l| !l.extinct).count()
    }

    pub fn filled_archive_cells(&self) -> usize {
        self.map_elites.iter().filter(|e| e.is_some()).count()
    }
}

fn classify_morphology(g: &MorphGenome, m: &BodyMetrics) -> String {
    let shape = if m.elongation > 0.45 {
        "Serpentiform"
    } else if g.spiral_twist.abs() > 1.2 {
        "Vortex"
    } else if g.angular_lobes >= 4.5 {
        "Stellate"
    } else if g.angular_lobes >= 2.5 {
        "Medusoid"
    } else if g.asymmetry > 0.25 {
        "Glider"
    } else {
        "Orbium"
    };
    let cortex = if m.membrane_mass > 0.04 {
        "Armored"
    } else if m.organelle_mass > 0.03 {
        "Luminous"
    } else {
        "Translucent"
    };
    format!("{cortex} {shape}")
}
