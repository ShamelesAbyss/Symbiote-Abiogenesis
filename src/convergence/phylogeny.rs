//! Evolutionary Tree of Life, Individual Ancestry Records & MAP-Elites Discovery Archive.
//!
//! Preserves individual organism snapshots (`OrganismRecord`) so the Research Laboratory
//! can perform direct parent-offspring and ancestor-descendant anatomical comparisons
//! (`diff_genomes`) and resurrect archived MAP-Elites specimens as explicit user interventions.

use crate::convergence::genomorph::{GenomorphGenome, OrganKind};
use crate::convergence::lenia_body::{BodyMetrics, MorphGenome};
use crate::convergence::neural_graph::NeuralGraph;
use crate::particle::Tribe;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const ARCHIVE_BINS: usize = 6;
const MAX_HISTORY_RECORDS: usize = 1024;

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

/// Historical snapshot of an individual organism for generational anatomical comparisons.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrganismRecord {
    pub id: u64,
    pub parent_id: Option<u64>,
    pub lineage_id: u64,
    pub generation: u32,
    pub birth_tick: u64,
    pub death_tick: Option<u64>,
    pub genomorph: GenomorphGenome,
    pub hidden_neurons: usize,
    pub active_synapses: usize,
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
    pub genomorph: GenomorphGenome,
    pub brain: NeuralGraph,
    pub metrics: BodyMetrics,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PhylogenyArchive {
    pub lineages: HashMap<u64, LineageRecord>,
    /// Individual organism ancestry records indexed by `organism_id`.
    pub history: HashMap<u64, OrganismRecord>,
    /// 6x6 MAP-Elites grid: [anatomical_complexity_bin][diet/locomotion_bin]
    pub map_elites: Vec<Option<DiscoveryEntry>>,
    pub total_discoveries: u64,
    pub extinction_events: u64,
}

impl PhylogenyArchive {
    pub fn new() -> Self {
        Self {
            lineages: HashMap::new(),
            history: HashMap::new(),
            map_elites: vec![None; ARCHIVE_BINS * ARCHIVE_BINS],
            total_discoveries: 0,
            extinction_events: 0,
        }
    }

    pub fn record_birth(
        &mut self,
        org_id: u64,
        parent_id: Option<u64>,
        lineage_id: u64,
        tribe: Tribe,
        generation: u32,
        tick: u64,
        genomorph: &GenomorphGenome,
        brain: &NeuralGraph,
    ) {
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

        // Keep founders and recent organisms bounded for long-duration runs
        if self.history.len() >= MAX_HISTORY_RECORDS {
            if let Some((&evict_id, _)) = self
                .history
                .iter()
                .filter(|(_, r)| r.parent_id.is_some() && r.death_tick.is_some())
                .min_by_key(|(_, r)| r.birth_tick)
            {
                self.history.remove(&evict_id);
            }
        }
        self.history.insert(
            org_id,
            OrganismRecord {
                id: org_id,
                parent_id,
                lineage_id,
                generation,
                birth_tick: tick,
                death_tick: None,
                genomorph: genomorph.clone(),
                hidden_neurons: brain.hidden_count(),
                active_synapses: brain.active_connection_count(),
            },
        );
    }

    pub fn record_death(&mut self, org_id: u64, lineage_id: u64, tick: u64) {
        if let Some(rec) = self.history.get_mut(&org_id) {
            rec.death_tick = Some(tick);
        }
        if let Some(entry) = self.lineages.get_mut(&lineage_id) {
            entry.living_members = entry.living_members.saturating_sub(1);
            entry.last_seen_tick = tick;
            if entry.living_members == 0 && !entry.extinct {
                entry.extinct = true;
                self.extinction_events += 1;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn consider_discovery(
        &mut self,
        org_id: u64,
        lineage_id: u64,
        generation: u32,
        tribe: Tribe,
        energy: f32,
        age: u64,
        morph_genome: MorphGenome,
        genomorph: &GenomorphGenome,
        brain: &NeuralGraph,
        metrics: BodyMetrics,
        tick: u64,
    ) {
        if self.map_elites.len() != ARCHIVE_BINS * ARCHIVE_BINS {
            self.map_elites = vec![None; ARCHIVE_BINS * ARCHIVE_BINS];
        }
        if age < 35 || metrics.mass < 0.03 {
            return;
        }

        if let Some(lin) = self.lineages.get_mut(&lineage_id) {
            lin.peak_energy = lin.peak_energy.max(energy);
            lin.last_seen_tick = tick;
        }

        // MAP-Elites Niches:
        // Axis 0: Anatomical gene complexity (2..14 genes)
        // Axis 1: Dietary specialization (0.0 herbivore .. 1.0 carnivore)
        let b0 = (((genomorph.genes.len().saturating_sub(2)) as f32 / 10.0) * ARCHIVE_BINS as f32)
            .clamp(0.0, (ARCHIVE_BINS - 1) as f32) as usize;
        let b1 = (genomorph.diet_carnivory * ARCHIVE_BINS as f32)
            .clamp(0.0, (ARCHIVE_BINS - 1) as f32) as usize;
        let idx = b0 * ARCHIVE_BINS + b1;

        let fitness = energy * 0.55
            + (age as f32).sqrt() * 2.4
            + generation as f32 * 5.0
            + brain.active_connection_count() as f32 * 0.4;
        let should_insert = match &self.map_elites[idx] {
            None => true,
            Some(existing) => fitness > existing.fitness * 1.08,
        };

        if should_insert {
            let label = classify_genomorph(genomorph);
            self.map_elites[idx] = Some(DiscoveryEntry {
                organism_id: org_id,
                lineage_id,
                discovered_tick: tick,
                generation,
                tribe,
                label,
                fitness,
                morph_genome,
                genomorph: genomorph.clone(),
                brain: brain.clone(),
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

pub fn classify_genomorph(g: &GenomorphGenome) -> String {
    let has_jaws = g.genes.iter().any(|x| x.enabled && x.kind == OrganKind::FeedingJaw);
    let has_armor = g.genes.iter().any(|x| x.enabled && x.kind == OrganKind::ArmorCarapace);
    let has_tail = g.genes.iter().any(|x| x.enabled && x.kind == OrganKind::UndulatingTail);
    let segs = g.genes.iter().filter(|x| x.enabled && x.kind == OrganKind::BodySegment).count();

    let diet = if g.diet_carnivory > 0.62 {
        "Predatory"
    } else if g.diet_carnivory < 0.28 {
        "Filter"
    } else {
        "Omnivorous"
    };
    let form = if segs >= 2 {
        "Serpent"
    } else if has_jaws && has_tail {
        "Hunter"
    } else if has_armor {
        "Carapace"
    } else if has_tail {
        "Swimmer"
    } else {
        "Drifter"
    };
    format!("{diet} {form} ({}g)", g.genes.len())
}
