//! GENOMORPH — Hierarchical Developmental Genome & Procedural Organism Anatomy.
//!
//! Each anatomical structure (`AnatomyGene`) carries a persistent homology `gene_id`
//! inherited across generations, allowing true structural mutations (duplication,
//! deletion, innovation, specialization, symmetry-breaking) and direct parent-offspring
//! anatomical diffing.

use crate::particle::Tribe;
use rand::{rngs::StdRng, Rng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::PI;

pub const MAX_ANATOMY_GENES: usize = 14;
pub const MAX_PHENOTYPE_ORGANS: usize = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrganKind {
    /// Primary central cellular soma.
    CoreSegment,
    /// Articulated body segment extending the spine/torso.
    BodySegment,
    /// Lateral oscillating fin generating forward thrust & turn authority.
    PropulsionFin,
    /// Posterior undulating tail generating strong forward propulsion.
    UndulatingTail,
    /// Anterior predatory jaw/pincer enabling directional bite attacks.
    FeedingJaw,
    /// Translucent filter-feeding veil boosting dissolved nutrient absorption.
    FilterMembrane,
    /// Delicate sensory filament extending directional organism/food detection range.
    SensoryAntenna,
    /// Thickened defensive plate reducing predation damage at a drag/cost tradeoff.
    ArmorCarapace,
    /// Internal digestive compartment expanding stomach capacity & assimilation.
    DigestiveVacuole,
    /// Secretory gland amplifying defensive toxin or social pheromone emission.
    ToxinGland,
}

impl OrganKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::CoreSegment => "Core Soma",
            Self::BodySegment => "Body Segment",
            Self::PropulsionFin => "Propulsion Fin",
            Self::UndulatingTail => "Undulating Tail",
            Self::FeedingJaw => "Feeding Jaw",
            Self::FilterMembrane => "Filter Veil",
            Self::SensoryAntenna => "Sensory Antenna",
            Self::ArmorCarapace => "Armor Carapace",
            Self::DigestiveVacuole => "Digestive Vacuole",
            Self::ToxinGland => "Secretory Gland",
        }
    }

    pub fn code(self) -> u32 {
        match self {
            Self::CoreSegment => 0,
            Self::BodySegment => 1,
            Self::PropulsionFin => 2,
            Self::UndulatingTail => 3,
            Self::FeedingJaw => 4,
            Self::FilterMembrane => 5,
            Self::SensoryAntenna => 6,
            Self::ArmorCarapace => 7,
            Self::DigestiveVacuole => 8,
            Self::ToxinGland => 9,
        }
    }

    pub fn random_appendage(rng: &mut StdRng) -> Self {
        match rng.gen_range(0..9) {
            0 => Self::BodySegment,
            1 => Self::PropulsionFin,
            2 => Self::UndulatingTail,
            3 => Self::FeedingJaw,
            4 => Self::FilterMembrane,
            5 => Self::SensoryAntenna,
            6 => Self::ArmorCarapace,
            7 => Self::DigestiveVacuole,
            _ => Self::ToxinGland,
        }
    }
}

/// A single heritable anatomical gene with stable homology identity.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct AnatomyGene {
    /// Persistent homology ID preserved across generations.
    pub gene_id: u64,
    /// Parent gene to which this structure attaches (`None` for root CoreSegment).
    pub parent_gene_id: Option<u64>,
    pub kind: OrganKind,
    /// If true, expresses symmetrically on both left and right sides of the parent axis.
    pub bilateral: bool,
    /// Attachment angle relative to parent orientation (radians, -PI..PI).
    pub attach_angle: f32,
    /// Normalized attachment distance along parent length (0.0..1.0).
    pub attach_dist: f32,
    /// Normalized structure length (0.18..1.65).
    pub length: f32,
    /// Normalized structure width/thickness (0.08..0.85).
    pub width: f32,
    /// Natural curvature / sweep angle (-1.2..1.2).
    pub curvature: f32,
    /// Dynamic oscillation / swimming flexibility (0.0..1.0).
    pub flexibility: f32,
    /// Bioluminescent hue offset relative to organism base hue.
    pub hue_shift: f32,
    pub enabled: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifeStage {
    Embryo,
    Juvenile,
    Adult,
    Elder,
}

impl LifeStage {
    pub fn name(self) -> &'static str {
        match self {
            Self::Embryo => "Embryo",
            Self::Juvenile => "Juvenile",
            Self::Adult => "Adult",
            Self::Elder => "Elder",
        }
    }
}

/// Complete GENOMORPH developmental genome encoding anatomy, digestion, and life history.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenomorphGenome {
    pub genes: Vec<AnatomyGene>,
    pub next_gene_id: u64,
    /// Heritable dietary specialization: 0.0 = pure herbivore/filter, 1.0 = obligate carnivore.
    pub diet_carnivory: f32,
    /// Heritable digestion speed (converts stomach contents -> energy at a metabolic cost).
    pub digestion_rate: f32,
    /// Overall genetically encoded mature body scale (0.65..1.55).
    pub base_scale: f32,
    /// Energy fraction invested into offspring upon reproduction (0.35..0.58).
    pub repro_investment: f32,
    /// Maximum biological lifespan in ticks before senescence.
    pub max_age: u64,
    /// Recent structural mutation summary for phylogenetic inspection.
    pub last_mutation: String,
}

impl GenomorphGenome {
    /// Minimal founding organism for strict *Genesis Zero* experiments.
    pub fn minimal_founder(rng: &mut StdRng) -> Self {
        let genes = vec![
            AnatomyGene {
                gene_id: 1,
                parent_gene_id: None,
                kind: OrganKind::CoreSegment,
                bilateral: false,
                attach_angle: 0.0,
                attach_dist: 0.0,
                length: 0.85,
                width: 0.55,
                curvature: 0.0,
                flexibility: 0.25,
                hue_shift: 0.0,
                enabled: true,
            },
            AnatomyGene {
                gene_id: 2,
                parent_gene_id: Some(1),
                kind: OrganKind::PropulsionFin,
                bilateral: true,
                attach_angle: 2.2,
                attach_dist: 0.5,
                length: 0.55,
                width: 0.16,
                curvature: 0.25,
                flexibility: 0.75,
                hue_shift: 0.06,
                enabled: true,
            },
        ];
        Self {
            genes,
            next_gene_id: 3,
            diet_carnivory: rng.gen_range(0.08..0.28),
            digestion_rate: rng.gen_range(0.35..0.55),
            base_scale: rng.gen_range(0.85..1.05),
            repro_investment: 0.46,
            max_age: rng.gen_range(2600..4200),
            last_mutation: "Genesis founder".to_string(),
        }
    }

    /// Procedurally generated diverse founding body plan seeded by tribe and RNG.
    pub fn from_tribe(tribe: Tribe, rng: &mut StdRng) -> Self {
        let mut genes = vec![AnatomyGene {
            gene_id: 1,
            parent_gene_id: None,
            kind: OrganKind::CoreSegment,
            bilateral: false,
            attach_angle: 0.0,
            attach_dist: 0.0,
            length: rng.gen_range(0.75..1.10),
            width: rng.gen_range(0.42..0.68),
            curvature: 0.0,
            flexibility: 0.25,
            hue_shift: 0.0,
            enabled: true,
        }];
        let mut next_id = 2u64;

        let mut push_gene = |parent: u64,
                             kind: OrganKind,
                             bilateral: bool,
                             angle: f32,
                             dist: f32,
                             len: f32,
                             w: f32,
                             curv: f32,
                             flex: f32,
                             hshift: f32| {
            let id = next_id;
            next_id += 1;
            genes.push(AnatomyGene {
                gene_id: id,
                parent_gene_id: Some(parent),
                kind,
                bilateral,
                attach_angle: angle,
                attach_dist: dist,
                length: len,
                width: w,
                curvature: curv,
                flexibility: flex,
                hue_shift: hshift,
                enabled: true,
            });
            id
        };

        let diet_carnivory = match tribe {
            // Blood: Predatory segmented hunter with jaws, fins & tail
            Tribe::Blood => {
                let seg = push_gene(1, OrganKind::BodySegment, false, PI, 0.85, 0.78, 0.36, 0.1, 0.45, 0.04);
                push_gene(1, OrganKind::FeedingJaw, true, 0.38, 0.90, 0.58, 0.14, -0.35, 0.35, 0.08);
                push_gene(1, OrganKind::PropulsionFin, true, 1.85, 0.55, 0.72, 0.18, 0.35, 0.80, -0.05);
                push_gene(seg, OrganKind::UndulatingTail, false, PI, 0.95, 1.05, 0.16, 0.2, 0.95, 0.10);
                push_gene(1, OrganKind::SensoryAntenna, true, 0.65, 0.85, 0.78, 0.09, 0.2, 0.65, 0.14);
                rng.gen_range(0.65..0.92)
            }
            // Moss: Filter-feeding armored grazer with translucent veils
            Tribe::Moss => {
                push_gene(1, OrganKind::FilterMembrane, true, 1.15, 0.65, 0.92, 0.28, 0.45, 0.60, 0.08);
                push_gene(1, OrganKind::ArmorCarapace, false, 0.0, 0.25, 0.80, 0.62, 0.0, 0.05, -0.06);
                push_gene(1, OrganKind::DigestiveVacuole, false, PI, 0.45, 0.55, 0.40, 0.0, 0.15, 0.12);
                push_gene(1, OrganKind::PropulsionFin, true, 2.35, 0.75, 0.58, 0.15, 0.2, 0.70, 0.04);
                rng.gen_range(0.05..0.22)
            }
            // Deep: Medusoid swimmer with long trailing tendrils & sensory filaments
            Tribe::Deep => {
                push_gene(1, OrganKind::UndulatingTail, true, 2.65, 0.80, 1.25, 0.12, 0.35, 0.95, 0.10);
                push_gene(1, OrganKind::SensoryAntenna, true, 0.85, 0.80, 0.95, 0.09, -0.25, 0.85, 0.16);
                push_gene(1, OrganKind::FilterMembrane, true, 1.65, 0.55, 0.75, 0.24, 0.3, 0.75, -0.08);
                push_gene(1, OrganKind::ToxinGland, false, PI, 0.60, 0.45, 0.26, 0.0, 0.25, 0.22);
                rng.gen_range(0.20..0.48)
            }
            // Solar: Armored omnivore with radiating fins & vacuoles
            Tribe::Solar => {
                push_gene(1, OrganKind::PropulsionFin, true, 1.45, 0.60, 0.82, 0.20, 0.25, 0.75, 0.06);
                push_gene(1, OrganKind::PropulsionFin, true, 2.40, 0.75, 0.68, 0.17, -0.2, 0.80, 0.12);
                push_gene(1, OrganKind::ArmorCarapace, true, 0.75, 0.55, 0.58, 0.24, 0.15, 0.08, -0.05);
                push_gene(1, OrganKind::FilterMembrane, false, 0.0, 0.85, 0.65, 0.32, 0.0, 0.45, 0.15);
                rng.gen_range(0.18..0.45)
            }
            // Dream: Asymmetric spiral/glider omnivore with jaws & secretory glands
            Tribe::Dream => {
                let seg = push_gene(1, OrganKind::BodySegment, false, PI * 0.92, 0.85, 0.72, 0.32, 0.35, 0.65, 0.08);
                push_gene(1, OrganKind::PropulsionFin, true, 1.95, 0.55, 0.88, 0.17, 0.55, 0.88, 0.12);
                push_gene(1, OrganKind::FeedingJaw, false, 0.12, 0.92, 0.55, 0.16, 0.25, 0.40, -0.10);
                push_gene(seg, OrganKind::ToxinGland, false, PI, 0.80, 0.50, 0.22, -0.2, 0.50, 0.20);
                rng.gen_range(0.42..0.72)
            }
            // Static: Multi-segmented serpentine swimmer
            Tribe::Static => {
                let s1 = push_gene(1, OrganKind::BodySegment, false, PI, 0.88, 0.82, 0.34, 0.0, 0.60, 0.04);
                let s2 = push_gene(s1, OrganKind::BodySegment, false, PI, 0.88, 0.70, 0.26, 0.0, 0.78, 0.08);
                push_gene(s2, OrganKind::UndulatingTail, false, PI, 0.92, 1.02, 0.14, 0.0, 0.96, 0.12);
                push_gene(1, OrganKind::SensoryAntenna, true, 0.52, 0.88, 0.76, 0.09, 0.18, 0.75, -0.06);
                push_gene(s1, OrganKind::PropulsionFin, true, 1.90, 0.50, 0.54, 0.14, 0.25, 0.80, 0.06);
                rng.gen_range(0.30..0.65)
            }
        };

        Self {
            genes,
            next_gene_id: next_id,
            diet_carnivory,
            digestion_rate: rng.gen_range(0.35..0.68),
            base_scale: rng.gen_range(0.85..1.18),
            repro_investment: rng.gen_range(0.42..0.52),
            max_age: rng.gen_range(2800..4600),
            last_mutation: format!("{} founder", tribe.name()),
        }
    }

    /// Produce a mutated offspring genome supporting both parametric drift
    /// and true structural mutations (duplication, deletion, innovation, specialization).
    pub fn mutate(&self, rng: &mut StdRng, rate: f32) -> Self {
        let mut child = self.clone();
        let r = rate.clamp(0.01, 0.30);
        let mut events: Vec<String> = Vec::new();

        child.diet_carnivory = (child.diet_carnivory + rng.gen_range(-r..r) * 0.45).clamp(0.0, 1.0);
        child.digestion_rate = (child.digestion_rate + rng.gen_range(-r..r) * 0.35).clamp(0.15, 1.0);
        child.base_scale = (child.base_scale + rng.gen_range(-r..r) * 0.30).clamp(0.65, 1.55);
        child.repro_investment =
            (child.repro_investment + rng.gen_range(-r..r) * 0.15).clamp(0.35, 0.58);

        // 1. Parametric drift on inherited anatomical genes (preserving homology IDs!)
        for gene in &mut child.genes {
            if rng.gen_bool((r * 2.2).min(0.65) as f64) {
                gene.length = (gene.length + rng.gen_range(-r..r) * 0.65).clamp(0.20, 1.65);
                gene.width = (gene.width + rng.gen_range(-r..r) * 0.32).clamp(0.08, 0.85);
                gene.curvature = (gene.curvature + rng.gen_range(-r..r) * 0.70).clamp(-1.2, 1.2);
                gene.flexibility =
                    (gene.flexibility + rng.gen_range(-r..r) * 0.45).clamp(0.05, 1.0);
                gene.hue_shift = (gene.hue_shift + rng.gen_range(-r..r) * 0.18).clamp(-0.35, 0.35);
                if gene.parent_gene_id.is_some() {
                    gene.attach_angle =
                        (gene.attach_angle + rng.gen_range(-r..r) * 0.80).clamp(-PI, PI);
                    gene.attach_dist =
                        (gene.attach_dist + rng.gen_range(-r..r) * 0.35).clamp(0.15, 1.0);
                }
            }
            // Occasional symmetry-breaking mutation
            if gene.parent_gene_id.is_some() && rng.gen_bool((r * 0.25).min(0.08) as f64) {
                gene.bilateral = !gene.bilateral;
                events.push(format!("Symmetry flip #{}", gene.gene_id));
            }
        }

        // 2. Structural Mutation: Duplicate an existing appendage/segment
        if child.genes.len() < MAX_ANATOMY_GENES && child.genes.len() > 1 && rng.gen_bool((r * 1.1).min(0.22) as f64) {
            let src_idx = rng.gen_range(1..child.genes.len());
            let mut dup = child.genes[src_idx];
            let new_id = child.next_gene_id;
            child.next_gene_id += 1;
            dup.gene_id = new_id;
            dup.attach_angle = (dup.attach_angle + rng.gen_range(-0.55..0.55)).clamp(-PI, PI);
            dup.attach_dist = (dup.attach_dist + rng.gen_range(-0.2..0.2)).clamp(0.2, 1.0);
            dup.length = (dup.length * rng.gen_range(0.82..1.18)).clamp(0.20, 1.65);
            events.push(format!("Duplicated {} -> #{}", dup.kind.name(), new_id));
            child.genes.push(dup);
        }

        // 3. Structural Mutation: Sprout a novel anatomical organ attached to an existing segment
        if child.genes.len() < MAX_ANATOMY_GENES && rng.gen_bool((r * 1.0).min(0.20) as f64) {
            let anchor_ids: Vec<u64> = child
                .genes
                .iter()
                .filter(|g| g.enabled && matches!(g.kind, OrganKind::CoreSegment | OrganKind::BodySegment))
                .map(|g| g.gene_id)
                .collect();
            let parent_id = anchor_ids[rng.gen_range(0..anchor_ids.len())];
            let kind = OrganKind::random_appendage(rng);
            let new_id = child.next_gene_id;
            child.next_gene_id += 1;
            child.genes.push(AnatomyGene {
                gene_id: new_id,
                parent_gene_id: Some(parent_id),
                kind,
                bilateral: !matches!(kind, OrganKind::BodySegment | OrganKind::UndulatingTail)
                    && rng.gen_bool(0.70),
                attach_angle: rng.gen_range(0.3..PI),
                attach_dist: rng.gen_range(0.35..0.95),
                length: rng.gen_range(0.42..1.05),
                width: rng.gen_range(0.11..0.32),
                curvature: rng.gen_range(-0.45..0.45),
                flexibility: rng.gen_range(0.25..0.90),
                hue_shift: rng.gen_range(-0.15..0.15),
                enabled: true,
            });
            events.push(format!("Sprouted {} #{}", kind.name(), new_id));
        }

        // 4. Structural Mutation: Delete a non-root leaf gene
        if child.genes.len() > 2 && rng.gen_bool((r * 0.75).min(0.14) as f64) {
            let idx = rng.gen_range(1..child.genes.len());
            let target_id = child.genes[idx].gene_id;
            let has_children = child
                .genes
                .iter()
                .any(|g| g.parent_gene_id == Some(target_id));
            if !has_children {
                let removed = child.genes.remove(idx);
                events.push(format!("Lost {} #{}", removed.kind.name(), removed.gene_id));
            }
        }

        // 5. Structural Mutation: Specialize an existing non-root organ into a new functional role
        if child.genes.len() > 1 && rng.gen_bool((r * 0.55).min(0.10) as f64) {
            let idx = rng.gen_range(1..child.genes.len());
            let old_kind = child.genes[idx].kind;
            let new_kind = OrganKind::random_appendage(rng);
            if new_kind != old_kind {
                child.genes[idx].kind = new_kind;
                events.push(format!(
                    "Specialized #{} {}->{}",
                    child.genes[idx].gene_id,
                    old_kind.name(),
                    new_kind.name()
                ));
            }
        }

        child.last_mutation = if events.is_empty() {
            "Parametric drift".to_string()
        } else {
            events.join(", ")
        };
        child
    }

    /// Homology-matched genetic crossover aligning corresponding `gene_id`s from two parents.
    pub fn crossover_homologous(&self, other: &Self, rng: &mut StdRng) -> Self {
        let other_map: HashMap<u64, AnatomyGene> =
            other.genes.iter().map(|g| (g.gene_id, *g)).collect();
        let mut genes = Vec::with_capacity(self.genes.len());
        for g in &self.genes {
            if let Some(og) = other_map.get(&g.gene_id) {
                genes.push(if rng.gen_bool(0.5) { *g } else { *og });
            } else {
                genes.push(*g);
            }
        }
        Self {
            genes,
            next_gene_id: self.next_gene_id.max(other.next_gene_id),
            diet_carnivory: (self.diet_carnivory + other.diet_carnivory) * 0.5,
            digestion_rate: (self.digestion_rate + other.digestion_rate) * 0.5,
            base_scale: (self.base_scale + other.base_scale) * 0.5,
            repro_investment: (self.repro_investment + other.repro_investment) * 0.5,
            max_age: (self.max_age + other.max_age) / 2,
            last_mutation: "Homologous crossover".to_string(),
        }
    }

    /// Structural & parametric genetic distance for speciation analysis.
    pub fn distance(&self, other: &Self) -> f32 {
        let a_map: HashMap<u64, &AnatomyGene> = self.genes.iter().map(|g| (g.gene_id, g)).collect();
        let b_map: HashMap<u64, &AnatomyGene> =
            other.genes.iter().map(|g| (g.gene_id, g)).collect();

        let mut disjoint = 0.0f32;
        let mut param_diff = 0.0f32;
        let mut matched = 0.0f32;

        for (id, ga) in &a_map {
            if let Some(gb) = b_map.get(id) {
                matched += 1.0;
                if ga.kind != gb.kind {
                    param_diff += 1.2;
                }
                param_diff += (ga.length - gb.length).abs() + (ga.width - gb.width).abs();
            } else {
                disjoint += 1.0;
            }
        }
        for id in b_map.keys() {
            if !a_map.contains_key(id) {
                disjoint += 1.0;
            }
        }

        disjoint * 0.85
            + (param_diff / matched.max(1.0))
            + (self.diet_carnivory - other.diet_carnivory).abs() * 2.5
    }
}

/// Homology comparison between an ancestor and descendant genome.
#[derive(Clone, Debug, Default)]
pub struct AnatomyDiff {
    pub retained: Vec<(u64, OrganKind)>,
    pub modified: Vec<(u64, OrganKind, String)>,
    pub added: Vec<(u64, OrganKind)>,
    pub removed: Vec<(u64, OrganKind)>,
}

pub fn diff_genomes(ancestor: &GenomorphGenome, descendant: &GenomorphGenome) -> AnatomyDiff {
    let anc_map: HashMap<u64, &AnatomyGene> =
        ancestor.genes.iter().map(|g| (g.gene_id, g)).collect();
    let desc_map: HashMap<u64, &AnatomyGene> =
        descendant.genes.iter().map(|g| (g.gene_id, g)).collect();

    let mut diff = AnatomyDiff::default();
    for (id, dg) in &desc_map {
        if let Some(ag) = anc_map.get(id) {
            if ag.kind != dg.kind {
                diff.modified.push((
                    *id,
                    dg.kind,
                    format!("{} -> {}", ag.kind.name(), dg.kind.name()),
                ));
            } else if (ag.length - dg.length).abs() > 0.08
                || (ag.width - dg.width).abs() > 0.05
                || ag.bilateral != dg.bilateral
            {
                diff.modified.push((
                    *id,
                    dg.kind,
                    format!("len {:.2}->{:.2}", ag.length, dg.length),
                ));
            } else {
                diff.retained.push((*id, dg.kind));
            }
        } else {
            diff.added.push((*id, dg.kind));
        }
    }
    for (id, ag) in &anc_map {
        if !desc_map.contains_key(id) {
            diff.removed.push((*id, ag.kind));
        }
    }
    diff
}

/// A realized 2D anatomical organ in organism-local space (ready for physics & GPU rendering).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct OrganInstance {
    pub gene_id: u64,
    pub kind: OrganKind,
    /// Local start & end coordinates (normalized organism radii).
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub width: f32,
    pub hue_shift: f32,
}

/// Realized functional anatomy & physical capabilities derived from `GenomorphGenome` + maturity.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AnatomyPhenotype {
    pub organs: Vec<OrganInstance>,
    pub propulsion_power: f32,
    pub turn_authority: f32,
    pub bite_power: f32,
    pub bite_range: f32,
    pub filter_efficiency: f32,
    pub sensor_range: f32,
    pub armor_rating: f32,
    pub stomach_capacity: f32,
    pub toxin_capacity: f32,
    pub total_metabolic_cost: f32,
    pub drag_factor: f32,
}

impl AnatomyPhenotype {
    /// Express `GenomorphGenome` into articulated 2D organs and coherent biological stats,
    /// modulated by developmental `maturity` (0.25..1.0) and dynamic `swim_phase`.
    pub fn express(genome: &GenomorphGenome, maturity: f32, swim_phase: f32) -> Self {
        let scale = genome.base_scale * maturity.clamp(0.25, 1.0);
        let mut pheno = Self {
            organs: Vec::with_capacity(MAX_PHENOTYPE_ORGANS),
            propulsion_power: 0.25,
            turn_authority: 0.35,
            bite_power: 0.08,
            bite_range: 0.045,
            filter_efficiency: 0.35,
            sensor_range: 0.16,
            armor_rating: 0.05,
            stomach_capacity: 28.0 * scale,
            toxin_capacity: 0.25,
            total_metabolic_cost: 0.08 * scale,
            drag_factor: 0.88,
        };

        // Map gene_id -> (x0, y0, x1, y1, angle) for parent attachment lookup
        let mut anchors: HashMap<u64, (f32, f32, f32, f32, f32)> = HashMap::new();

        for gene in &genome.genes {
            if !gene.enabled || pheno.organs.len() >= MAX_PHENOTYPE_ORGANS {
                continue;
            }
            let len = gene.length * scale * 0.55;
            let width = gene.width * scale * 0.28;

            if gene.parent_gene_id.is_none() {
                let x0 = -len * 0.45;
                let x1 = len * 0.55;
                anchors.insert(gene.gene_id, (x0, 0.0, x1, 0.0, 0.0));
                pheno.organs.push(OrganInstance {
                    gene_id: gene.gene_id,
                    kind: gene.kind,
                    x0,
                    y0: 0.0,
                    x1,
                    y1: 0.0,
                    width,
                    hue_shift: gene.hue_shift,
                });
                continue;
            }

            let pid = gene.parent_gene_id.unwrap();
            let &(px0, py0, px1, py1, p_ang) = anchors.get(&pid).unwrap_or(&(0.0, 0.0, 0.2, 0.0, 0.0));
            let ax = px0 + (px1 - px0) * gene.attach_dist;
            let ay = py0 + (py1 - py0) * gene.attach_dist;

            let sides: &[f32] = if gene.bilateral { &[1.0, -1.0] } else { &[1.0] };
            for &side in sides {
                if pheno.organs.len() >= MAX_PHENOTYPE_ORGANS {
                    break;
                }
                let osc = (swim_phase + gene.gene_id as f32 * 0.9).sin()
                    * gene.flexibility
                    * 0.32
                    * side;
                let ang = p_ang + side * (gene.attach_angle + gene.curvature * 0.4) + osc;
                let ex = ax + ang.cos() * len;
                let ey = ay + ang.sin() * len;

                if side > 0.0 {
                    anchors.insert(gene.gene_id, (ax, ay, ex, ey, ang));
                }

                pheno.organs.push(OrganInstance {
                    gene_id: gene.gene_id,
                    kind: gene.kind,
                    x0: ax,
                    y0: ay,
                    x1: ex,
                    y1: ey,
                    width,
                    hue_shift: gene.hue_shift,
                });

                // Functional biological effects & honest metabolic/drag tradeoffs
                let area = len * width * 6.0;
                match gene.kind {
                    OrganKind::CoreSegment => {}
                    OrganKind::BodySegment => {
                        pheno.stomach_capacity += 12.0 * scale;
                        pheno.armor_rating += 0.06;
                        pheno.total_metabolic_cost += 0.045 * area;
                        pheno.drag_factor -= 0.012;
                    }
                    OrganKind::PropulsionFin => {
                        pheno.propulsion_power += 0.34 * len * (0.5 + gene.flexibility);
                        pheno.turn_authority += 0.28 * len;
                        pheno.total_metabolic_cost += 0.038 * area;
                    }
                    OrganKind::UndulatingTail => {
                        pheno.propulsion_power += 0.58 * len * (0.4 + gene.flexibility);
                        pheno.total_metabolic_cost += 0.048 * area;
                    }
                    OrganKind::FeedingJaw => {
                        pheno.bite_power += 0.28 * (len + width);
                        pheno.bite_range = (pheno.bite_range + len * 0.04).min(0.11);
                        pheno.total_metabolic_cost += 0.042 * area;
                    }
                    OrganKind::FilterMembrane => {
                        pheno.filter_efficiency += 0.42 * area;
                        pheno.total_metabolic_cost += 0.025 * area;
                        pheno.drag_factor -= 0.015;
                    }
                    OrganKind::SensoryAntenna => {
                        pheno.sensor_range = (pheno.sensor_range + 0.12 * len).min(0.48);
                        pheno.total_metabolic_cost += 0.032 * area;
                    }
                    OrganKind::ArmorCarapace => {
                        pheno.armor_rating = (pheno.armor_rating + 0.24 * area).min(0.78);
                        pheno.total_metabolic_cost += 0.045 * area;
                        pheno.drag_factor -= 0.025;
                    }
                    OrganKind::DigestiveVacuole => {
                        pheno.stomach_capacity += 22.0 * area;
                        pheno.total_metabolic_cost += 0.030 * area;
                    }
                    OrganKind::ToxinGland => {
                        pheno.toxin_capacity += 0.35 * area;
                        pheno.total_metabolic_cost += 0.036 * area;
                    }
                }
            }
        }

        pheno.drag_factor = pheno.drag_factor.clamp(0.76, 0.93);
        pheno
    }
}
