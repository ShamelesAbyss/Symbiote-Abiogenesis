//! Unified multiscale artificial-life engine (ConvergenceWorld).
//!
//! Integrates all 6 interacting scales:
//! 1. Primitive particle chemistry (Symbiote's Particle + RuleMatrix)
//! 2. Localized high-resolution continuous cellular bodies (LeniaBody)
//! 3. Individual biological organisms (MorphOrganism)
//! 4. Evolving species & lineages (phylogenetic tracking + heritable MorphGenome)
//! 5. Regional ecological communities (physical contact, predation, symbiosis)
//! 6. Persistent planetary-scale environmental systems (EcoField + Environment)

use crate::app::{App, Environment};
use crate::convergence::eco_field::{EcoField, ECO_HEIGHT, ECO_WIDTH};
use crate::convergence::lenia_body::{LeniaBody, LocalEnvInput, MorphGenome, MorphKernel};
use crate::particle::{Genome, Tribe};
use crate::species::{derive_archetype, Archetype};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::f32::consts::TAU;

pub const MAX_ORGANISMS: usize = 96;
pub const INITIAL_ORGANISMS: usize = 36;

#[derive(Clone, Debug)]
pub struct MorphOrganism {
    pub id: u64,
    pub parent_id: Option<u64>,
    pub lineage_id: u64,
    pub generation: u32,
    pub tribe: Tribe,
    pub archetype: Archetype,
    pub eco_genome: Genome,
    pub morph_genome: MorphGenome,
    pub body: LeniaBody,
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub energy: f32,
    pub age: u64,
    pub offspring_count: u32,
    pub alive: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ConvergenceStats {
    pub tick: u64,
    pub organism_count: usize,
    pub active_lineages: usize,
    pub mean_mass: f32,
    pub mean_membrane: f32,
    pub mean_activity: f32,
    pub mean_nutrient: f32,
    pub mean_mu: f32,
    pub mean_sigma: f32,
    pub mean_asymmetry: f32,
    pub total_births: u64,
    pub total_deaths: u64,
    pub total_predations: u64,
    pub max_generation: u32,
    pub gpu_active: bool,
}

pub struct ConvergenceWorld {
    pub symbiote: App,
    pub eco_field: EcoField,
    pub organisms: Vec<MorphOrganism>,
    pub rng: StdRng,
    pub next_org_id: u64,
    pub tick: u64,
    pub total_births: u64,
    pub total_deaths: u64,
    pub total_predations: u64,
    pub use_gpu: bool,
    #[cfg(feature = "gpu")]
    pub gpu_compute: Option<crate::convergence::gpu_compute::ConvergenceGpu>,
}

impl ConvergenceWorld {
    pub fn new(seed: u64, prefer_gpu: bool) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let symbiote = App::new();
        let eco_field = EcoField::new(ECO_WIDTH, ECO_HEIGHT);

        let mut organisms = Vec::with_capacity(MAX_ORGANISMS);
        let mut next_org_id = 1u64;

        for i in 0..INITIAL_ORGANISMS {
            let tribe = Tribe::from_index(i);
            let eco_genome = if let Some(p) = symbiote.particles.get(i * 4) {
                p.genome
            } else {
                random_eco_genome(&mut rng)
            };
            let morph_genome = MorphGenome::from_ecological(eco_genome, tribe, &mut rng);
            let heading = rng.gen_range(0.0..TAU);
            let body = LeniaBody::new(&morph_genome, heading, &mut rng);
            let archetype = derive_archetype(eco_genome, crate::particle::RareTrait::None, 8);

            let x = rng.gen_range(-0.85..0.85);
            let y = rng.gen_range(-0.85..0.85);
            let id = next_org_id;
            next_org_id += 1;

            organisms.push(MorphOrganism {
                id,
                parent_id: None,
                lineage_id: id,
                generation: 0,
                tribe,
                archetype,
                eco_genome,
                morph_genome,
                body,
                x,
                y,
                vx: 0.0,
                vy: 0.0,
                energy: 55.0 + rng.gen_range(0.0..20.0),
                age: 0,
                offspring_count: 0,
                alive: true,
            });
        }

        #[cfg(feature = "gpu")]
        let gpu_compute = if prefer_gpu {
            crate::convergence::gpu_compute::ConvergenceGpu::try_new(
                ECO_WIDTH as u32,
                ECO_HEIGHT as u32,
                MAX_ORGANISMS as u32,
            )
            .ok()
        } else {
            None
        };

        #[cfg(feature = "gpu")]
        let use_gpu = gpu_compute.is_some();
        #[cfg(not(feature = "gpu"))]
        let use_gpu = {
            let _ = prefer_gpu;
            false
        };

        Self {
            symbiote,
            eco_field,
            organisms,
            rng,
            next_org_id,
            tick: 0,
            total_births: 0,
            total_deaths: 0,
            total_predations: 0,
            use_gpu,
            #[cfg(feature = "gpu")]
            gpu_compute,
        }
    }

    /// Advance the unified multiscale ecosystem by one tick.
    pub fn step(&mut self) {
        self.tick += 1;

        // Scale 1 & 6: Step Symbiote's underlying particle chemistry & ecology every 2 ticks
        if self.tick % 2 == 0 {
            self.symbiote.step();
            self.eco_field
                .couple_symbiote_ecology(&self.symbiote.ecology, &self.symbiote.substrate);
        }

        // Scale 6: Step continuous reaction-diffusion ecological field (GPU or Rayon CPU)
        let env = self.symbiote.environment;
        let mut stepped_field_gpu = false;
        #[cfg(feature = "gpu")]
        if self.use_gpu {
            if let Some(gpu) = &self.gpu_compute {
                if gpu.step_eco_field(&mut self.eco_field, env).is_ok() {
                    stepped_field_gpu = true;
                }
            }
        }
        if !stepped_field_gpu {
            self.eco_field.step_cpu(env);
        }

        // Scale 2: Gather local environmental inputs for each organism
        let env_inputs: Vec<LocalEnvInput> = self
            .organisms
            .iter()
            .map(|org| {
                let s = self.eco_field.sample_world(org.x, org.y);
                LocalEnvInput {
                    nutrient: s.nutrient,
                    pheromone: s.pheromone,
                    toxin: s.toxin,
                    nutrient_grad_x: s.nutrient_grad_x,
                    nutrient_grad_y: s.nutrient_grad_y,
                    mutagen: if env == Environment::Storm { 0.35 } else { 0.05 },
                }
            })
            .collect();

        // Step all localized continuous Lenia bodies in parallel (Rayon CPU)
        self.organisms
            .par_iter_mut()
            .zip(env_inputs.par_iter())
            .for_each(|(org, &local_env)| {
                if !org.alive {
                    return;
                }
                org.body.step_cpu(&org.morph_genome, local_env);
            });

        // Scale 3 & 5: Bidirectional coupling — locomotion, metabolism, resource exchange,
        // physical contact interactions, predation, reproduction, and death.
        let count = self.organisms.len();
        let positions: Vec<(f32, f32, Tribe, f32, f32)> = self
            .organisms
            .iter()
            .map(|o| (o.x, o.y, o.tribe, o.body.metrics.mass, o.body.metrics.membrane_mass))
            .collect();

        let mut new_offspring: Vec<MorphOrganism> = Vec::new();
        let mut deposits: Vec<(f32, f32, f32, f32, f32, f32)> = Vec::with_capacity(count);

        for i in 0..count {
            if !self.organisms[i].alive {
                continue;
            }
            self.organisms[i].age += 1;

            let m = self.organisms[i].body.metrics;
            let local = env_inputs[i];

            // Emergent locomotion: continuous body center-of-mass drift + chemotaxis
            let swim_scale = 0.0085 * (0.4 + self.organisms[i].morph_genome.asymmetry * 2.2);
            let chemo_fx = local.nutrient_grad_x * 0.018 * self.organisms[i].morph_genome.nutrient_sensitivity;
            let chemo_fy = local.nutrient_grad_y * 0.018 * self.organisms[i].morph_genome.nutrient_sensitivity;

            let mut fx = m.com_dx * swim_scale + chemo_fx;
            let mut fy = m.com_dy * swim_scale + chemo_fy;

            // Physical contact interactions with nearby organisms (RuleMatrix + predation/symbiosis)
            let my_x = positions[i].0;
            let my_y = positions[i].1;
            let my_tribe = positions[i].2;
            let my_radius = 0.032 + m.radius_norm * 0.045;

            for j in 0..count {
                if i == j {
                    continue;
                }
                let dx = positions[j].0 - my_x;
                let dy = positions[j].1 - my_y;
                let d2 = dx * dx + dy * dy;
                if d2 < 1e-6 || d2 > 0.09 {
                    continue;
                }
                let d = d2.sqrt();
                let affinity = self.symbiote.rules[my_tribe.index()][positions[j].2.index()];
                let contact_r = my_radius + 0.032 + positions[j].3 * 0.04;

                if d < contact_r {
                    // Soft physical membrane collision / repulsion
                    let push = (contact_r - d) * 0.14;
                    fx -= (dx / d) * push;
                    fy -= (dy / d) * push;

                    // Predation if strongly antagonistic & larger mass
                    if affinity < -0.35 && m.mass > positions[j].3 * 1.15 {
                        let bite = (0.08 * (1.0 - positions[j].4 * 0.5)).clamp(0.02, 0.12);
                        self.organisms[i].energy += bite * 45.0;
                        self.total_predations += 1;
                        deposits.push((my_x, my_y, 0.0, 0.0, 0.12, 0.0));
                    } else if affinity > 0.35 {
                        // Symbiotic energy sharing
                        self.organisms[i].energy += 0.15;
                    }
                } else {
                    // Chemotactic social attraction/repulsion
                    let pull = affinity * 0.00045 / d.max(0.04);
                    fx += (dx / d) * pull;
                    fy += (dy / d) * pull;
                }
            }

            let org = &mut self.organisms[i];
            org.vx = (org.vx + fx) * 0.88;
            org.vy = (org.vy + fy) * 0.88;
            org.x += org.vx;
            org.y += org.vy;

            // Soft toroidal-like or elastic world containment
            if org.x < -0.95 {
                org.x = -0.95;
                org.vx = org.vx.abs() * 0.5;
            } else if org.x > 0.95 {
                org.x = 0.95;
                org.vx = -org.vx.abs() * 0.5;
            }
            if org.y < -0.95 {
                org.y = -0.95;
                org.vy = org.vy.abs() * 0.5;
            } else if org.y > 0.95 {
                org.y = 0.95;
                org.vy = -org.vy.abs() * 0.5;
            }

            // Bidirectional metabolism: absorb nutrients proportional to membrane surface,
            // pay metabolic cost proportional to cytoplasm mass & environment hunger.
            let absorption = local.nutrient
                * (0.45 + m.membrane_mass * 2.2)
                * org.morph_genome.nutrient_sensitivity;
            let cost = (0.14 + m.mass * 0.85 + org.eco_genome.metabolism * 2.0) * env.hunger_mult();
            org.energy = (org.energy + absorption * 1.45 - cost - local.toxin * 0.65).clamp(0.0, 140.0);

            // Deposit metabolic footprint back into EcoField:
            // consume nutrient, secrete pheromone if healthy, secrete toxin if stressed
            let d_nutr = -(absorption * 0.045).min(0.08);
            let d_pher = if org.energy > 45.0 { 0.025 } else { 0.0 };
            let d_tox = if org.energy < 18.0 { 0.015 } else { 0.0 };
            deposits.push((org.x, org.y, d_nutr, d_pher, d_tox, 0.0));

            // Death condition: energy depleted or cellular body collapsed
            if org.energy <= 1.0 || m.mass < 0.012 {
                org.alive = false;
                self.total_deaths += 1;
                // Dead organism returns its remaining organic mass to the nutrient field!
                let recycled = (0.25 + m.mass * 1.8).clamp(0.15, 0.85);
                deposits.push((org.x, org.y, recycled, 0.0, 0.04, 0.0));
                continue;
            }

            // Mitotic reproduction: thriving organism with sufficient mass & energy divides
            if org.energy > 86.0
                && m.mass > 0.08
                && count + new_offspring.len() < MAX_ORGANISMS
            {
                let mut_rate = if env == Environment::Storm { 0.09 } else { 0.045 };
                let child_morph = org.morph_genome.mutate(&mut self.rng, mut_rate);
                let mut child_eco = org.eco_genome;
                child_eco.metabolism =
                    (child_eco.metabolism + self.rng.gen_range(-0.004..0.004)).clamp(0.01, 0.12);
                child_eco.membrane =
                    (child_eco.membrane + self.rng.gen_range(-0.05..0.05)).clamp(0.1, 1.4);

                let daughter_body = org.body.divide(&child_morph, &mut self.rng);
                org.energy *= 0.48;
                org.offspring_count += 1;

                let angle = self.rng.gen_range(0.0..TAU);
                let child_id = self.next_org_id;
                self.next_org_id += 1;
                self.total_births += 1;

                new_offspring.push(MorphOrganism {
                    id: child_id,
                    parent_id: Some(org.id),
                    lineage_id: org.lineage_id,
                    generation: org.generation + 1,
                    tribe: org.tribe,
                    archetype: derive_archetype(child_eco, crate::particle::RareTrait::None, 8),
                    eco_genome: child_eco,
                    morph_genome: child_morph,
                    body: daughter_body,
                    x: (org.x + angle.cos() * 0.05).clamp(-0.92, 0.92),
                    y: (org.y + angle.sin() * 0.05).clamp(-0.92, 0.92),
                    vx: angle.cos() * 0.004,
                    vy: angle.sin() * 0.004,
                    energy: org.energy,
                    age: 0,
                    offspring_count: 0,
                    alive: true,
                });
            }
        }

        // Apply all ecological field deposits
        for (x, y, dn, dp, dt, dr) in deposits {
            self.eco_field.deposit_world(x, y, dn, dp, dt, dr);
        }

        // Prune dead organisms and add newborn daughters
        self.organisms.retain(|o| o.alive);
        self.organisms.extend(new_offspring);

        // Abiogenesis floor: if ecosystem collapses below 8 organisms, seed a new protocell
        // from a nutrient-rich location so long-duration runs can recover organically.
        if self.organisms.len() < 8 && self.tick % 20 == 0 {
            let tribe = Tribe::from_index(self.rng.gen_range(0..6));
            let eco_genome = random_eco_genome(&mut self.rng);
            let morph_genome = MorphGenome::from_ecological(eco_genome, tribe, &mut self.rng);
            let heading = self.rng.gen_range(0.0..TAU);
            let body = LeniaBody::new(&morph_genome, heading, &mut self.rng);
            let id = self.next_org_id;
            self.next_org_id += 1;
            self.organisms.push(MorphOrganism {
                id,
                parent_id: None,
                lineage_id: id,
                generation: 0,
                tribe,
                archetype: derive_archetype(eco_genome, crate::particle::RareTrait::None, 6),
                eco_genome,
                morph_genome,
                body,
                x: self.rng.gen_range(-0.8..0.8),
                y: self.rng.gen_range(-0.8..0.8),
                vx: 0.0,
                vy: 0.0,
                energy: 52.0,
                age: 0,
                offspring_count: 0,
                alive: true,
            });
        }
    }

    pub fn stats(&self) -> ConvergenceStats {
        let n = self.organisms.len();
        let inv = if n > 0 { 1.0 / n as f32 } else { 0.0 };

        let mut lineages = std::collections::HashSet::new();
        let mut mass = 0.0;
        let mut mem = 0.0;
        let mut act = 0.0;
        let mut mu = 0.0;
        let mut sigma = 0.0;
        let mut asym = 0.0;
        let mut max_gen = 0;

        for o in &self.organisms {
            lineages.insert(o.lineage_id);
            mass += o.body.metrics.mass;
            mem += o.body.metrics.membrane_mass;
            act += o.body.metrics.activity;
            mu += o.morph_genome.mu;
            sigma += o.morph_genome.sigma;
            asym += o.morph_genome.asymmetry;
            max_gen = max_gen.max(o.generation);
        }

        ConvergenceStats {
            tick: self.tick,
            organism_count: n,
            active_lineages: lineages.len(),
            mean_mass: mass * inv,
            mean_membrane: mem * inv,
            mean_activity: act * inv,
            mean_nutrient: self.eco_field.mean_nutrient(),
            mean_mu: mu * inv,
            mean_sigma: sigma * inv,
            mean_asymmetry: asym * inv,
            total_births: self.total_births,
            total_deaths: self.total_deaths,
            total_predations: self.total_predations,
            max_generation: max_gen,
            gpu_active: self.use_gpu,
        }
    }

    /// Rebuild a kernel if an organism's heading changed significantly.
    #[allow(dead_code)]
    pub fn refresh_kernel(org: &mut MorphOrganism) {
        org.body.kernel = MorphKernel::build(&org.morph_genome, org.body.heading);
    }
}

fn random_eco_genome(rng: &mut StdRng) -> Genome {
    Genome {
        perception: rng.gen_range(0.14..0.34),
        hunger: rng.gen_range(0.012..0.042),
        bonding: rng.gen_range(0.45..1.15),
        volatility: rng.gen_range(0.75..1.45),
        orbit: rng.gen_range(0.25..0.85),
        membrane: rng.gen_range(0.35..1.10),
        metabolism: rng.gen_range(0.02..0.065),
        fertility: rng.gen_range(0.45..0.95),
    }
}
