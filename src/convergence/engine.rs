//! Unified multiscale artificial-life engine (ConvergenceWorld).
//!
//! Integrates all 6 interacting scales:
//! 1. Primitive particle chemistry (Symbiote's Particle + RuleMatrix)
//! 2. Localized high-resolution 3-channel continuous cellular bodies (LeniaBody)
//! 3. Individual biological organisms with heritable NeuralBrain controllers
//! 4. Evolving species, lineages, and MAP-Elites discovery archive
//! 5. Regional ecological communities via O(N) spatial hashing (predation, symbiosis)
//! 6. Persistent planetary-scale environmental systems (EcoField + UniverseConfig)

use crate::app::{App, Environment};
use crate::convergence::eco_field::{EcoField, ECO_HEIGHT, ECO_WIDTH};
use crate::convergence::lenia_body::{LeniaBody, LocalEnvInput, MorphGenome, NeuralBrain};
use crate::convergence::phylogeny::PhylogenyArchive;
use crate::convergence::universes::{UniverseConfig, UniversePreset};
use crate::particle::{Genome, Tribe};
use crate::species::{derive_archetype, Archetype};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::TAU;

pub const MAX_ORGANISMS_BUFFER: usize = 320;
const HASH_CELL_SIZE: f32 = 0.16;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MorphOrganism {
    pub id: u64,
    pub parent_id: Option<u64>,
    pub lineage_id: u64,
    pub generation: u32,
    pub tribe: Tribe,
    pub archetype: Archetype,
    pub eco_genome: Genome,
    pub morph_genome: MorphGenome,
    pub brain: NeuralBrain,
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
    pub archive_discoveries: usize,
    pub extinction_events: u64,
    pub mean_mass: f32,
    pub mean_membrane: f32,
    pub mean_organelle: f32,
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
    pub seed: u64,
    pub universe: UniverseConfig,
    pub symbiote: App,
    pub eco_field: EcoField,
    pub organisms: Vec<MorphOrganism>,
    pub phylogeny: PhylogenyArchive,
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
        Self::from_universe(seed, UniversePreset::PrimordialOcean.config(), prefer_gpu)
    }

    pub fn from_universe(seed: u64, universe: UniverseConfig, prefer_gpu: bool) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let symbiote = App::new();
        let eco_field = EcoField::new(ECO_WIDTH, ECO_HEIGHT);
        let mut phylogeny = PhylogenyArchive::new();

        let cap = universe.max_organisms.min(MAX_ORGANISMS_BUFFER);
        let mut organisms = Vec::with_capacity(cap);
        let mut next_org_id = 1u64;

        for i in 0..universe.initial_organisms.min(cap) {
            let tribe = Tribe::from_index(i);
            let eco_genome = if let Some(p) = symbiote.particles.get(i * 4) {
                p.genome
            } else {
                random_eco_genome(&mut rng)
            };
            let mut morph_genome = MorphGenome::from_ecological(eco_genome, tribe, &mut rng);
            morph_genome.spiral_twist =
                (morph_genome.spiral_twist + rng.gen_range(-1.0..1.0) * universe.anomaly_twist_boost)
                    .clamp(-2.8, 2.8);
            let brain = NeuralBrain::random(&mut rng);
            let heading = rng.gen_range(0.0..TAU);
            let body = LeniaBody::new(&morph_genome, heading, &mut rng);
            let archetype = derive_archetype(eco_genome, crate::particle::RareTrait::None, 8);

            let x = rng.gen_range(-0.85..0.85);
            let y = rng.gen_range(-0.85..0.85);
            let id = next_org_id;
            next_org_id += 1;
            phylogeny.record_birth(id, tribe, 0, 0);

            organisms.push(MorphOrganism {
                id,
                parent_id: None,
                lineage_id: id,
                generation: 0,
                tribe,
                archetype,
                eco_genome,
                morph_genome,
                brain,
                body,
                x,
                y,
                vx: 0.0,
                vy: 0.0,
                energy: 58.0 + rng.gen_range(0.0..20.0),
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
                MAX_ORGANISMS_BUFFER as u32,
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
            seed,
            universe,
            symbiote,
            eco_field,
            organisms,
            phylogeny,
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

    pub fn step(&mut self) {
        self.tick += 1;

        // Periodic catastrophe for Extinction Protocol universe
        if self.universe.catastrophe_interval > 0
            && self.tick % self.universe.catastrophe_interval == 0
        {
            self.trigger_catastrophe();
        }

        // Scale 1 & 6: Step Symbiote's underlying particle chemistry & ecology every 2 ticks
        if self.tick % 2 == 0 {
            self.symbiote.step();
            self.eco_field
                .couple_symbiote_ecology(&self.symbiote.ecology, &self.symbiote.substrate);
        }

        // Scale 6: Step continuous reaction-diffusion EcoField (GPU or Rayon CPU)
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
                    nutrient: s.nutrient * self.universe.nutrient_regen_mult.clamp(0.4, 1.8),
                    pheromone: s.pheromone,
                    toxin: s.toxin,
                    nutrient_grad_x: s.nutrient_grad_x,
                    nutrient_grad_y: s.nutrient_grad_y,
                    mutagen: if env == Environment::Storm { 0.35 } else { 0.05 },
                }
            })
            .collect();

        // Step all 3-channel Lenia bodies + NeuralBrain controllers in parallel (Rayon)
        self.organisms
            .par_iter_mut()
            .zip(env_inputs.par_iter())
            .for_each(|(org, &local_env)| {
                if !org.alive {
                    return;
                }
                org.body.step_cpu(&org.morph_genome, local_env);
                let brain_in = [
                    local_env.nutrient,
                    local_env.nutrient_grad_x * 6.0,
                    local_env.nutrient_grad_y * 6.0,
                    local_env.toxin,
                    local_env.pheromone,
                    (org.energy / 100.0).clamp(0.0, 1.5),
                ];
                org.brain.step(brain_in);
            });

        // Build O(N) Spatial Hash Grid for physical contact, predation & symbiosis
        let count = self.organisms.len();
        let mut grid: HashMap<(i32, i32), Vec<usize>> = HashMap::with_capacity(count);
        for (i, org) in self.organisms.iter().enumerate() {
            let cell = hash_cell(org.x, org.y);
            grid.entry(cell).or_default().push(i);
        }

        let snap: Vec<(f32, f32, Tribe, f32, f32, f32)> = self
            .organisms
            .iter()
            .map(|o| {
                (
                    o.x,
                    o.y,
                    o.tribe,
                    o.body.metrics.mass,
                    o.body.metrics.membrane_mass,
                    o.energy,
                )
            })
            .collect();

        // Coherent resource accounting accumulators:
        // damage_to[j] = physical cellular damage fraction inflicted on organism j
        // energy_delta[i] = real transferred energy (predation or zero-sum symbiosis)
        let mut damage_to = vec![0.0f32; count];
        let mut energy_delta = vec![0.0f32; count];
        let mut deposits: Vec<(f32, f32, f32, f32, f32, f32)> = Vec::with_capacity(count * 2);
        let mut new_offspring: Vec<MorphOrganism> = Vec::new();
        let mut deaths_lineage: Vec<u64> = Vec::new();
        let max_orgs = self.universe.max_organisms.min(MAX_ORGANISMS_BUFFER);

        for i in 0..count {
            if !self.organisms[i].alive {
                continue;
            }
            self.organisms[i].age += 1;

            let m = self.organisms[i].body.metrics;
            let local = env_inputs[i];
            let brain_out = self.organisms[i].brain.last_out;

            // Emergent locomotion: continuous body COM drift + neural thrust + organelle motility
            let motility = 0.6 + m.organelle_mass * 3.5;
            let swim_scale =
                0.0085 * (0.4 + self.organisms[i].morph_genome.asymmetry * 2.0) * motility;
            let neural_fx = brain_out[0] * 0.0042 * motility;
            let neural_fy = brain_out[1] * 0.0042 * motility;

            let mut fx = m.com_dx * swim_scale + neural_fx;
            let mut fy = m.com_dy * swim_scale + neural_fy;

            let (my_x, my_y, my_tribe, my_mass, _, my_energy) = snap[i];
            let my_radius = 0.030 + m.radius_norm * 0.042;
            let (cx, cy) = hash_cell(my_x, my_y);

            for oy in -1..=1 {
                for ox in -1..=1 {
                    let Some(bucket) = grid.get(&(cx + ox, cy + oy)) else {
                        continue;
                    };
                    for &j in bucket {
                        if i == j {
                            continue;
                        }
                        let (ox_pos, oy_pos, o_tribe, o_mass, o_mem, o_energy) = snap[j];
                        let dx = toroidal_delta(ox_pos - my_x);
                        let dy = toroidal_delta(oy_pos - my_y);
                        let d2 = dx * dx + dy * dy;
                        if d2 < 1e-6 || d2 > 0.0625 {
                            continue;
                        }
                        let d = d2.sqrt();
                        let affinity = self.symbiote.rules[my_tribe.index()][o_tribe.index()];
                        let contact_r = my_radius + 0.030 + o_mass * 0.04;

                        if d < contact_r {
                            let push = (contact_r - d) * 0.14;
                            fx -= (dx / d) * push;
                            fy -= (dy / d) * push;

                            // Genuine Predation: inflicts real cellular damage on prey j,
                            // transferring energy proportional to actual damaged biomass.
                            if affinity < -0.30 && my_mass > o_mass * 1.12 {
                                let bite_frac =
                                    (0.14 * (1.0 - o_mem * 0.55)).clamp(0.03, 0.18);
                                damage_to[j] = (damage_to[j] + bite_frac).min(0.65);
                                let stolen = (o_mass * bite_frac * 95.0).min(o_energy * 0.35);
                                energy_delta[i] += stolen;
                                energy_delta[j] -= stolen;
                                self.total_predations += 1;
                                deposits.push((ox_pos, oy_pos, 0.0, 0.0, 0.10, 0.0));
                            } else if affinity > 0.35 && my_energy > o_energy + 8.0 {
                                // Genuine Symbiosis: zero-sum energy sharing from richer to poorer partner
                                let transfer = ((my_energy - o_energy) * 0.04).min(1.2);
                                energy_delta[i] -= transfer;
                                energy_delta[j] += transfer;
                            }
                        } else {
                            let pull = affinity * 0.00040 / d.max(0.04);
                            fx += (dx / d) * pull;
                            fy += (dy / d) * pull;
                        }
                    }
                }
            }

            let org = &mut self.organisms[i];
            org.vx = (org.vx + fx) * 0.88;
            org.vy = (org.vy + fy) * 0.88;
            // Seamless toroidal wrapping eliminates boundary congregation completely!
            org.x = wrap_world_coord(org.x + org.vx);
            org.y = wrap_world_coord(org.y + org.vy);

            // Smoothly orient organism heading toward actual swimming velocity
            let speed = (org.vx * org.vx + org.vy * org.vy).sqrt();
            if speed > 0.0008 {
                let target_h = org.vy.atan2(org.vx);
                let mut diff = (target_h - org.body.heading).rem_euclid(TAU);
                if diff > std::f32::consts::PI {
                    diff -= TAU;
                }
                org.body.heading = (org.body.heading + diff * 0.08).rem_euclid(TAU);
            }

            // Coherent metabolic accounting:
            // - Nutrient absorption scales with membrane surface area
            // - Maintenance costs scale with cytoplasm mass, membrane mass, and kinetic speed^2
            let speed2 = org.vx * org.vx + org.vy * org.vy;
            let absorption = local.nutrient
                * (0.42 + m.membrane_mass * 2.4)
                * org.morph_genome.nutrient_sensitivity;
            let maintenance = (0.14
                + m.mass * 0.95
                + m.membrane_mass * 0.45
                + speed2 * 120.0
                + org.eco_genome.metabolism * 2.0)
                * env.hunger_mult()
                * self.universe.metabolic_cost_mult;

            org.energy = (org.energy + absorption * 1.40 - maintenance - local.toxin * 0.65)
                .clamp(0.0, 150.0);

            // Neural secretion + metabolic footprint into EcoField
            let d_nutr = -(absorption * 0.048).min(0.09);
            let d_pher = if brain_out[2] > 0.15 && org.energy > 40.0 {
                0.03 * brain_out[2]
            } else {
                0.0
            };
            let d_tox = if brain_out[2] < -0.25 {
                0.025 * (-brain_out[2])
            } else {
                0.0
            };
            deposits.push((org.x, org.y, d_nutr, d_pher, d_tox, 0.0));
        }

        // Apply deferred physical predation damage, energy transfers, discovery checks, death & mitosis
        for i in 0..count {
            if !self.organisms[i].alive {
                continue;
            }
            if damage_to[i] > 0.0 {
                self.organisms[i].body.apply_damage(damage_to[i]);
            }
            self.organisms[i].energy = (self.organisms[i].energy + energy_delta[i]).clamp(0.0, 150.0);

            let m = self.organisms[i].body.metrics;

            if self.tick % 20 == 0 {
                let o = &self.organisms[i];
                self.phylogeny.consider_discovery(
                    o.id,
                    o.lineage_id,
                    o.generation,
                    o.tribe,
                    o.energy,
                    o.age,
                    o.morph_genome,
                    m,
                    self.tick,
                );
            }

            // Death & organic decomposition back into EcoField
            if self.organisms[i].energy <= 1.0 || m.mass < 0.012 {
                self.organisms[i].alive = false;
                self.total_deaths += 1;
                deaths_lineage.push(self.organisms[i].lineage_id);
                let recycled = (0.25 + m.mass * 1.9).clamp(0.15, 0.90);
                deposits.push((self.organisms[i].x, self.organisms[i].y, recycled, 0.0, 0.03, 0.0));
                continue;
            }

            // Mitotic reproduction
            if self.organisms[i].energy > 88.0
                && m.mass > 0.075
                && count + new_offspring.len() < max_orgs
            {
                let mut_rate = self.universe.mutation_rate
                    * if env == Environment::Storm { 1.5 } else { 1.0 };
                let child_morph = self.organisms[i].morph_genome.mutate(&mut self.rng, mut_rate);
                let child_brain = self.organisms[i].brain.mutate(&mut self.rng, mut_rate * 1.4);
                let mut child_eco = self.organisms[i].eco_genome;
                child_eco.metabolism =
                    (child_eco.metabolism + self.rng.gen_range(-0.004..0.004)).clamp(0.01, 0.12);
                child_eco.membrane =
                    (child_eco.membrane + self.rng.gen_range(-0.05..0.05)).clamp(0.1, 1.4);

                let daughter_body = self.organisms[i].body.divide(&child_morph, &mut self.rng);
                self.organisms[i].energy *= 0.48;
                self.organisms[i].offspring_count += 1;

                let angle = self.rng.gen_range(0.0..TAU);
                let child_id = self.next_org_id;
                self.next_org_id += 1;
                self.total_births += 1;

                let parent = &self.organisms[i];
                let child_gen = parent.generation + 1;
                self.phylogeny
                    .record_birth(parent.lineage_id, parent.tribe, child_gen, self.tick);

                new_offspring.push(MorphOrganism {
                    id: child_id,
                    parent_id: Some(parent.id),
                    lineage_id: parent.lineage_id,
                    generation: child_gen,
                    tribe: parent.tribe,
                    archetype: derive_archetype(child_eco, crate::particle::RareTrait::None, 8),
                    eco_genome: child_eco,
                    morph_genome: child_morph,
                    brain: child_brain,
                    body: daughter_body,
                    x: (parent.x + angle.cos() * 0.05).clamp(-0.92, 0.92),
                    y: (parent.y + angle.sin() * 0.05).clamp(-0.92, 0.92),
                    vx: angle.cos() * 0.004,
                    vy: angle.sin() * 0.004,
                    energy: parent.energy,
                    age: 0,
                    offspring_count: 0,
                    alive: true,
                });
            }
        }

        for lin in deaths_lineage {
            self.phylogeny.record_death(lin, self.tick);
        }
        for (x, y, dn, dp, dt, dr) in deposits {
            self.eco_field.deposit_world(x, y, dn, dp, dt, dr);
        }

        self.organisms.retain(|o| o.alive);
        self.organisms.extend(new_offspring);

        // Configurable abiogenesis floor (can be disabled for permanent extinction experiments)
        if self.universe.allow_abiogenesis && self.organisms.len() < 8 && self.tick % 20 == 0 {
            let tribe = Tribe::from_index(self.rng.gen_range(0..6));
            let eco_genome = random_eco_genome(&mut self.rng);
            let morph_genome = MorphGenome::from_ecological(eco_genome, tribe, &mut self.rng);
            let brain = NeuralBrain::random(&mut self.rng);
            let heading = self.rng.gen_range(0.0..TAU);
            let body = LeniaBody::new(&morph_genome, heading, &mut self.rng);
            let id = self.next_org_id;
            self.next_org_id += 1;
            self.phylogeny.record_birth(id, tribe, 0, self.tick);

            self.organisms.push(MorphOrganism {
                id,
                parent_id: None,
                lineage_id: id,
                generation: 0,
                tribe,
                archetype: derive_archetype(eco_genome, crate::particle::RareTrait::None, 6),
                eco_genome,
                morph_genome,
                brain,
                body,
                x: self.rng.gen_range(-0.8..0.8),
                y: self.rng.gen_range(-0.8..0.8),
                vx: 0.0,
                vy: 0.0,
                energy: 55.0,
                age: 0,
                offspring_count: 0,
                alive: true,
            });
        }
    }

    fn trigger_catastrophe(&mut self) {
        // Inject a toxin wave and deplete half of ambient nutrients
        for i in 0..(self.eco_field.width * self.eco_field.height) {
            let idx = i * 4;
            self.eco_field.cells[idx] *= 0.45;
            self.eco_field.cells[idx + 2] = (self.eco_field.cells[idx + 2] + 0.42).min(1.0);
        }
    }

    pub fn stats(&self) -> ConvergenceStats {
        let n = self.organisms.len();
        let inv = if n > 0 { 1.0 / n as f32 } else { 0.0 };

        let mut mass = 0.0;
        let mut mem = 0.0;
        let mut org_m = 0.0;
        let mut act = 0.0;
        let mut mu = 0.0;
        let mut sigma = 0.0;
        let mut asym = 0.0;
        let mut max_gen = 0;

        for o in &self.organisms {
            mass += o.body.metrics.mass;
            mem += o.body.metrics.membrane_mass;
            org_m += o.body.metrics.organelle_mass;
            act += o.body.metrics.activity;
            mu += o.morph_genome.mu;
            sigma += o.morph_genome.sigma;
            asym += o.morph_genome.asymmetry;
            max_gen = max_gen.max(o.generation);
        }

        ConvergenceStats {
            tick: self.tick,
            organism_count: n,
            active_lineages: self.phylogeny.active_lineage_count(),
            archive_discoveries: self.phylogeny.filled_archive_cells(),
            extinction_events: self.phylogeny.extinction_events,
            mean_mass: mass * inv,
            mean_membrane: mem * inv,
            mean_organelle: org_m * inv,
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
}

#[inline]
fn hash_cell(x: f32, y: f32) -> (i32, i32) {
    ((x / HASH_CELL_SIZE).floor() as i32, (y / HASH_CELL_SIZE).floor() as i32)
}

#[inline]
pub fn wrap_world_coord(v: f32) -> f32 {
    let mut w = (v + 1.0).rem_euclid(2.0) - 1.0;
    if w >= 1.0 {
        w -= 2.0;
    }
    w
}

#[inline]
pub fn toroidal_delta(d: f32) -> f32 {
    if d > 1.0 {
        d - 2.0
    } else if d < -1.0 {
        d + 2.0
    } else {
        d
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
