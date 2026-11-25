//! Unified multiscale artificial-life engine (ConvergenceWorld — Project GENOMORPH).
//!
//! Integrates:
//! 1. Primitive particle chemistry (Symbiote's Particle + RuleMatrix)
//! 2. Localized 40x40x3 continuous Lenia cellular bodies (LeniaBody)
//! 3. Procedural heritable anatomy (GenomorphGenome -> AnatomyPhenotype) & Artificial Embryogenesis
//! 4. Evolving NEAT-inspired neural controllers (NeuralGraph: 10 inputs -> evolvable hidden -> 4 outputs)
//! 5. Digestive biology (stomach_plant/stomach_meat -> gradual assimilation by diet_carnivory)
//! 6. Genuine anatomy-driven predation, armor defense, zero-sum symbiosis & MAP-Elites phylogeny

use crate::app::{App, Environment};
use crate::convergence::camera::WorldBounds;
use crate::convergence::eco_field::{EcoField, ECO_HEIGHT, ECO_WIDTH};
use crate::convergence::genomorph::{AnatomyPhenotype, GenomorphGenome, LifeStage};
use crate::convergence::lenia_body::{LeniaBody, LocalEnvInput, MorphGenome};
use crate::convergence::neural_graph::NeuralGraph;
use crate::convergence::phylogeny::{DiscoveryEntry, PhylogenyArchive};
use crate::convergence::universes::{UniverseConfig, UniversePreset};
use crate::particle::{Genome, Tribe};
use crate::species::{derive_archetype, Archetype};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::TAU;

pub const MAX_ORGANISMS_BUFFER: usize = 320;
const HASH_CELL_SIZE: f32 = 0.85;

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
    pub genomorph: GenomorphGenome,
    #[serde(default)]
    pub phenotype: AnatomyPhenotype,
    pub brain: NeuralGraph,
    pub body: LeniaBody,
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub energy: f32,
    pub stomach_plant: f32,
    pub stomach_meat: f32,
    /// Developmental maturity (0.28 = Embryo -> 0.55 = Juvenile -> 1.0 = Adult).
    pub maturity: f32,
    pub life_stage: LifeStage,
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
    pub mean_genes: f32,
    pub mean_synapses: f32,
    pub mean_carnivory: f32,
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
    pub bounds: WorldBounds,
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
        let bounds = WorldBounds::new(universe.world_width, universe.world_height);
        let symbiote = App::new();
        let eco_field = EcoField::new(ECO_WIDTH, ECO_HEIGHT, bounds, seed);
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

            let genomorph = if universe.minimal_founders {
                GenomorphGenome::minimal_founder(&mut rng)
            } else {
                GenomorphGenome::from_tribe(tribe, &mut rng)
            };
            let brain = if universe.minimal_founders {
                NeuralGraph::minimal(&mut rng)
            } else {
                NeuralGraph::random(&mut rng)
            };
            let heading = rng.gen_range(0.0..TAU);
            let body = LeniaBody::new(&morph_genome, heading, &mut rng);
            let maturity = rng.gen_range(0.65..0.95);
            let phenotype = AnatomyPhenotype::express(&genomorph, maturity, body.pulse_phase);
            let archetype = derive_archetype(eco_genome, crate::particle::RareTrait::None, 8);

            let x = rng.gen_range(0.4..(bounds.width - 0.4));
            let y = rng.gen_range(0.4..(bounds.height - 0.4));
            let id = next_org_id;
            next_org_id += 1;
            phylogeny.record_birth(id, None, id, tribe, 0, 0, &genomorph, &brain);

            organisms.push(MorphOrganism {
                id,
                parent_id: None,
                lineage_id: id,
                generation: 0,
                tribe,
                archetype,
                eco_genome,
                morph_genome,
                genomorph,
                phenotype,
                brain,
                body,
                x,
                y,
                vx: 0.0,
                vy: 0.0,
                energy: 62.0 + rng.gen_range(0.0..18.0),
                stomach_plant: 10.0,
                stomach_meat: 0.0,
                maturity,
                life_stage: stage_from_maturity(maturity, 0, 3200),
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
            bounds,
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

    /// Re-introduce an archived MAP-Elites organism into the live ecosystem
    /// (explicitly marked as user intervention).
    pub fn spawn_from_discovery(&mut self, entry: &DiscoveryEntry) -> Option<u64> {
        let cap = self.universe.max_organisms.min(MAX_ORGANISMS_BUFFER);
        if self.organisms.len() >= cap {
            return None;
        }
        let id = self.next_org_id;
        self.next_org_id += 1;
        let heading = self.rng.gen_range(0.0..TAU);
        let body = LeniaBody::new(&entry.morph_genome, heading, &mut self.rng);
        let mut genomorph = entry.genomorph.clone();
        genomorph.last_mutation = format!("[USER INTERVENTION] Resurrected {}", entry.label);
        let phenotype = AnatomyPhenotype::express(&genomorph, 0.85, body.pulse_phase);
        let eco_genome = random_eco_genome(&mut self.rng);

        self.phylogeny.record_birth(
            id,
            Some(entry.organism_id),
            entry.lineage_id,
            entry.tribe,
            entry.generation,
            self.tick,
            &genomorph,
            &entry.brain,
        );
        self.organisms.push(MorphOrganism {
            id,
            parent_id: Some(entry.organism_id),
            lineage_id: entry.lineage_id,
            generation: entry.generation,
            tribe: entry.tribe,
            archetype: derive_archetype(eco_genome, crate::particle::RareTrait::None, 8),
            eco_genome,
            morph_genome: entry.morph_genome,
            genomorph,
            phenotype,
            brain: entry.brain.clone(),
            body,
            x: self.rng.gen_range(1.0..(self.bounds.width - 1.0)),
            y: self.rng.gen_range(1.0..(self.bounds.height - 1.0)),
            vx: 0.0,
            vy: 0.0,
            energy: 68.0,
            stomach_plant: 12.0,
            stomach_meat: 0.0,
            maturity: 0.85,
            life_stage: LifeStage::Adult,
            age: 0,
            offspring_count: 0,
            alive: true,
        });
        Some(id)
    }

    pub fn step(&mut self) {
        self.tick += 1;

        if self.universe.catastrophe_interval > 0
            && self.tick % self.universe.catastrophe_interval == 0
        {
            self.trigger_catastrophe();
        }

        if self.tick % 2 == 0 {
            self.symbiote.step();
            self.eco_field
                .couple_symbiote_ecology(&self.symbiote.ecology, &self.symbiote.substrate);
        }

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

        let count = self.organisms.len();

        // Build O(N) Toroidal Spatial Hash Grid
        let mut grid: HashMap<(i32, i32), Vec<usize>> = HashMap::with_capacity(count);
        for (i, org) in self.organisms.iter().enumerate() {
            grid.entry(hash_cell(org.x, org.y)).or_default().push(i);
        }

        // Gather grounded local sensory inputs (limited by each organism's actual `sensor_range`!)
        let snap: Vec<(f32, f32, f32, f32, f32, f32, f32, f32)> = self
            .organisms
            .iter()
            .map(|o| {
                (
                    o.x,
                    o.y,
                    o.body.heading,
                    o.body.metrics.mass,
                    o.phenotype.armor_rating,
                    o.phenotype.sensor_range,
                    o.energy,
                    o.maturity * o.genomorph.base_scale,
                )
            })
            .collect();

        let env_and_sense: Vec<(LocalEnvInput, [f32; 10])> = self
            .organisms
            .iter()
            .enumerate()
            .map(|(i, org)| {
                let s = self.eco_field.sample_world(org.x, org.y);
                let local_env = LocalEnvInput {
                    nutrient: s.nutrient * self.universe.nutrient_regen_mult.clamp(0.4, 1.8),
                    pheromone: s.pheromone,
                    toxin: s.toxin,
                    nutrient_grad_x: s.nutrient_grad_x,
                    nutrient_grad_y: s.nutrient_grad_y,
                    mutagen: if env == Environment::Storm { 0.35 } else { 0.05 },
                };

                // Sense nearest neighbor within organism's evolved SensoryAntenna range (world units)
                let (my_x, my_y, my_h, _, _, sensor_r_norm, _, my_scale) = snap[i];
                let sensor_r = sensor_r_norm * 5.2;
                let (cx, cy) = hash_cell(my_x, my_y);
                let mut best_d = sensor_r;
                let mut best_rel_angle = 0.0f32;
                let mut best_rel_size = 0.0f32;
                let mut found = false;

                for oy in -2..=2 {
                    for ox in -2..=2 {
                        let Some(bucket) = grid.get(&(cx + ox, cy + oy)) else {
                            continue;
                        };
                        for &j in bucket {
                            if i == j {
                                continue;
                            }
                            let dx = self.bounds.delta_x(snap[j].0 - my_x);
                            let dy = self.bounds.delta_y(snap[j].1 - my_y);
                            let d = (dx * dx + dy * dy).sqrt();
                            if d > 1e-4 && d < best_d {
                                best_d = d;
                                let ang = dy.atan2(dx) - my_h;
                                best_rel_angle = ang.sin();
                                best_rel_size = ((snap[j].7 / my_scale.max(0.2)) - 1.0).clamp(-1.0, 2.0);
                                found = true;
                            }
                        }
                    }
                }

                let closeness = if found {
                    (1.0 - best_d / sensor_r.max(0.05)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let stomach_fill = ((org.stomach_plant + org.stomach_meat)
                    / org.phenotype.stomach_capacity.max(1.0))
                .clamp(0.0, 1.0);

                let brain_in = [
                    local_env.nutrient,
                    local_env.nutrient_grad_x * 6.0,
                    local_env.nutrient_grad_y * 6.0,
                    local_env.toxin,
                    local_env.pheromone,
                    best_rel_angle,
                    closeness,
                    best_rel_size,
                    (org.energy / 100.0).clamp(0.0, 1.5),
                    stomach_fill,
                ];
                (local_env, brain_in)
            })
            .collect();

        // Step 40x40x3 Lenia bodies, express dynamic AnatomyPhenotype, and evaluate NeuralGraph in parallel
        self.organisms
            .par_iter_mut()
            .zip(env_and_sense.par_iter())
            .for_each(|(org, &(local_env, brain_in))| {
                if !org.alive {
                    return;
                }
                org.body.step_cpu(&org.morph_genome, local_env);
                org.brain.step(brain_in);
                org.phenotype =
                    AnatomyPhenotype::express(&org.genomorph, org.maturity, org.body.pulse_phase);
            });

        // Coherent resource accounting accumulators
        let mut damage_to = vec![0.0f32; count];
        let mut meat_gained = vec![0.0f32; count];
        let mut energy_delta = vec![0.0f32; count];
        let mut deposits: Vec<(f32, f32, f32, f32, f32, f32)> = Vec::with_capacity(count * 2);
        let mut new_offspring: Vec<MorphOrganism> = Vec::new();
        let mut deaths_rec: Vec<(u64, u64)> = Vec::new();
        let max_orgs = self.universe.max_organisms.min(MAX_ORGANISMS_BUFFER);

        for i in 0..count {
            if !self.organisms[i].alive {
                continue;
            }
            self.organisms[i].age += 1;

            // Artificial Embryogenesis: progressive growth from Embryo -> Juvenile -> Adult -> Elder
            // Childhood scarcity genuinely constrains developmental growth!
            if self.organisms[i].maturity < 1.0 && self.organisms[i].energy > 28.0 {
                let growth = 0.0028 * (self.organisms[i].energy / 65.0).clamp(0.25, 1.4);
                self.organisms[i].maturity = (self.organisms[i].maturity + growth).min(1.0);
                self.organisms[i].energy -= growth * 18.0;
            }
            self.organisms[i].life_stage = stage_from_maturity(
                self.organisms[i].maturity,
                self.organisms[i].age,
                self.organisms[i].genomorph.max_age,
            );

            let m = self.organisms[i].body.metrics;
            let local = env_and_sense[i].0;
            let out = self.organisms[i].brain.last_out;
            let pheno = self.organisms[i].phenotype.clone();

            // Anatomy-driven steering & propulsion:
            // turn_torque rotates heading scaled by turn_authority; thrust acts along heading scaled by propulsion_power!
            let turn = out[1] * 0.14 * pheno.turn_authority;
            self.organisms[i].body.heading = (self.organisms[i].body.heading + turn).rem_euclid(TAU);
            let h = self.organisms[i].body.heading;

            let thrust_mag = ((out[0] + 0.35).max(0.0) * 0.022 * pheno.propulsion_power)
                * (0.65 + m.organelle_mass * 2.0);
            let mut fx = h.cos() * thrust_mag + m.com_dx * 0.018;
            let mut fy = h.sin() * thrust_mag + m.com_dy * 0.018;

            let my_x = snap[i].0;
            let my_y = snap[i].1;
            let my_radius = 0.16 + m.radius_norm * 0.22 * self.organisms[i].maturity;
            let want_bite = out[2] > 0.18 || self.organisms[i].genomorph.diet_carnivory > 0.62;
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
                        let dx = self.bounds.delta_x(snap[j].0 - my_x);
                        let dy = self.bounds.delta_y(snap[j].1 - my_y);
                        let d2 = dx * dx + dy * dy;
                        if d2 < 1e-6 || d2 > 1.2 {
                            continue;
                        }
                        let d = d2.sqrt();
                        let contact_r = my_radius + 0.16 + snap[j].3 * 0.22;

                        if d < contact_r {
                            let push = (contact_r - d) * 0.14;
                            fx -= (dx / d) * push;
                            fy -= (dy / d) * push;

                            let forward_dot = (dx / d) * h.cos() + (dy / d) * h.sin();
                            let world_bite_r = pheno.bite_range * 5.0;
                            if want_bite && d <= world_bite_r + contact_r * 0.6 && forward_dot > 0.25 && pheno.bite_power > 0.12 {
                                let defender_armor = snap[j].4;
                                let bite_frac = (pheno.bite_power * 0.32 * (1.0 - defender_armor))
                                    .clamp(0.02, 0.26);
                                damage_to[j] = (damage_to[j] + bite_frac).min(0.65);
                                let flesh = (snap[j].3 * bite_frac * 95.0).min(snap[j].6 * 0.35);
                                meat_gained[i] += flesh;
                                energy_delta[j] -= flesh * 0.85;
                                energy_delta[i] -= 0.45;
                                self.total_predations += 1;
                                deposits.push((snap[j].0, snap[j].1, 0.0, 0.0, 0.10, 0.0));
                            } else if self.organisms[i].lineage_id == self.organisms[j].lineage_id
                                && snap[i].6 > snap[j].6 + 10.0
                            {
                                let transfer = ((snap[i].6 - snap[j].6) * 0.035).min(1.0);
                                energy_delta[i] -= transfer;
                                energy_delta[j] += transfer;
                            }
                        }
                    }
                }
            }

            let bounds = self.bounds;
            let org = &mut self.organisms[i];
            org.vx = (org.vx + fx) * pheno.drag_factor;
            org.vy = (org.vy + fy) * pheno.drag_factor;
            org.x = bounds.wrap_x(org.x + org.vx);
            org.y = bounds.wrap_y(org.y + org.vy);

            // Digestive Biology:
            // 1. Filter/absorb dissolved ambient nutrients into stomach_plant
            let free_stomach =
                (pheno.stomach_capacity - (org.stomach_plant + org.stomach_meat)).max(0.0);
            let absorbed_plant = (local.nutrient
                * pheno.filter_efficiency
                * (1.15 - org.genomorph.diet_carnivory * 0.70)
                * 1.35)
                .min(free_stomach);
            org.stomach_plant += absorbed_plant;

            // 2. Gradual digestion converts stomach_plant & stomach_meat into usable energy
            let dig = org.genomorph.digestion_rate;
            let plant_dig = (org.stomach_plant * 0.09 * dig).min(org.stomach_plant);
            let meat_dig = (org.stomach_meat * 0.12 * dig).min(org.stomach_meat);
            org.stomach_plant -= plant_dig;
            org.stomach_meat -= meat_dig;

            let plant_eff = 1.15 - org.genomorph.diet_carnivory * 0.72;
            let meat_eff = 0.40 + org.genomorph.diet_carnivory * 0.95;
            let gained_energy = plant_dig * plant_eff * 1.45 + meat_dig * meat_eff * 1.85;

            // 3. Coherent Metabolic Costs (anatomy + Lenia mass + swimming + digestion + senescence)
            let speed2 = org.vx * org.vx + org.vy * org.vy;
            let senescence = if org.life_stage == LifeStage::Elder {
                1.45
            } else {
                1.0
            };
            let maintenance = (0.10
                + pheno.total_metabolic_cost
                + m.mass * 0.65
                + speed2 * 8.5
                + dig * 0.08)
                * env.hunger_mult()
                * self.universe.metabolic_cost_mult
                * senescence;

            org.energy = (org.energy + gained_energy - maintenance - local.toxin * 0.55)
                .clamp(0.0, 160.0);

            // Chemical secretion scaled by ToxinGland
            let d_nutr = -(absorbed_plant * 0.045).min(0.08);
            let d_pher = if out[3] > 0.20 && org.energy > 35.0 {
                0.03 * out[3] * pheno.toxin_capacity
            } else {
                0.0
            };
            let d_tox = if out[3] < -0.25 {
                0.03 * (-out[3]) * pheno.toxin_capacity
            } else {
                0.0
            };
            deposits.push((org.x, org.y, d_nutr, d_pher, d_tox, 0.0));
        }

        // Apply deferred predation damage, meat intake, discovery checks, death, and reproduction
        for i in 0..count {
            if !self.organisms[i].alive {
                continue;
            }
            if damage_to[i] > 0.0 {
                self.organisms[i].body.apply_damage(damage_to[i]);
            }
            if meat_gained[i] > 0.0 {
                let cap = self.organisms[i].phenotype.stomach_capacity;
                let room = (cap - (self.organisms[i].stomach_plant + self.organisms[i].stomach_meat))
                    .max(0.0);
                self.organisms[i].stomach_meat += meat_gained[i].min(room);
            }
            self.organisms[i].energy =
                (self.organisms[i].energy + energy_delta[i]).clamp(0.0, 160.0);

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
                    &o.genomorph,
                    &o.brain,
                    m,
                    self.tick,
                );
            }

            // Death & organic decomposition back into EcoField
            if self.organisms[i].energy <= 1.0
                || m.mass < 0.012
                || self.organisms[i].age > self.organisms[i].genomorph.max_age + 800
            {
                self.organisms[i].alive = false;
                self.total_deaths += 1;
                deaths_rec.push((self.organisms[i].id, self.organisms[i].lineage_id));
                let recycled = (0.25
                    + m.mass * 1.9
                    + (self.organisms[i].stomach_plant + self.organisms[i].stomach_meat) * 0.015)
                    .clamp(0.18, 0.95);
                deposits.push((
                    self.organisms[i].x,
                    self.organisms[i].y,
                    recycled,
                    0.0,
                    0.03,
                    0.0,
                ));
                continue;
            }

            // Mature Reproduction (requires Adult stage + sufficient energy + non-collapsed body)
            if self.organisms[i].maturity >= 0.85
                && self.organisms[i].energy > 90.0
                && m.mass > 0.065
                && count + new_offspring.len() < max_orgs
            {
                let mut_rate = self.universe.mutation_rate
                    * if env == Environment::Storm { 1.4 } else { 1.0 };
                let child_morph = self.organisms[i].morph_genome.mutate(&mut self.rng, mut_rate);
                let child_geno = self.organisms[i].genomorph.mutate(&mut self.rng, mut_rate);
                let child_brain = self.organisms[i].brain.mutate(&mut self.rng, mut_rate * 1.35);
                let mut child_eco = self.organisms[i].eco_genome;
                child_eco.metabolism =
                    (child_eco.metabolism + self.rng.gen_range(-0.004..0.004)).clamp(0.01, 0.12);

                let invest = self.organisms[i].genomorph.repro_investment;
                let child_energy = self.organisms[i].energy * invest;
                self.organisms[i].energy *= 1.0 - invest;
                self.organisms[i].offspring_count += 1;

                let daughter_body = self.organisms[i].body.divide(&child_morph, &mut self.rng);
                let child_maturity = 0.30;
                let child_pheno =
                    AnatomyPhenotype::express(&child_geno, child_maturity, daughter_body.pulse_phase);

                let angle = self.rng.gen_range(0.0..TAU);
                let child_id = self.next_org_id;
                self.next_org_id += 1;
                self.total_births += 1;

                let parent = &self.organisms[i];
                let child_gen = parent.generation + 1;
                self.phylogeny.record_birth(
                    child_id,
                    Some(parent.id),
                    parent.lineage_id,
                    parent.tribe,
                    child_gen,
                    self.tick,
                    &child_geno,
                    &child_brain,
                );

                new_offspring.push(MorphOrganism {
                    id: child_id,
                    parent_id: Some(parent.id),
                    lineage_id: parent.lineage_id,
                    generation: child_gen,
                    tribe: parent.tribe,
                    archetype: derive_archetype(child_eco, crate::particle::RareTrait::None, 8),
                    eco_genome: child_eco,
                    morph_genome: child_morph,
                    genomorph: child_geno,
                    phenotype: child_pheno,
                    brain: child_brain,
                    body: daughter_body,
                    x: self.bounds.wrap_x(parent.x + angle.cos() * 0.32),
                    y: self.bounds.wrap_y(parent.y + angle.sin() * 0.32),
                    vx: angle.cos() * 0.016,
                    vy: angle.sin() * 0.016,
                    energy: child_energy,
                    stomach_plant: 4.0,
                    stomach_meat: 0.0,
                    maturity: child_maturity,
                    life_stage: LifeStage::Embryo,
                    age: 0,
                    offspring_count: 0,
                    alive: true,
                });
            }
        }

        for (oid, lin) in deaths_rec {
            self.phylogeny.record_death(oid, lin, self.tick);
        }
        for (x, y, dn, dp, dt, dr) in deposits {
            self.eco_field.deposit_world(x, y, dn, dp, dt, dr);
        }

        self.organisms.retain(|o| o.alive);
        self.organisms.extend(new_offspring);

        // Configurable abiogenesis floor (disabled in Genesis Zero, Abyssal, Arena, etc.)
        if self.universe.allow_abiogenesis && self.organisms.len() < 8 && self.tick % 20 == 0 {
            let tribe = Tribe::from_index(self.rng.gen_range(0..6));
            let eco_genome = random_eco_genome(&mut self.rng);
            let morph_genome = MorphGenome::from_ecological(eco_genome, tribe, &mut self.rng);
            let genomorph = if self.universe.minimal_founders {
                GenomorphGenome::minimal_founder(&mut self.rng)
            } else {
                GenomorphGenome::from_tribe(tribe, &mut self.rng)
            };
            let brain = NeuralGraph::minimal(&mut self.rng);
            let heading = self.rng.gen_range(0.0..TAU);
            let body = LeniaBody::new(&morph_genome, heading, &mut self.rng);
            let phenotype = AnatomyPhenotype::express(&genomorph, 0.55, body.pulse_phase);
            let id = self.next_org_id;
            self.next_org_id += 1;
            self.phylogeny
                .record_birth(id, None, id, tribe, 0, self.tick, &genomorph, &brain);

            self.organisms.push(MorphOrganism {
                id,
                parent_id: None,
                lineage_id: id,
                generation: 0,
                tribe,
                archetype: derive_archetype(eco_genome, crate::particle::RareTrait::None, 6),
                eco_genome,
                morph_genome,
                genomorph,
                phenotype,
                brain,
                body,
                x: self.rng.gen_range(0.5..(self.bounds.width - 0.5)),
                y: self.rng.gen_range(0.5..(self.bounds.height - 0.5)),
                vx: 0.0,
                vy: 0.0,
                energy: 58.0,
                stomach_plant: 8.0,
                stomach_meat: 0.0,
                maturity: 0.55,
                life_stage: LifeStage::Juvenile,
                age: 0,
                offspring_count: 0,
                alive: true,
            });
        }
    }

    fn trigger_catastrophe(&mut self) {
        for i in 0..(self.eco_field.width * self.eco_field.height) {
            let idx = i * 4;
            self.eco_field.cells[idx] *= 0.45;
            self.eco_field.cells[idx + 2] = (self.eco_field.cells[idx + 2] + 0.42).min(1.0);
        }
    }

    pub fn stats(&self) -> ConvergenceStats {
        let n = self.organisms.len();
        let inv = if n > 0 { 1.0 / n as f32 } else { 0.0 };

        let mut genes = 0.0;
        let mut synapses = 0.0;
        let mut carn = 0.0;
        let mut mass = 0.0;
        let mut mem = 0.0;
        let mut org_m = 0.0;
        let mut act = 0.0;
        let mut mu = 0.0;
        let mut sigma = 0.0;
        let mut asym = 0.0;
        let mut max_gen = 0;

        for o in &self.organisms {
            genes += o.genomorph.genes.len() as f32;
            synapses += o.brain.active_connection_count() as f32;
            carn += o.genomorph.diet_carnivory;
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
            mean_genes: genes * inv,
            mean_synapses: synapses * inv,
            mean_carnivory: carn * inv,
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

fn stage_from_maturity(maturity: f32, age: u64, max_age: u64) -> LifeStage {
    if maturity < 0.45 {
        LifeStage::Embryo
    } else if maturity < 0.85 {
        LifeStage::Juvenile
    } else if age > max_age {
        LifeStage::Elder
    } else {
        LifeStage::Adult
    }
}

#[inline]
fn hash_cell(x: f32, y: f32) -> (i32, i32) {
    (
        (x / HASH_CELL_SIZE).floor() as i32,
        (y / HASH_CELL_SIZE).floor() as i32,
    )
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
