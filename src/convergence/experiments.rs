//! Experimental validation suite & automated scientific assertions (Experiments A-J)
//! for Symbiote: Abiogenesis v0.4.0 — Project GENOMORPH.

use crate::app::Environment;
use crate::convergence::engine::ConvergenceWorld;
use crate::convergence::genomorph::{diff_genomes, AnatomyPhenotype, GenomorphGenome, OrganKind};
use crate::convergence::lenia_body::{LeniaBody, LocalEnvInput, MorphGenome, BODY_CELLS};
use crate::convergence::neural_graph::NeuralGraph;
use crate::convergence::universes::UniversePreset;
use crate::particle::{Genome, Tribe};
use rand::{rngs::StdRng, SeedableRng};
use std::time::Instant;

pub fn run_all_experiments(prefer_gpu: bool) {
    println!("==================================================================");
    println!("  SYMBIOTE: ABIOGENESIS v0.4.0 — GENOMORPH VALIDATION SUITE");
    println!("==================================================================");

    experiment_1_morphological_stability();
    experiment_2_bidirectional_coupling();
    experiment_3_genomorph_evolution(prefer_gpu);
    experiment_4_performance_and_memory(prefer_gpu);

    println!("==================================================================");
    println!("  ALL GENOMORPH EXPERIMENTS PASSED");
    println!("==================================================================");
}

fn experiment_1_morphological_stability() {
    println!("\n[Exp 1] 3-Channel Morphological Stability & Persistence (200 steps per tribe)");
    let mut rng = StdRng::seed_from_u64(0xAB10_6E4E);
    let eco = Genome {
        perception: 0.24,
        hunger: 0.025,
        bonding: 0.85,
        volatility: 1.0,
        orbit: 0.5,
        membrane: 0.72,
        metabolism: 0.04,
        fertility: 0.70,
    };

    for tribe in [
        Tribe::Blood,
        Tribe::Moss,
        Tribe::Deep,
        Tribe::Solar,
        Tribe::Dream,
        Tribe::Static,
    ] {
        let mg = MorphGenome::from_ecological(eco, tribe, &mut rng);
        let mut body = LeniaBody::new(&mg, 0.0, &mut rng);
        let init_mass = body.metrics.mass;

        let env = LocalEnvInput {
            nutrient: 0.65,
            pheromone: 0.1,
            toxin: 0.0,
            nutrient_grad_x: 0.05,
            nutrient_grad_y: 0.0,
            mutagen: 0.0,
        };

        let mut alive_steps = 0;
        for _ in 0..200 {
            let m = body.step_cpu(&mg, env);
            assert!(
                m.mass.is_finite() && m.membrane_mass.is_finite() && m.organelle_mass.is_finite()
            );
            if m.mass > 0.015 {
                alive_steps += 1;
            }
        }

        println!(
            "  Tribe {:3} | init={:.3} final={:.3} mem={:.3} org={:.3} persist={}/200",
            tribe.name(),
            init_mass,
            body.metrics.mass,
            body.metrics.membrane_mass,
            body.metrics.organelle_mass,
            alive_steps
        );
    }
}

fn experiment_2_bidirectional_coupling() {
    println!("\n[Exp 2] Bidirectional Nutrient-Morphology Coupling");
    let mut rng = StdRng::seed_from_u64(0xC00F_1EED);
    let eco = Genome {
        perception: 0.25,
        hunger: 0.03,
        bonding: 0.80,
        volatility: 1.0,
        orbit: 0.45,
        membrane: 0.75,
        metabolism: 0.04,
        fertility: 0.75,
    };
    let mg = MorphGenome::from_ecological(eco, Tribe::Moss, &mut rng);

    let mut rich_body = LeniaBody::new(&mg, 0.0, &mut rng);
    let mut starved_body = rich_body.clone();

    let rich_env = LocalEnvInput {
        nutrient: 0.90,
        pheromone: 0.2,
        toxin: 0.0,
        nutrient_grad_x: 0.1,
        nutrient_grad_y: 0.0,
        mutagen: 0.0,
    };
    let starved_env = LocalEnvInput {
        nutrient: 0.02,
        pheromone: 0.0,
        toxin: 0.35,
        nutrient_grad_x: 0.0,
        nutrient_grad_y: 0.0,
        mutagen: 0.0,
    };

    for _ in 0..120 {
        rich_body.step_cpu(&mg, rich_env);
        starved_body.step_cpu(&mg, starved_env);
    }

    println!(
        "  Nutrient-rich  (0.90) -> mass={:.3}, mem={:.3}, org={:.3}",
        rich_body.metrics.mass, rich_body.metrics.membrane_mass, rich_body.metrics.organelle_mass
    );
    println!(
        "  Nutrient-poor  (0.02) -> mass={:.3}, mem={:.3}, org={:.3}",
        starved_body.metrics.mass,
        starved_body.metrics.membrane_mass,
        starved_body.metrics.organelle_mass
    );
    assert!(rich_body.metrics.mass > starved_body.metrics.mass);
}

fn experiment_3_genomorph_evolution(prefer_gpu: bool) {
    println!("\n[Exp 3] GENOMORPH Structural & Neural Evolution (400 ticks, Predator's Paradise)");
    let mut world = ConvergenceWorld::from_universe(
        0xE010_7104,
        UniversePreset::PredatorsParadise.config(),
        prefer_gpu,
    );
    let s0 = world.stats();
    println!(
        "  t=  0 | orgs={:3} lin={:2} genes={:.2} syn={:.1} carn={:.2} backend={}",
        s0.organism_count,
        s0.active_lineages,
        s0.mean_genes,
        s0.mean_synapses,
        s0.mean_carnivory,
        if s0.gpu_active { "GPU+Rayon" } else { "CPU-Rayon" }
    );

    for t in 1..=400 {
        if t == 180 {
            world.symbiote.environment = Environment::Hunger;
        } else if t == 320 {
            world.symbiote.environment = Environment::Bloom;
        }
        world.step();

        if t % 100 == 0 {
            let s = world.stats();
            println!(
                "  t={:3} | orgs={:3} lin={:2} births={:3} deaths={:3} pred={:4} gen={:2} genes={:.2} syn={:.1} arch={:2}/36",
                s.tick,
                s.organism_count,
                s.active_lineages,
                s.total_births,
                s.total_deaths,
                s.total_predations,
                s.max_generation,
                s.mean_genes,
                s.mean_synapses,
                s.archive_discoveries
            );
        }
    }
}

fn experiment_4_performance_and_memory(prefer_gpu: bool) {
    println!("\n[Exp 4] Throughput & Memory Footprint Benchmark (150 ticks)");
    let mut world = ConvergenceWorld::new(0xBEEF_4070, prefer_gpu);
    let start = Instant::now();
    for _ in 0..150 {
        world.step();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let tps = 150.0 / elapsed.max(1e-6);
    let body_bytes = world.organisms.len() * BODY_CELLS * std::mem::size_of::<f32>() * 2;
    let eco_bytes = world.eco_field.cells.len() * std::mem::size_of::<f32>() * 2;
    let total_kb = (body_bytes + eco_bytes) as f64 / 1024.0;

    println!(
        "  150 ticks in {:.3}s ({:.1} ticks/sec) | orgs={} | continuous fields={:.1} KB | backend={}",
        elapsed,
        tps,
        world.organisms.len(),
        total_kb,
        if world.use_gpu { "GPU-Compute + Rayon" } else { "CPU-Rayon" }
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convergence::persistence::{load_world, save_world_atomic};

    /// Experiment A: Anatomical Inheritance — verify stable homology IDs are preserved.
    #[test]
    fn exp_a_anatomical_inheritance() {
        let mut rng = StdRng::seed_from_u64(101);
        let parent = GenomorphGenome::from_tribe(Tribe::Blood, &mut rng);
        let child = parent.clone();
        let diff = diff_genomes(&parent, &child);
        assert_eq!(diff.retained.len(), parent.genes.len());
        assert!(diff.added.is_empty() && diff.removed.is_empty() && diff.modified.is_empty());
    }

    /// Experiment B: Structural Mutation — duplication, innovation, deletion & validity bounds.
    #[test]
    fn exp_b_structural_mutation_and_crossover() {
        let mut rng = StdRng::seed_from_u64(202);
        let founder = GenomorphGenome::minimal_founder(&mut rng);
        let mut cur = founder.clone();
        let mut saw_innovation = false;
        for _ in 0..25 {
            cur = cur.mutate(&mut rng, 0.25);
            if cur.genes.len() > founder.genes.len() {
                saw_innovation = true;
            }
            assert!(cur.genes.len() >= 1 && cur.genes.len() <= 14);
        }
        assert!(saw_innovation, "Structural mutations must sprout new anatomical genes");
        let diff = diff_genomes(&founder, &cur);
        assert!(!diff.added.is_empty() || !diff.modified.is_empty());

        let partner = GenomorphGenome::from_tribe(Tribe::Deep, &mut rng);
        let hybrid = cur.crossover_homologous(&partner, &mut rng);
        assert!(!hybrid.genes.is_empty());
    }

    /// Experiment C & D & E: Predatory & Defensive Anatomy Trade-offs.
    #[test]
    fn exp_de_predatory_and_defensive_anatomy_tradeoffs() {
        let mut rng = StdRng::seed_from_u64(303);
        let hunter = GenomorphGenome::from_tribe(Tribe::Blood, &mut rng);
        let armored = GenomorphGenome::from_tribe(Tribe::Moss, &mut rng);
        let p_hunter = AnatomyPhenotype::express(&hunter, 1.0, 0.0);
        let p_armored = AnatomyPhenotype::express(&armored, 1.0, 0.0);

        assert!(
            p_hunter.bite_power > p_armored.bite_power,
            "Jawed hunter must have greater bite power than filter grazer"
        );
        assert!(
            p_armored.armor_rating > p_hunter.armor_rating,
            "Carapace grazer must have stronger armor rating than hunter"
        );
        assert!(
            p_armored.filter_efficiency > p_hunter.filter_efficiency,
            "Filter veil grazer must absorb ambient nutrients more efficiently"
        );
    }

    /// Experiment F: Neural Evolution — structural neural mutations & bounded outputs.
    #[test]
    fn exp_f_neural_evolution() {
        let mut rng = StdRng::seed_from_u64(404);
        let mut brain = NeuralGraph::minimal(&mut rng);
        let init_nodes = brain.nodes.len();
        let init_conns = brain.connections.len();
        for _ in 0..15 {
            brain = brain.mutate(&mut rng, 0.25);
        }
        assert!(
            brain.nodes.len() > init_nodes || brain.connections.len() > init_conns,
            "NeuralGraph must evolve new nodes or connections"
        );
        let out = brain.step([0.5, 0.2, -0.1, 0.0, 0.3, 0.4, 0.8, 0.1, 0.7, 0.2]);
        assert!(out.iter().all(|v| v.is_finite() && *v >= -1.0 && *v <= 1.0));
    }

    /// Experiment G & H: Developmental Biology, Numerical Stability & Damage Accounting.
    #[test]
    fn exp_gh_development_and_resource_accounting() {
        let mut rng = StdRng::seed_from_u64(505);
        let geno = GenomorphGenome::from_tribe(Tribe::Static, &mut rng);
        let embryo = AnatomyPhenotype::express(&geno, 0.30, 0.0);
        let adult = AnatomyPhenotype::express(&geno, 1.00, 0.0);
        assert!(
            adult.stomach_capacity > embryo.stomach_capacity
                && adult.propulsion_power > embryo.propulsion_power,
            "Adult phenotype must scale beyond embryonic phenotype"
        );

        let eco = Genome {
            perception: 0.25,
            hunger: 0.025,
            bonding: 0.8,
            volatility: 1.0,
            orbit: 0.5,
            membrane: 0.7,
            metabolism: 0.04,
            fertility: 0.7,
        };
        let mg = MorphGenome::from_ecological(eco, Tribe::Deep, &mut rng);
        let mut body = LeniaBody::new(&mg, 0.0, &mut rng);
        let before = body.metrics.mass;
        let lost = body.apply_damage(0.35);
        assert!(lost > 0.0 && body.metrics.mass < before);
        assert!(body.cells.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
    }

    /// Experiment I & J: Persistent Evolution (v4 Atomic Save/Load) & Multi-Tick Stability.
    #[test]
    fn exp_ij_persistent_evolution_and_stability() {
        let mut world =
            ConvergenceWorld::from_universe(1337, UniversePreset::GenesisZero.config(), false);
        for _ in 0..30 {
            world.step();
        }
        assert!(world.organisms.iter().all(|o| {
            o.energy.is_finite()
                && o.x.is_finite()
                && o.y.is_finite()
                && world.bounds.contains(o.x, o.y)
        }));
        let path = save_world_atomic(&world, "unit_test_infinity_v4").expect("save failed");
        let restored = load_world(&path, false).expect("load failed");
        assert_eq!(restored.tick, world.tick);
        assert_eq!(restored.bounds, world.bounds);
        assert_eq!(restored.organisms.len(), world.organisms.len());
        assert_eq!(restored.universe.preset, UniversePreset::GenesisZero);
        if let (Some(a), Some(b)) = (world.organisms.first(), restored.organisms.first()) {
            assert_eq!(a.genomorph.genes.len(), b.genomorph.genes.len());
            assert_eq!(a.brain.connections.len(), b.brain.connections.len());
            assert_eq!(a.genomorph.genes[0].kind, OrganKind::CoreSegment);
        }
        let _ = std::fs::remove_file(path);
    }

    /// Project INFINITY: Camera projection, cursor-centered zoom, aspect-ratio fit_world,
    /// toroidal delta symmetry, and frustum culling invariants.
    #[test]
    fn exp_k_infinity_camera_and_world_bounds() {
        use crate::convergence::camera::{SimCamera, WorldBounds};

        let bounds = WorldBounds::new(16.0, 9.0);
        // 1. Toroidal wrap & shortest signed delta symmetry
        assert!((bounds.wrap_x(-0.25) - 15.75).abs() < 1e-5);
        assert!((bounds.wrap_y(9.5) - 0.5).abs() < 1e-5);
        assert!((bounds.delta_x(15.5) - (-0.5)).abs() < 1e-5);
        assert!((bounds.delta_x(-15.5) - 0.5).abs() < 1e-5);
        assert!((bounds.delta_y(8.5) - (-0.5)).abs() < 1e-5);

        // 2. fit_world shows entire world without cropping across 16:9, 21:9, and 4:3
        for aspect in [16.0 / 9.0, 21.0 / 9.0, 4.0 / 3.0] {
            let cam = SimCamera::fit_world(bounds, aspect);
            let (min_x, min_y, max_x, max_y) = cam.visible_rect(aspect);
            assert!(min_x <= 0.0 && max_x >= bounds.width);
            assert!(min_y <= 0.0 && max_y >= bounds.height);
        }

        // 3. screen_to_world <-> world_to_uv roundtrip
        let mut cam = SimCamera::fit_world(bounds, 16.0 / 9.0);
        let (wx, wy) = cam.screen_to_world(480.0, 270.0, 1440.0, 810.0);
        let (u, v) = cam.world_to_uv(wx, wy, 1440.0 / 810.0);
        assert!((u - (480.0 / 1440.0)).abs() < 1e-4);
        assert!((v - (270.0 / 810.0)).abs() < 1e-4);

        // 4. Cursor-centered zoom keeps target point stable when zoomed in
        cam.half_h = 2.0;
        cam.center_x = 8.0;
        cam.center_y = 4.5;
        let (before_x, before_y) = cam.screen_to_world(900.0, 320.0, 1440.0, 810.0);
        cam.zoom_at_cursor(1.5, 900.0, 320.0, 1440.0, 810.0, bounds);
        let (after_x, after_y) = cam.screen_to_world(900.0, 320.0, 1440.0, 810.0);
        assert!((before_x - after_x).abs() < 1e-3);
        assert!((before_y - after_y).abs() < 1e-3);

        // 5. Frustum culling excludes distant off-screen organisms
        let micro = SimCamera {
            center_x: 8.0,
            center_y: 4.5,
            half_h: 0.8,
            show_bounds: true,
        };
        assert!(micro.is_visible(8.1, 4.4, 0.3, bounds, 16.0 / 9.0));
        assert!(!micro.is_visible(1.0, 1.0, 0.3, bounds, 16.0 / 9.0));
    }
}
