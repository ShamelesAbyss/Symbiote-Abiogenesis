//! Experimental validation suite & automated scientific assertions for Symbiote: Abiogenesis.

use crate::app::Environment;
use crate::convergence::engine::ConvergenceWorld;
use crate::convergence::lenia_body::{LeniaBody, LocalEnvInput, MorphGenome, BODY_CELLS};
use crate::convergence::universes::UniversePreset;
use crate::particle::{Genome, Tribe};
use rand::{rngs::StdRng, SeedableRng};
use std::time::Instant;

pub fn run_all_experiments(prefer_gpu: bool) {
    println!("==================================================================");
    println!("  SYMBIOTE: ABIOGENESIS — EVOLUTION UNBOUND VALIDATION SUITE");
    println!("==================================================================");

    experiment_1_morphological_stability();
    experiment_2_bidirectional_coupling();
    experiment_3_evolution_and_adaptation(prefer_gpu);
    experiment_4_performance_and_memory(prefer_gpu);

    println!("==================================================================");
    println!("  ALL CONVERGENCE EXPERIMENTS COMPLETED");
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
            assert!(m.mass.is_finite() && m.membrane_mass.is_finite() && m.organelle_mass.is_finite());
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
        starved_body.metrics.mass, starved_body.metrics.membrane_mass, starved_body.metrics.organelle_mass
    );
    assert!(
        rich_body.metrics.mass > starved_body.metrics.mass,
        "Nutrient-rich body must sustain greater mass than starved+toxin body"
    );
}

fn experiment_3_evolution_and_adaptation(prefer_gpu: bool) {
    println!("\n[Exp 3] Multi-Generation Evolution, Phylogeny & MAP-Elites Discovery (500 ticks)");
    let mut world = ConvergenceWorld::from_universe(
        0xE010_7104,
        UniversePreset::EvolutionaryArena.config(),
        prefer_gpu,
    );
    let s0 = world.stats();
    println!(
        "  t=  0 | orgs={:3} lineages={:2} mu={:.4} asym={:.3} nutr={:.3} backend={}",
        s0.organism_count,
        s0.active_lineages,
        s0.mean_mu,
        s0.mean_asymmetry,
        s0.mean_nutrient,
        if s0.gpu_active { "GPU+Rayon" } else { "CPU-Rayon" }
    );

    for t in 1..=500 {
        if t == 200 {
            world.symbiote.environment = Environment::Hunger;
        } else if t == 380 {
            world.symbiote.environment = Environment::Bloom;
        }
        world.step();

        if t % 125 == 0 {
            let s = world.stats();
            println!(
                "  t={:3} | orgs={:3} lin={:2} births={:3} deaths={:3} pred={:4} gen={:2} archive={:2}/36 mu={:.4} env={}",
                s.tick,
                s.organism_count,
                s.active_lineages,
                s.total_births,
                s.total_deaths,
                s.total_predations,
                s.max_generation,
                s.archive_discoveries,
                s.mean_mu,
                world.symbiote.environment.name()
            );
        }
    }
}

fn experiment_4_performance_and_memory(prefer_gpu: bool) {
    println!("\n[Exp 4] Throughput & Memory Footprint Benchmark (200 ticks)");
    let mut world = ConvergenceWorld::new(0xBEEF_4070, prefer_gpu);
    let start = Instant::now();
    for _ in 0..200 {
        world.step();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let tps = 200.0 / elapsed.max(1e-6);
    let body_bytes = world.organisms.len() * BODY_CELLS * std::mem::size_of::<f32>() * 2;
    let eco_bytes = world.eco_field.cells.len() * std::mem::size_of::<f32>() * 2;
    let total_kb = (body_bytes + eco_bytes) as f64 / 1024.0;

    println!(
        "  200 ticks in {:.3}s ({:.1} ticks/sec) | orgs={} | continuous fields={:.1} KB | backend={}",
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

    #[test]
    fn test_numerical_stability_and_damage_accounting() {
        let mut rng = StdRng::seed_from_u64(42);
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

    #[test]
    fn test_mutation_bounds_and_crossover() {
        let mut rng = StdRng::seed_from_u64(99);
        let eco = Genome {
            perception: 0.2,
            hunger: 0.02,
            bonding: 0.7,
            volatility: 0.9,
            orbit: 0.4,
            membrane: 0.6,
            metabolism: 0.03,
            fertility: 0.6,
        };
        let a = MorphGenome::from_ecological(eco, Tribe::Blood, &mut rng);
        let b = MorphGenome::from_ecological(eco, Tribe::Solar, &mut rng);
        let child = a.crossover(&b, &mut rng).mutate(&mut rng, 0.2);
        assert!((0.10..=0.30).contains(&child.mu));
        assert!((0.025..=0.088).contains(&child.sigma));
        assert!((5..=9).contains(&child.kernel_radius));
    }

    #[test]
    fn test_atomic_persistence_roundtrip() {
        let mut world = ConvergenceWorld::from_universe(
            1337,
            UniversePreset::Genesis.config(),
            false,
        );
        for _ in 0..10 {
            world.step();
        }
        let path = save_world_atomic(&world, "unit_test_world").expect("save failed");
        let restored = load_world(&path, false).expect("load failed");
        assert_eq!(restored.tick, world.tick);
        assert_eq!(restored.organisms.len(), world.organisms.len());
        assert_eq!(restored.universe.preset, UniversePreset::Genesis);
        let _ = std::fs::remove_file(path);
    }
}
