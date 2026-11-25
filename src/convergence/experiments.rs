//! Experimental validation suite for Project Convergence.
//!
//! Runs isolated quantitative experiments measuring:
//! 1. Morphological stability & persistence across nutrient regimes
//! 2. Bidirectional nutrient-morphology coupling & resource recycling
//! 3. Multi-generation heritable variation & environmental adaptation
//! 4. Long-duration stability, throughput, and memory footprint

use crate::app::Environment;
use crate::convergence::engine::ConvergenceWorld;
use crate::convergence::lenia_body::{LeniaBody, LocalEnvInput, MorphGenome, BODY_CELLS};
use crate::particle::{Genome, Tribe};
use rand::{rngs::StdRng, SeedableRng};
use std::time::Instant;

pub fn run_all_experiments(prefer_gpu: bool) {
    println!("==================================================================");
    println!("  SYMBIOTE: PROJECT ABIOGENESIS — CONVERGENCE VALIDATION SUITE");
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
    println!("\n[Exp 1] Morphological Stability & Persistence (200 steps per tribe)");
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

    let tribes = [
        Tribe::Blood,
        Tribe::Moss,
        Tribe::Deep,
        Tribe::Solar,
        Tribe::Dream,
        Tribe::Static,
    ];

    for tribe in tribes {
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
            if m.mass > 0.015 {
                alive_steps += 1;
            }
        }

        println!(
            "  Tribe {:3} | init_mass={:.3} final_mass={:.3} mem={:.3} act={:.4} persist={}/200",
            tribe.name(),
            init_mass,
            body.metrics.mass,
            body.metrics.membrane_mass,
            body.metrics.activity,
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
        "  Nutrient-rich  (0.90) -> mass={:.3}, membrane={:.3}, com_dx={:+.3}",
        rich_body.metrics.mass, rich_body.metrics.membrane_mass, rich_body.metrics.com_dx
    );
    println!(
        "  Nutrient-poor  (0.02) -> mass={:.3}, membrane={:.3}, com_dx={:+.3}",
        starved_body.metrics.mass, starved_body.metrics.membrane_mass, starved_body.metrics.com_dx
    );
    println!(
        "  Coupling delta (rich - poor mass): {:+.3}",
        rich_body.metrics.mass - starved_body.metrics.mass
    );
}

fn experiment_3_evolution_and_adaptation(prefer_gpu: bool) {
    println!("\n[Exp 3] Multi-Generation Heritable Variation & Environmental Adaptation (600 ticks)");
    let mut world = ConvergenceWorld::new(0xE010_7104, prefer_gpu);
    let s0 = world.stats();
    println!(
        "  t=  0 | orgs={:2} lineages={:2} mu={:.4} sigma={:.4} asym={:.3} nutr={:.3} backend={}",
        s0.organism_count,
        s0.active_lineages,
        s0.mean_mu,
        s0.mean_sigma,
        s0.mean_asymmetry,
        s0.mean_nutrient,
        if s0.gpu_active { "GPU+Rayon" } else { "CPU-Rayon" }
    );

    for t in 1..=600 {
        if t == 200 {
            world.symbiote.environment = Environment::Hunger;
        } else if t == 400 {
            world.symbiote.environment = Environment::Bloom;
        }
        world.step();

        if t % 150 == 0 {
            let s = world.stats();
            println!(
                "  t={:3} | orgs={:2} lineages={:2} births={:3} deaths={:3} max_gen={:2} mu={:.4} asym={:.3} nutr={:.3} env={}",
                s.tick,
                s.organism_count,
                s.active_lineages,
                s.total_births,
                s.total_deaths,
                s.max_generation,
                s.mean_mu,
                s.mean_asymmetry,
                s.mean_nutrient,
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
