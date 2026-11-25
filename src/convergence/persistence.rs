//! Versioned (`v3`), atomic world persistence for Symbiote: Abiogenesis — Project GENOMORPH.
//!
//! Preserves complete organism state (`GenomorphGenome`, `NeuralGraph`, `LeniaBody`,
//! developmental maturity, stomach contents), `EcoField`, `PhylogenyArchive`,
//! deterministic RNG step counter, and universe config using atomic write-then-rename.

use crate::app::Environment;
use crate::convergence::eco_field::{ECO_HEIGHT, ECO_WIDTH};
use crate::convergence::engine::{ConvergenceWorld, MorphOrganism, MAX_ORGANISMS_BUFFER};
use crate::convergence::genomorph::AnatomyPhenotype;
use crate::convergence::phylogeny::PhylogenyArchive;
use crate::convergence::universes::UniverseConfig;
use anyhow::{bail, Result};
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const SAVE_VERSION: u32 = 4;
pub const WORLDS_DIR: &str = "saves/worlds";
const MAX_SAVE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
pub struct WorldSnapshot {
    pub version: u32,
    pub name: String,
    pub seed: u64,
    pub tick: u64,
    pub rng_state_seed: u64,
    pub environment: Environment,
    pub universe: UniverseConfig,
    pub eco_cells: Vec<f32>,
    pub organisms: Vec<MorphOrganism>,
    pub phylogeny: PhylogenyArchive,
    pub next_org_id: u64,
    pub total_births: u64,
    pub total_deaths: u64,
    pub total_predations: u64,
}

pub fn save_world_atomic(world: &ConvergenceWorld, name: &str) -> Result<PathBuf> {
    fs::create_dir_all(WORLDS_DIR)?;
    let safe_name: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let final_path = Path::new(WORLDS_DIR).join(format!("{safe_name}.abio.json"));
    let temp_path = Path::new(WORLDS_DIR).join(format!("{safe_name}.abio.json.tmp"));

    let rng_state_seed = world.seed ^ world.tick.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let snap = WorldSnapshot {
        version: SAVE_VERSION,
        name: safe_name,
        seed: world.seed,
        tick: world.tick,
        rng_state_seed,
        environment: world.symbiote.environment,
        universe: world.universe,
        eco_cells: world.eco_field.cells.clone(),
        organisms: world.organisms.clone(),
        phylogeny: world.phylogeny.clone(),
        next_org_id: world.next_org_id,
        total_births: world.total_births,
        total_deaths: world.total_deaths,
        total_predations: world.total_predations,
    };

    let json = serde_json::to_vec(&snap)?;
    fs::write(&temp_path, json)?;
    fs::rename(&temp_path, &final_path)?;
    Ok(final_path)
}

pub fn load_world(path: &Path, prefer_gpu: bool) -> Result<ConvergenceWorld> {
    let data = fs::read(path)?;
    if data.len() > MAX_SAVE_BYTES {
        bail!("save file exceeds maximum allowed size ({MAX_SAVE_BYTES} bytes)");
    }
    let snap: WorldSnapshot = serde_json::from_slice(&data)?;
    if snap.version != SAVE_VERSION {
        bail!(
            "unsupported save version {} (expected v{SAVE_VERSION}; older v2 saves are preserved untouched on disk)",
            snap.version
        );
    }
    if snap.organisms.len() > MAX_ORGANISMS_BUFFER {
        bail!(
            "save contains {} organisms, exceeding safe limit {MAX_ORGANISMS_BUFFER}",
            snap.organisms.len()
        );
    }

    let mut world = ConvergenceWorld::from_universe(snap.seed, snap.universe, prefer_gpu);
    world.tick = snap.tick;
    world.rng = StdRng::seed_from_u64(snap.rng_state_seed);
    world.symbiote.environment = snap.environment;
    if snap.eco_cells.len() == ECO_WIDTH * ECO_HEIGHT * 4 {
        world.eco_field.cells = snap.eco_cells;
    }
    world.organisms = snap.organisms;
    for org in &mut world.organisms {
        org.body.ensure_runtime_buffers(&org.morph_genome);
        org.phenotype =
            AnatomyPhenotype::express(&org.genomorph, org.maturity, org.body.pulse_phase);
    }
    world.phylogeny = snap.phylogeny;
    world.next_org_id = snap.next_org_id;
    world.total_births = snap.total_births;
    world.total_deaths = snap.total_deaths;
    world.total_predations = snap.total_predations;
    Ok(world)
}

#[allow(dead_code)]
pub fn list_saved_worlds() -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(WORLDS_DIR) else {
        return Vec::new();
    };
    entries
        .filter_map(|e| e.ok().map(|d| d.path()))
        .filter(|p| p.to_string_lossy().ends_with(".abio.json"))
        .collect()
}
