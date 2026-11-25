//! Continuous 4-channel reaction-diffusion planetary & regional ecological field.
//!
//! Channels (interleaved RGBA f32):
//! - 0: Nutrient (diffuses, regenerates from roots/zones, consumed by organisms, replenished on death)
//! - 1: Pheromone / Growth signal (secreted by thriving organisms & colonies)
//! - 2: Toxin / Danger (secreted during predation/stress, inhibits cellular growth)
//! - 3: Living Root / Mycelial Network (persistent nutrient-conducting filaments)

use crate::app::Environment;
use crate::automata::{CellKind, CellularAutomata};
use crate::ecology::{Ecology, ZoneKind};
use rayon::prelude::*;

pub const ECO_WIDTH: usize = 128;
pub const ECO_HEIGHT: usize = 128;
pub const ECO_CHANNELS: usize = 4;

#[derive(Clone, Copy, Debug, Default)]
pub struct EcoSample {
    pub nutrient: f32,
    pub pheromone: f32,
    pub toxin: f32,
    pub root: f32,
    pub nutrient_grad_x: f32,
    pub nutrient_grad_y: f32,
}

#[derive(Clone)]
pub struct EcoField {
    pub width: usize,
    pub height: usize,
    /// Interleaved [nutrient, pheromone, toxin, root] per cell.
    pub cells: Vec<f32>,
    pub next: Vec<f32>,
    pub tick: u64,
}

impl EcoField {
    pub fn new(width: usize, height: usize) -> Self {
        let len = width * height * ECO_CHANNELS;
        let mut cells = vec![0.0f32; len];

        // Initialize with organic nutrient gradient and bottom-seeded root tendrils
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * ECO_CHANNELS;
                let nx = x as f32 / width as f32;
                let ny = y as f32 / height as f32;
                let wave = ((nx * 9.0).sin() * (ny * 7.0).cos() * 0.5 + 0.5) * 0.45 + 0.25;
                cells[idx] = wave.clamp(0.15, 0.85);
                cells[idx + 1] = 0.0;
                cells[idx + 2] = 0.0;
                // Bottom-seeded living root filaments
                let root_seed = if ny > 0.82 && ((nx * 19.0).sin().abs() > 0.72) {
                    (ny - 0.82) * 4.5
                } else {
                    0.0
                };
                cells[idx + 3] = root_seed.clamp(0.0, 1.0);
            }
        }

        Self {
            width,
            height,
            next: cells.clone(),
            cells,
            tick: 0,
        }
    }

    #[inline]
    pub fn world_to_grid(&self, wx: f32, wy: f32) -> (usize, usize) {
        let gx = (((wx + 1.0) * 0.5) * self.width as f32)
            .clamp(0.0, (self.width - 1) as f32) as usize;
        let gy = (((wy + 1.0) * 0.5) * self.height as f32)
            .clamp(0.0, (self.height - 1) as f32) as usize;
        (gx, gy)
    }

    pub fn sample_world(&self, wx: f32, wy: f32) -> EcoSample {
        let (gx, gy) = self.world_to_grid(wx, wy);
        let idx = (gy * self.width + gx) * ECO_CHANNELS;

        let lx = gx.saturating_sub(1);
        let rx = (gx + 1).min(self.width - 1);
        let uy = gy.saturating_sub(1);
        let dy = (gy + 1).min(self.height - 1);

        let n_left = self.cells[(gy * self.width + lx) * ECO_CHANNELS];
        let n_right = self.cells[(gy * self.width + rx) * ECO_CHANNELS];
        let n_up = self.cells[(uy * self.width + gx) * ECO_CHANNELS];
        let n_down = self.cells[(dy * self.width + gx) * ECO_CHANNELS];

        EcoSample {
            nutrient: self.cells[idx],
            pheromone: self.cells[idx + 1],
            toxin: self.cells[idx + 2],
            root: self.cells[idx + 3],
            nutrient_grad_x: (n_right - n_left) * 0.5,
            nutrient_grad_y: (n_down - n_up) * 0.5,
        }
    }

    /// Deposit or consume resources in a small smooth radius around world position (wx, wy).
    pub fn deposit_world(
        &mut self,
        wx: f32,
        wy: f32,
        d_nutrient: f32,
        d_pheromone: f32,
        d_toxin: f32,
        d_root: f32,
    ) {
        let (cx, cy) = self.world_to_grid(wx, wy);
        for oy in -1i32..=1 {
            for ox in -1i32..=1 {
                let nx = cx as i32 + ox;
                let ny = cy as i32 + oy;
                if nx < 0 || ny < 0 || nx >= self.width as i32 || ny >= self.height as i32 {
                    continue;
                }
                let w = if ox == 0 && oy == 0 { 0.6 } else { 0.1 };
                let idx = (ny as usize * self.width + nx as usize) * ECO_CHANNELS;
                self.cells[idx] = (self.cells[idx] + d_nutrient * w).clamp(0.0, 1.0);
                self.cells[idx + 1] = (self.cells[idx + 1] + d_pheromone * w).clamp(0.0, 1.0);
                self.cells[idx + 2] = (self.cells[idx + 2] + d_toxin * w).clamp(0.0, 1.0);
                self.cells[idx + 3] = (self.cells[idx + 3] + d_root * w).clamp(0.0, 1.0);
            }
        }
    }

    /// Synchronize Symbiote's discrete Ecology zones and CellularAutomata roots
    /// into the continuous reaction-diffusion substrate.
    pub fn couple_symbiote_ecology(&mut self, ecology: &Ecology, substrate: &CellularAutomata) {
        for zone in &ecology.zones {
            let (dn, dp, dt) = match zone.kind {
                ZoneKind::Nutrient => (0.015 * zone.strength, 0.002, -0.005),
                ZoneKind::Dead => (-0.008 * zone.strength, 0.0, 0.014 * zone.strength),
                ZoneKind::Mutagen => (0.003, 0.010 * zone.strength, 0.006 * zone.strength),
                ZoneKind::Nest => (0.010 * zone.strength, 0.012 * zone.strength, -0.004),
                ZoneKind::Turbulent => (0.002, 0.004, 0.002),
            };
            self.deposit_world(zone.x, zone.y, dn, dp, dt, 0.0);
        }

        // Couple Symbiote's living root cells into Channel 3 (Root network)
        if substrate.width > 0 && substrate.height > 0 {
            for sy in 0..substrate.height {
                for sx in 0..substrate.width {
                    let cell = substrate.cells[sy * substrate.width + sx];
                    if cell.kind == CellKind::Root {
                        let gx = (sx * self.width / substrate.width).min(self.width - 1);
                        let gy = (sy * self.height / substrate.height).min(self.height - 1);
                        let idx = (gy * self.width + gx) * ECO_CHANNELS;
                        self.cells[idx + 3] = (self.cells[idx + 3] + 0.08).clamp(0.0, 1.0);
                    }
                }
            }
        }
    }

    /// Advance the continuous reaction-diffusion field by one step on CPU (parallelized via Rayon).
    pub fn step_cpu(&mut self, env: Environment) {
        self.tick += 1;
        let w = self.width;
        let h = self.height;
        let src = &self.cells;

        let regen_rate = match env {
            Environment::Bloom => 0.0065,
            Environment::Calm => 0.0038,
            Environment::Drift => 0.0032,
            Environment::Storm => 0.0028,
            Environment::Hunger => 0.0014,
        };

        self.next
            .par_chunks_mut(w * ECO_CHANNELS)
            .enumerate()
            .for_each(|(y, row)| {
                let y_up = y.saturating_sub(1);
                let y_dn = (y + 1).min(h - 1);

                for x in 0..w {
                    let x_lt = x.saturating_sub(1);
                    let x_rt = (x + 1).min(w - 1);

                    let c_idx = (y * w + x) * ECO_CHANNELS;
                    let u_idx = (y_up * w + x) * ECO_CHANNELS;
                    let d_idx = (y_dn * w + x) * ECO_CHANNELS;
                    let l_idx = (y * w + x_lt) * ECO_CHANNELS;
                    let r_idx = (y * w + x_rt) * ECO_CHANNELS;

                    let n = src[c_idx];
                    let p = src[c_idx + 1];
                    let t = src[c_idx + 2];
                    let root = src[c_idx + 3];

                    let lap_n =
                        (src[u_idx] + src[d_idx] + src[l_idx] + src[r_idx]) * 0.25 - n;
                    let lap_p = (src[u_idx + 1]
                        + src[d_idx + 1]
                        + src[l_idx + 1]
                        + src[r_idx + 1])
                        * 0.25
                        - p;
                    let lap_t = (src[u_idx + 2]
                        + src[d_idx + 2]
                        + src[l_idx + 2]
                        + src[r_idx + 2])
                        * 0.25
                        - t;

                    // Living roots conduct nutrients upward and regenerate local nutrients
                    let root_below = src[d_idx + 3];
                    let root_grow = if root_below > 0.35 && n > 0.22 && t < 0.35 {
                        0.0025 * root_below * n
                    } else {
                        0.0
                    };
                    let next_root = (root * 0.9992 + root_grow).clamp(0.0, 1.0);

                    // Reaction-diffusion equations:
                    // Nutrient diffuses, regenerates near roots, is degraded by toxin
                    let next_n = (n + lap_n * 0.18 + regen_rate * (1.0 - n)
                        + root * 0.008 * (1.0 - n)
                        - t * 0.004)
                        .clamp(0.0, 1.0);

                    // Pheromone diffuses and decays smoothly
                    let next_p = (p + lap_p * 0.22) * 0.975;

                    // Toxin diffuses slowly and decays
                    let next_t = (t + lap_t * 0.14) * 0.968;

                    let out = x * ECO_CHANNELS;
                    row[out] = next_n;
                    row[out + 1] = next_p.clamp(0.0, 1.0);
                    row[out + 2] = next_t.clamp(0.0, 1.0);
                    row[out + 3] = next_root;
                }
            });

        std::mem::swap(&mut self.cells, &mut self.next);
    }

    pub fn mean_nutrient(&self) -> f32 {
        let mut sum = 0.0f32;
        let count = (self.width * self.height).max(1);
        for i in 0..count {
            sum += self.cells[i * ECO_CHANNELS];
        }
        sum / count as f32
    }
}
