//! Continuous 256x144x4 widescreen toroidal reaction-diffusion ecological field
//! with spatially heterogeneous procedural biomes.
//!
//! Channels (interleaved RGBA f32):
//! - 0: Nutrient (hydrothermal vents, reef zones, pelagic upwelling)
//! - 1: Pheromone / Social Signal
//! - 2: Toxin / Danger (predation plumes & volcanic trenches)
//! - 3: Living Mycelial / Root Network (organic sinuous filaments)

use crate::app::Environment;
use crate::automata::{CellKind, CellularAutomata};
use crate::convergence::camera::WorldBounds;
use crate::ecology::{Ecology, ZoneKind};
use rayon::prelude::*;

pub const ECO_WIDTH: usize = 256;
pub const ECO_HEIGHT: usize = 144;
pub const ECO_CHANNELS: usize = 4;

#[derive(Clone, Copy, Debug, Default)]
pub struct EcoSample {
    pub nutrient: f32,
    pub pheromone: f32,
    pub toxin: f32,
    #[allow(dead_code)]
    pub root: f32,
    pub nutrient_grad_x: f32,
    pub nutrient_grad_y: f32,
}

#[derive(Clone)]
pub struct EcoField {
    pub width: usize,
    pub height: usize,
    pub bounds: WorldBounds,
    /// Interleaved [nutrient, pheromone, toxin, root] per cell.
    pub cells: Vec<f32>,
    pub next: Vec<f32>,
    pub tick: u64,
}

impl EcoField {
    pub fn new(width: usize, height: usize, bounds: WorldBounds, seed: u64) -> Self {
        let len = width * height * ECO_CHANNELS;
        let mut cells = vec![0.0f32; len];

        // Seed-dependent heterogeneous biomes across the expansive 16:9 world:
        // - Hydrothermal nutrient vents (rich emerald/teal plumes)
        // - Toxic abyssal trenches (amber/crimson chemical barriers encouraging isolation)
        // - Sinuous mycelial reef filaments
        let phase = ((seed & 0xFFFF) as f32 / 65535.0) * std::f32::consts::TAU;
        let vents: [(f32, f32, f32); 7] = [
            (0.16, 0.28, 0.12),
            (0.42, 0.22, 0.14),
            (0.76, 0.26, 0.13),
            (0.28, 0.68, 0.15),
            (0.58, 0.56, 0.14),
            (0.84, 0.74, 0.13),
            (0.10, 0.82, 0.11),
        ];
        let trenches: [(f32, f32, f32); 3] = [
            (0.50, 0.34, 0.08),
            (0.35, 0.82, 0.07),
            (0.88, 0.45, 0.07),
        ];

        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * ECO_CHANNELS;
                let nx = x as f32 / width as f32;
                let ny = y as f32 / height as f32;

                let mut nutr = 0.16f32;
                for (vi, &(vx, vy, vr)) in vents.iter().enumerate() {
                    let svx = (vx + (phase + vi as f32 * 1.7).sin() * 0.06).rem_euclid(1.0);
                    let svy = (vy + (phase + vi as f32 * 2.3).cos() * 0.06).rem_euclid(1.0);
                    let dx = (nx - svx).abs().min(1.0 - (nx - svx).abs());
                    let dy = (ny - svy).abs().min(1.0 - (ny - svy).abs());
                    let d2 = dx * dx + dy * dy;
                    nutr += (-d2 / (2.0 * vr * vr)).exp() * 0.56;
                }

                let mut tox = 0.0f32;
                for (ti, &(tx, ty, tr)) in trenches.iter().enumerate() {
                    let stx = (tx + (phase + ti as f32 * 3.1).cos() * 0.05).rem_euclid(1.0);
                    let sty = (ty + (phase + ti as f32 * 1.9).sin() * 0.05).rem_euclid(1.0);
                    let dx = (nx - stx).abs().min(1.0 - (nx - stx).abs());
                    let dy = (ny - sty).abs().min(1.0 - (ny - sty).abs());
                    let d2 = dx * dx + dy * dy;
                    tox += (-d2 / (2.0 * tr * tr)).exp() * 0.28;
                }

                // Organic sinuous mycelial filaments unique across the wide ocean
                let warp = (nx * 18.85 + phase + (ny * 6.283).sin() * 1.6).sin()
                    * (ny * 12.566 + (nx * 9.424 + phase).cos() * 1.5).cos();
                let ridge = (1.0 - warp.abs()).powf(8.0) * 0.62 * (nutr * 1.2).clamp(0.2, 1.0);

                cells[idx] = nutr.clamp(0.08, 0.92);
                cells[idx + 1] = 0.0;
                cells[idx + 2] = tox.clamp(0.0, 0.45);
                cells[idx + 3] = ridge.clamp(0.0, 0.85);
            }
        }

        Self {
            width,
            height,
            bounds,
            next: cells.clone(),
            cells,
            tick: 0,
        }
    }

    #[inline]
    pub fn world_to_grid(&self, wx: f32, wy: f32) -> (usize, usize) {
        let u = (wx / self.bounds.width).rem_euclid(1.0);
        let v = (wy / self.bounds.height).rem_euclid(1.0);
        let gx = ((u * self.width as f32).floor() as usize).min(self.width - 1);
        let gy = ((v * self.height as f32).floor() as usize).min(self.height - 1);
        (gx, gy)
    }

    pub fn sample_world(&self, wx: f32, wy: f32) -> EcoSample {
        let (gx, gy) = self.world_to_grid(wx, wy);
        let idx = (gy * self.width + gx) * ECO_CHANNELS;

        let lx = (gx + self.width - 1) % self.width;
        let rx = (gx + 1) % self.width;
        let uy = (gy + self.height - 1) % self.height;
        let dy = (gy + 1) % self.height;

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
                let nx = (cx as i32 + ox).rem_euclid(self.width as i32) as usize;
                let ny = (cy as i32 + oy).rem_euclid(self.height as i32) as usize;
                let w = if ox == 0 && oy == 0 { 0.55 } else { 0.11 };
                let idx = (ny * self.width + nx) * ECO_CHANNELS;
                self.cells[idx] = (self.cells[idx] + d_nutrient * w).clamp(0.0, 1.0);
                self.cells[idx + 1] = (self.cells[idx + 1] + d_pheromone * w).clamp(0.0, 1.0);
                self.cells[idx + 2] = (self.cells[idx + 2] + d_toxin * w).clamp(0.0, 1.0);
                self.cells[idx + 3] = (self.cells[idx + 3] + d_root * w).clamp(0.0, 1.0);
            }
        }
    }

    pub fn couple_symbiote_ecology(&mut self, ecology: &Ecology, substrate: &CellularAutomata) {
        for zone in &ecology.zones {
            let wx = ((zone.x + 1.0) * 0.5).clamp(0.0, 1.0) * self.bounds.width;
            let wy = ((zone.y + 1.0) * 0.5).clamp(0.0, 1.0) * self.bounds.height;
            let (dn, dp, dt) = match zone.kind {
                ZoneKind::Nutrient => (0.016 * zone.strength, 0.002, -0.005),
                ZoneKind::Dead => (-0.008 * zone.strength, 0.0, 0.012 * zone.strength),
                ZoneKind::Mutagen => (0.003, 0.010 * zone.strength, 0.005 * zone.strength),
                ZoneKind::Nest => (0.010 * zone.strength, 0.012 * zone.strength, -0.004),
                ZoneKind::Turbulent => (0.002, 0.004, 0.002),
            };
            self.deposit_world(wx, wy, dn, dp, dt, 0.0);
        }

        if substrate.width > 0 && substrate.height > 0 {
            for sy in 0..substrate.height {
                for sx in 0..substrate.width {
                    let cell = substrate.cells[sy * substrate.width + sx];
                    if cell.kind == CellKind::Root {
                        let gx = (sx * self.width / substrate.width).min(self.width - 1);
                        let gy = (sy * self.height / substrate.height).min(self.height - 1);
                        let idx = (gy * self.width + gx) * ECO_CHANNELS;
                        self.cells[idx + 3] = (self.cells[idx + 3] + 0.02).clamp(0.0, 0.88);
                    }
                }
            }
        }
    }

    pub fn step_cpu(&mut self, env: Environment) {
        self.tick += 1;
        let w = self.width;
        let h = self.height;
        let src = &self.cells;

        let regen_rate = match env {
            Environment::Bloom => 0.0052,
            Environment::Calm => 0.0032,
            Environment::Drift => 0.0028,
            Environment::Storm => 0.0024,
            Environment::Hunger => 0.0012,
        };

        self.next
            .par_chunks_mut(w * ECO_CHANNELS)
            .enumerate()
            .for_each(|(y, row)| {
                let y_up = (y + h - 1) % h;
                let y_dn = (y + 1) % h;

                for x in 0..w {
                    let x_lt = (x + w - 1) % w;
                    let x_rt = (x + 1) % w;

                    let c_idx = (y * w + x) * ECO_CHANNELS;
                    let u_idx = (y_up * w + x) * ECO_CHANNELS;
                    let d_idx = (y_dn * w + x) * ECO_CHANNELS;
                    let l_idx = (y * w + x_lt) * ECO_CHANNELS;
                    let r_idx = (y * w + x_rt) * ECO_CHANNELS;

                    let n = src[c_idx];
                    let p = src[c_idx + 1];
                    let t = src[c_idx + 2];
                    let root = src[c_idx + 3];

                    let lap_n = (src[u_idx] + src[d_idx] + src[l_idx] + src[r_idx]) * 0.25 - n;
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
                    let lap_r = (src[u_idx + 3]
                        + src[d_idx + 3]
                        + src[l_idx + 3]
                        + src[r_idx + 3])
                        * 0.25
                        - root;

                    let next_root = (root + lap_r * 0.03) * 0.9996;
                    let next_n = (n + lap_n * 0.20 + regen_rate * (0.72 - n)
                        + root * 0.007 * (1.0 - n)
                        - t * 0.004)
                        .clamp(0.0, 1.0);
                    let next_p = (p + lap_p * 0.24) * 0.968;
                    let next_t = (t + lap_t * 0.16) * 0.964;

                    let out = x * ECO_CHANNELS;
                    row[out] = next_n;
                    row[out + 1] = next_p.clamp(0.0, 1.0);
                    row[out + 2] = next_t.clamp(0.0, 1.0);
                    row[out + 3] = next_root.clamp(0.0, 1.0);
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
