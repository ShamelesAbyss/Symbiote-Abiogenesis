//! Localized high-resolution continuous cellular bodies (Lenia morphology).
//!
//! Each evolving organism owns a compact continuous 2-channel cellular field
//! (Channel 0 = Cytoplasm / Structural Mass, Channel 1 = Membrane / Cortex)
//! governed by heritable Lenia kernel & growth parameters coupled to Symbiote's
//! ecological genome and local environmental fields.
//!
//! Lenia kernel shell and Gaussian growth equations adapted from
//! Bert Chan (2018, MIT License) and Primordia (ShamelesAbyss).

use crate::particle::{Genome, Tribe};
use rand::{rngs::StdRng, Rng};
use serde::{Deserialize, Serialize};
use std::f32::consts::TAU;

/// Resolution of each organism's localized continuous cellular field.
/// 24x24x2 = 1,152 f32s (~4.6 KB per organism), enabling hundreds of
/// high-resolution continuous organisms within a tiny fraction of 8 GB VRAM.
pub const BODY_RES: usize = 24;
pub const BODY_CHANNELS: usize = 2;
pub const BODY_CELLS: usize = BODY_RES * BODY_RES * BODY_CHANNELS;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShellKind {
    Exponential,
    Polynomial,
}

/// Heritable continuous morphological genome.
/// Directly influences kernel shape, growth dynamics, membrane formation,
/// locomotion asymmetry, nutrient sensitivity, and regeneration.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct MorphGenome {
    /// Kernel radius in local body cells (3..=7).
    pub kernel_radius: u32,
    /// Primary and secondary radial ring peak weights.
    pub ring_peaks: [f32; 3],
    /// Radial kernel shell function.
    pub shell: ShellKind,
    /// Angular symmetry lobes (1.0 = bilateral glider, 2.0..6.0 = radial/medusa/reef).
    pub angular_lobes: f32,
    /// Directional asymmetry driving emergent locomotion (0.0..0.45).
    pub asymmetry: f32,
    /// Growth function center (mu).
    pub mu: f32,
    /// Growth function width (sigma).
    pub sigma: f32,
    /// Integration step rate (dt).
    pub dt: f32,
    /// Membrane formation coupling strength (Channel 0 -> Channel 1 boundary).
    pub membrane_affinity: f32,
    /// Nutrient-to-growth coupling sensitivity.
    pub nutrient_sensitivity: f32,
    /// Cellular regeneration strength when damaged.
    pub regeneration: f32,
    /// Bioluminescent hue shift (0.0..1.0) inherited and mutated along lineages.
    pub biolum_hue: f32,
}

impl MorphGenome {
    /// Derive a viable morphological genome from Symbiote's ecological Genome and Tribe.
    pub fn from_ecological(eco: Genome, tribe: Tribe, rng: &mut StdRng) -> Self {
        let (base_mu, base_sigma, base_lobes, base_asym, hue) = match tribe {
            Tribe::Blood => (0.165, 0.036, 1.0, 0.28, 0.02),
            Tribe::Moss => (0.195, 0.048, 3.0, 0.08, 0.34),
            Tribe::Deep => (0.175, 0.042, 4.0, 0.14, 0.58),
            Tribe::Solar => (0.185, 0.045, 5.0, 0.16, 0.14),
            Tribe::Dream => (0.160, 0.038, 2.0, 0.24, 0.82),
            Tribe::Static => (0.180, 0.040, 6.0, 0.18, 0.48),
        };

        let kernel_radius = (4.0 + eco.perception * 6.0).round().clamp(3.0, 7.0) as u32;
        let ring_peaks = [
            (0.85 + eco.bonding * 0.25).clamp(0.3, 1.4),
            (0.25 + eco.membrane * 0.45).clamp(0.0, 1.0),
            (0.10 + eco.orbit * 0.30).clamp(0.0, 0.8),
        ];

        Self {
            kernel_radius,
            ring_peaks,
            shell: if eco.membrane > 0.55 {
                ShellKind::Exponential
            } else {
                ShellKind::Polynomial
            },
            angular_lobes: (base_lobes as f32 + rng.gen_range(-0.5_f32..0.5_f32)).clamp(1.0, 6.0),
            asymmetry: (base_asym + eco.volatility * 0.08 + rng.gen_range(-0.04_f32..0.04_f32))
                .clamp(0.02, 0.42),
            mu: (base_mu + (eco.metabolism - 0.04) * 0.4 + rng.gen_range(-0.015_f32..0.015_f32))
                .clamp(0.11, 0.28),
            sigma: (base_sigma + eco.fertility * 0.012 + rng.gen_range(-0.005_f32..0.005_f32))
                .clamp(0.020, 0.075),
            dt: 0.11,
            membrane_affinity: (0.45 + eco.membrane * 0.45).clamp(0.2, 1.2),
            nutrient_sensitivity: (0.65 + eco.hunger * 10.0).clamp(0.3, 1.5),
            regeneration: (0.35 + eco.fertility * 0.45).clamp(0.15, 1.0),
            biolum_hue: (hue as f32 + rng.gen_range(-0.04_f32..0.04_f32)).rem_euclid(1.0),
        }
    }

    /// Produce a mutated offspring morphological genome.
    pub fn mutate(&self, rng: &mut StdRng, rate: f32) -> Self {
        let mut child = *self;
        let r = rate.clamp(0.005, 0.25);

        child.mu = (child.mu + rng.gen_range(-r..r) * 0.18).clamp(0.10, 0.30);
        child.sigma = (child.sigma + rng.gen_range(-r..r) * 0.08).clamp(0.018, 0.085);
        child.asymmetry = (child.asymmetry + rng.gen_range(-r..r) * 0.25).clamp(0.01, 0.45);
        child.angular_lobes =
            (child.angular_lobes + rng.gen_range(-r..r) * 2.0).clamp(1.0, 6.0);
        child.membrane_affinity =
            (child.membrane_affinity + rng.gen_range(-r..r) * 0.4).clamp(0.15, 1.35);
        child.nutrient_sensitivity =
            (child.nutrient_sensitivity + rng.gen_range(-r..r) * 0.4).clamp(0.25, 1.6);
        child.regeneration = (child.regeneration + rng.gen_range(-r..r) * 0.3).clamp(0.1, 1.0);
        child.biolum_hue = (child.biolum_hue + rng.gen_range(-r..r) * 0.15).rem_euclid(1.0);

        for peak in &mut child.ring_peaks {
            *peak = (*peak + rng.gen_range(-r..r) * 0.35).clamp(0.0, 1.5);
        }

        if rng.gen_bool((r * 0.5) as f64) {
            let delta: i32 = if rng.gen_bool(0.5) { 1 } else { -1 };
            child.kernel_radius = (child.kernel_radius as i32 + delta).clamp(3, 7) as u32;
        }

        child
    }

    /// Genetic distance between two morphological genomes for speciation & phylogeny.
    pub fn distance(&self, other: &Self) -> f32 {
        let d_mu = (self.mu - other.mu).abs() * 6.0;
        let d_sigma = (self.sigma - other.sigma).abs() * 14.0;
        let d_asym = (self.asymmetry - other.asymmetry).abs() * 2.5;
        let d_lobes = (self.angular_lobes - other.angular_lobes).abs() * 0.35;
        let d_mem = (self.membrane_affinity - other.membrane_affinity).abs() * 0.8;
        let d_peaks = (self.ring_peaks[0] - other.ring_peaks[0]).abs()
            + (self.ring_peaks[1] - other.ring_peaks[1]).abs();
        d_mu + d_sigma + d_asym + d_lobes + d_mem + d_peaks * 0.5
    }
}

/// Precomputed radial + angular Lenia kernel for a MorphGenome.
#[derive(Clone, Debug)]
pub struct MorphKernel {
    pub radius: i32,
    pub taps: Vec<(i32, i32, f32)>,
}

impl MorphKernel {
    pub fn build(genome: &MorphGenome, heading: f32) -> Self {
        let radius = genome.kernel_radius.clamp(3, 7) as i32;
        let mut taps = Vec::with_capacity(((2 * radius + 1) * (2 * radius + 1)) as usize);
        let mut sum = 0.0f32;

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                let r = dist / radius as f32;
                if r >= 1.0 {
                    continue;
                }

                let band_pos = r * 3.0;
                let band = (band_pos.floor() as usize).min(2);
                let local_r = band_pos.fract();
                let shell_v = kernel_shell(genome.shell, local_r);
                let peak = genome.ring_peaks[band];

                let angle = (dy as f32).atan2(dx as f32) - heading;
                let angular = 1.0 + genome.asymmetry * (angle * genome.angular_lobes).cos();
                let directional = 1.0 + genome.asymmetry * 0.45 * angle.cos();

                let w = (peak * shell_v * angular * directional).max(0.0);
                if w > 0.0005 {
                    taps.push((dx, dy, w));
                    sum += w;
                }
            }
        }

        if sum > 0.0001 {
            for tap in &mut taps {
                tap.2 /= sum;
            }
        }

        Self { radius, taps }
    }
}

#[inline]
pub fn kernel_shell(shell: ShellKind, r: f32) -> f32 {
    let r = r.clamp(0.0, 1.0);
    let k = 4.0 * r * (1.0 - r);
    if k <= 0.0 {
        return 0.0;
    }
    match shell {
        ShellKind::Exponential => (4.0 * (1.0 - 1.0 / k)).exp(),
        ShellKind::Polynomial => k.powi(4),
    }
}

#[inline]
pub fn growth_gaussian(u: f32, mu: f32, sigma: f32) -> f32 {
    let s = sigma.max(0.001);
    let d = u - mu;
    2.0 * (-(d * d) / (2.0 * s * s)).exp() - 1.0
}

/// Emergent physical measurements extracted from a continuous cellular body.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct BodyMetrics {
    /// Total cytoplasm mass (normalized 0.0..1.0).
    pub mass: f32,
    /// Total membrane integrity (normalized 0.0..1.0).
    pub membrane_mass: f32,
    /// Emergent center-of-mass offset from body center (-1.0..1.0).
    pub com_dx: f32,
    pub com_dy: f32,
    /// Structural elongation (0.0 = circular, 1.0 = strongly elongated / ready for mitosis).
    pub elongation: f32,
    /// Temporal change rate (morphological vitality / pulsation).
    pub activity: f32,
    /// Effective bounding radius in local normalized coordinates.
    pub radius_norm: f32,
}

/// Environmental inputs felt by a localized cellular body during one step.
#[derive(Clone, Copy, Debug, Default)]
pub struct LocalEnvInput {
    pub nutrient: f32,
    pub pheromone: f32,
    pub toxin: f32,
    pub nutrient_grad_x: f32,
    pub nutrient_grad_y: f32,
    pub mutagen: f32,
}

/// A localized continuous 2-channel Lenia cellular body.
#[derive(Clone, Debug)]
pub struct LeniaBody {
    /// Interleaved [cytoplasm, membrane] per cell, length = BODY_CELLS.
    pub cells: Vec<f32>,
    pub next: Vec<f32>,
    pub kernel: MorphKernel,
    pub metrics: BodyMetrics,
    pub heading: f32,
    pub pulse_phase: f32,
}

impl LeniaBody {
    pub fn new(genome: &MorphGenome, heading: f32, rng: &mut StdRng) -> Self {
        let mut cells = vec![0.0f32; BODY_CELLS];
        let center = (BODY_RES as f32 - 1.0) * 0.5;
        let seed_r = (BODY_RES as f32) * 0.24;

        for y in 0..BODY_RES {
            for x in 0..BODY_RES {
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let d = (dx * dx + dy * dy).sqrt();
                let angle = dy.atan2(dx) - heading;
                let lobe_mod = 1.0 + 0.18 * (angle * genome.angular_lobes).cos();
                let r_norm = d / (seed_r * lobe_mod).max(1.0);

                if r_norm < 1.35 {
                    let core = (-r_norm * r_norm * 1.4).exp();
                    let jitter = rng.gen_range(-0.06..0.06);
                    let cyto = (core + jitter).clamp(0.0, 1.0);
                    let ring = (-(r_norm - 0.85).powi(2) / 0.14).exp() * genome.membrane_affinity;
                    let idx = (y * BODY_RES + x) * BODY_CHANNELS;
                    cells[idx] = cyto;
                    cells[idx + 1] = ring.clamp(0.0, 1.0);
                }
            }
        }

        let kernel = MorphKernel::build(genome, heading);
        let mut body = Self {
            next: cells.clone(),
            cells,
            kernel,
            metrics: BodyMetrics::default(),
            heading,
            pulse_phase: rng.gen_range(0.0..TAU),
        };
        body.recompute_metrics(0.0);
        body
    }

    /// Step the localized continuous cellular body on CPU, bidirectionally coupled
    /// to local environmental nutrients, toxins, and chemotactic gradients.
    pub fn step_cpu(&mut self, genome: &MorphGenome, env: LocalEnvInput) -> BodyMetrics {
        self.pulse_phase = (self.pulse_phase + 0.16) % TAU;

        // Nutrient availability shifts effective growth center and growth rate;
        // toxins depress growth and erode unprotected cytoplasm.
        let nutrient_boost = (env.nutrient * genome.nutrient_sensitivity).clamp(0.0, 1.2);
        let toxin_stress = env.toxin.clamp(0.0, 1.0);
        let eff_mu = (genome.mu - nutrient_boost * 0.018 + toxin_stress * 0.022).clamp(0.08, 0.34);
        let eff_dt = genome.dt * (0.72 + nutrient_boost * 0.55);

        // Mild chemotactic bias along local nutrient gradient
        let grad_mag = (env.nutrient_grad_x * env.nutrient_grad_x
            + env.nutrient_grad_y * env.nutrient_grad_y)
            .sqrt();
        let (gx, gy) = if grad_mag > 0.0001 {
            (env.nutrient_grad_x / grad_mag, env.nutrient_grad_y / grad_mag)
        } else {
            (0.0, 0.0)
        };

        let center = (BODY_RES as f32 - 1.0) * 0.5;
        let max_r = (BODY_RES as f32) * 0.48;
        let mut total_delta = 0.0f32;

        for y in 0..BODY_RES {
            for x in 0..BODY_RES {
                let mut u = 0.0f32;
                for &(dx, dy, w) in &self.kernel.taps {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && ny >= 0 && nx < BODY_RES as i32 && ny < BODY_RES as i32 {
                        let nidx = (ny as usize * BODY_RES + nx as usize) * BODY_CHANNELS;
                        u += self.cells[nidx] * w;
                    }
                }

                let idx = (y * BODY_RES + x) * BODY_CHANNELS;
                let cyto = self.cells[idx];
                let mem = self.cells[idx + 1];

                // Soft circular boundary containment so the organism stays cohesive
                let rx = x as f32 - center;
                let ry = y as f32 - center;
                let dist = (rx * rx + ry * ry).sqrt();
                let boundary_fade = ((max_r - dist) / 2.5).clamp(0.0, 1.0);

                // Chemotactic polarization: cells on the nutrient-facing side grow slightly faster
                let chemo = if dist > 0.5 {
                    ((rx * gx + ry * gy) / dist) * grad_mag.min(1.0) * 0.14
                } else {
                    0.0
                };

                let g = growth_gaussian(u, eff_mu, genome.sigma) + chemo;
                // Membrane shields cytoplasm from toxin damage
                let toxin_loss = toxin_stress * 0.065 * (1.0 - mem * 0.65);
                // Homeostatic regeneration helps recover small wounds when nutrients are present
                let regen = if cyto > 0.05 && cyto < 0.45 && nutrient_boost > 0.25 {
                    genome.regeneration * 0.015 * nutrient_boost
                } else {
                    0.0
                };

                let next_cyto = ((cyto + eff_dt * g + regen - toxin_loss) * boundary_fade)
                    .clamp(0.0, 1.0);

                // Channel 1 (Membrane): forms organically at the gradient boundary of cytoplasm
                // where cyto is moderate (0.15..0.65) and local potential u is lower than core.
                let edge_signal = (4.0 * next_cyto * (1.0 - next_cyto)).clamp(0.0, 1.0);
                let target_mem =
                    (edge_signal * genome.membrane_affinity * (0.65 + 0.35 * (1.0 - u)))
                        .clamp(0.0, 1.0);
                let next_mem = (mem * 0.80 + target_mem * 0.20) * boundary_fade;

                total_delta += (next_cyto - cyto).abs();
                self.next[idx] = next_cyto;
                self.next[idx + 1] = next_mem.clamp(0.0, 1.0);
            }
        }

        std::mem::swap(&mut self.cells, &mut self.next);
        self.recenter_and_measure(total_delta / (BODY_RES * BODY_RES) as f32)
    }

    /// Re-center the local field when the organism's continuous center of mass drifts,
    /// returning the emergent displacement so the world-space organism translates fluidly.
    fn recenter_and_measure(&mut self, activity: f32) -> BodyMetrics {
        self.recompute_metrics(activity);

        // If center of mass drifted by >= 1 cell, shift the local grid back by 1 cell
        // so the organism never clips its own local bounding box.
        let shift_x = if self.metrics.com_dx > 0.09 {
            1
        } else if self.metrics.com_dx < -0.09 {
            -1
        } else {
            0
        };
        let shift_y = if self.metrics.com_dy > 0.09 {
            1
        } else if self.metrics.com_dy < -0.09 {
            -1
        } else {
            0
        };

        if shift_x != 0 || shift_y != 0 {
            self.next.fill(0.0);
            for y in 0..BODY_RES {
                for x in 0..BODY_RES {
                    let sx = x as i32 + shift_x;
                    let sy = y as i32 + shift_y;
                    if sx >= 0 && sy >= 0 && sx < BODY_RES as i32 && sy < BODY_RES as i32 {
                        let dst = (y * BODY_RES + x) * BODY_CHANNELS;
                        let src = (sy as usize * BODY_RES + sx as usize) * BODY_CHANNELS;
                        self.next[dst] = self.cells[src];
                        self.next[dst + 1] = self.cells[src + 1];
                    }
                }
            }
            std::mem::swap(&mut self.cells, &mut self.next);
        }

        self.metrics
    }

    pub fn recompute_metrics(&mut self, activity: f32) {
        let center = (BODY_RES as f32 - 1.0) * 0.5;
        let mut mass_sum = 0.0f32;
        let mut mem_sum = 0.0f32;
        let mut wx = 0.0f32;
        let mut wy = 0.0f32;

        for y in 0..BODY_RES {
            for x in 0..BODY_RES {
                let idx = (y * BODY_RES + x) * BODY_CHANNELS;
                let c = self.cells[idx];
                let m = self.cells[idx + 1];
                mass_sum += c;
                mem_sum += m;
                wx += (x as f32 - center) * c;
                wy += (y as f32 - center) * c;
            }
        }

        let area = (BODY_RES * BODY_RES) as f32;
        let (com_dx, com_dy) = if mass_sum > 0.001 {
            (
                (wx / mass_sum) / (BODY_RES as f32 * 0.5),
                (wy / mass_sum) / (BODY_RES as f32 * 0.5),
            )
        } else {
            (0.0, 0.0)
        };

        // Compute second central moments for elongation & radius
        let cx = center + com_dx * (BODY_RES as f32 * 0.5);
        let cy = center + com_dy * (BODY_RES as f32 * 0.5);
        let mut mxx = 0.0f32;
        let mut myy = 0.0f32;
        let mut mxy = 0.0f32;

        if mass_sum > 0.001 {
            for y in 0..BODY_RES {
                for x in 0..BODY_RES {
                    let c = self.cells[(y * BODY_RES + x) * BODY_CHANNELS];
                    if c > 0.02 {
                        let dx = x as f32 - cx;
                        let dy = y as f32 - cy;
                        mxx += dx * dx * c;
                        myy += dy * dy * c;
                        mxy += dx * dy * c;
                    }
                }
            }
            mxx /= mass_sum;
            myy /= mass_sum;
            mxy /= mass_sum;
        }

        let trace = mxx + myy;
        let disc = ((mxx - myy) * (mxx - myy) + 4.0 * mxy * mxy).sqrt();
        let lam1 = (trace + disc) * 0.5;
        let lam2 = ((trace - disc) * 0.5).max(0.0001);
        let elongation = if lam1 > 0.01 {
            (1.0 - (lam2 / lam1).sqrt()).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let radius_norm = (trace.sqrt() / (BODY_RES as f32 * 0.5)).clamp(0.05, 1.0);

        self.metrics = BodyMetrics {
            mass: (mass_sum / area).clamp(0.0, 1.0),
            membrane_mass: (mem_sum / area).clamp(0.0, 1.0),
            com_dx: com_dx.clamp(-1.0, 1.0),
            com_dy: com_dy.clamp(-1.0, 1.0),
            elongation,
            activity,
            radius_norm,
        };
    }

    /// Mitotic division: splits this continuous cellular body into a daughter body
    /// along its minor axis, halving mass cleanly between parent and offspring.
    pub fn divide(&mut self, daughter_genome: &MorphGenome, rng: &mut StdRng) -> Self {
        let mut daughter_cells = vec![0.0f32; BODY_CELLS];
        let half = BODY_RES / 2;

        for y in 0..BODY_RES {
            for x in 0..BODY_RES {
                let idx = (y * BODY_RES + x) * BODY_CHANNELS;
                if x >= half {
                    // Transfer right half to daughter, shifted toward daughter's center
                    let dx = (x - half / 2).min(BODY_RES - 1);
                    let didx = (y * BODY_RES + dx) * BODY_CHANNELS;
                    daughter_cells[didx] = (self.cells[idx] * 0.92).clamp(0.0, 1.0);
                    daughter_cells[didx + 1] = (self.cells[idx + 1] * 0.92).clamp(0.0, 1.0);
                    self.cells[idx] *= 0.25;
                    self.cells[idx + 1] *= 0.25;
                } else {
                    self.cells[idx] = (self.cells[idx] * 0.92).clamp(0.0, 1.0);
                    self.cells[idx + 1] = (self.cells[idx + 1] * 0.92).clamp(0.0, 1.0);
                }
            }
        }

        let heading = (self.heading + rng.gen_range(-0.6..0.6)).rem_euclid(TAU);
        let kernel = MorphKernel::build(daughter_genome, heading);
        self.recompute_metrics(0.05);

        let mut daughter = Self {
            next: daughter_cells.clone(),
            cells: daughter_cells,
            kernel,
            metrics: BodyMetrics::default(),
            heading,
            pulse_phase: rng.gen_range(0.0..TAU),
        };
        daughter.recompute_metrics(0.05);
        daughter
    }

    /// Inflict physical damage (e.g., from predation or collision) to the body's cytoplasm.
    pub fn apply_damage(&mut self, fraction: f32) {
        let retain = (1.0 - fraction.clamp(0.0, 0.95)).max(0.05);
        for c in &mut self.cells {
            *c *= retain;
        }
        self.recompute_metrics(fraction);
    }
}
