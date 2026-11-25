//! Localized 3-channel continuous cellular morphogenesis (Multichannel Lenia)
//! + Primitive Adaptive Intelligence (heritable recurrent neural controller).
//!
//! Channels per cell (24x24x3 = 1,728 f32s = 6.9 KB per organism):
//! - Ch 0: Cytoplasm / Structural Mass (primary Lenia self-organization)
//! - Ch 1: Organic Membrane / Cortex (forms at cytoplasm gradient boundaries, shields toxins)
//! - Ch 2: Organelle / Motility / Bioluminescence (cross-coupled internal differentiation)
//!
//! Lenia kernel shell and Gaussian growth equations adapted with attribution from
//! Bert Chan (2018, MIT License) and Primordia (ShamelesAbyss).

use crate::particle::{Genome, Tribe};
use rand::{rngs::StdRng, Rng};
use serde::{Deserialize, Serialize};
use std::f32::consts::TAU;

pub const BODY_RES: usize = 24;
pub const BODY_CHANNELS: usize = 3;
pub const BODY_CELLS: usize = BODY_RES * BODY_RES * BODY_CHANNELS;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShellKind {
    Exponential,
    Polynomial,
}

/// Heritable continuous morphological genome.
/// Controls multichannel Lenia kernels, symmetry breaking, membrane physics,
/// organelle differentiation, chemotaxis, and bioluminescent pigmentation.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct MorphGenome {
    pub kernel_radius: u32,
    pub ring_peaks: [f32; 3],
    pub shell: ShellKind,
    /// Angular symmetry lobes (1 = bilateral swimmer, 2..6 = medusa/star/radial).
    pub angular_lobes: f32,
    /// Directional asymmetry driving emergent locomotion & symmetry breaking.
    pub asymmetry: f32,
    /// Spiral twist factor in kernel angle (creates vortex/spiral morphologies).
    pub spiral_twist: f32,
    /// Primary cytoplasm growth center & width.
    pub mu: f32,
    pub sigma: f32,
    /// Secondary organelle (Ch 2) growth center & coupling weight.
    pub org_mu: f32,
    pub org_coupling: f32,
    pub dt: f32,
    /// Membrane formation affinity (Ch 0 boundary -> Ch 1).
    pub membrane_affinity: f32,
    /// Nutrient-to-growth coupling sensitivity.
    pub nutrient_sensitivity: f32,
    /// Homeostatic regeneration rate when damaged.
    pub regeneration: f32,
    /// Bioluminescent base hue (0.0..1.0) + secondary organelle hue shift.
    pub biolum_hue: f32,
    pub accent_shift: f32,
}

impl MorphGenome {
    pub fn from_ecological(eco: Genome, tribe: Tribe, rng: &mut StdRng) -> Self {
        let (base_mu, base_sigma, base_lobes, base_asym, base_twist, hue) = match tribe {
            Tribe::Blood => (0.162, 0.036, 1.0_f32, 0.30_f32, 0.25_f32, 0.96_f32),
            Tribe::Moss => (0.192, 0.048, 3.0_f32, 0.10_f32, 0.60_f32, 0.36_f32),
            Tribe::Deep => (0.174, 0.042, 4.0_f32, 0.16_f32, 1.40_f32, 0.56_f32),
            Tribe::Solar => (0.184, 0.045, 5.0_f32, 0.18_f32, 0.90_f32, 0.12_f32),
            Tribe::Dream => (0.158, 0.038, 2.0_f32, 0.26_f32, 1.80_f32, 0.78_f32),
            Tribe::Static => (0.178, 0.040, 6.0_f32, 0.20_f32, -1.20_f32, 0.48_f32),
        };

        let kernel_radius = (4.0 + eco.perception * 6.0).round().clamp(3.0, 7.0) as u32;
        let ring_peaks = [
            (0.85 + eco.bonding * 0.25).clamp(0.25, 1.45),
            (0.25 + eco.membrane * 0.45).clamp(0.0, 1.1),
            (0.10 + eco.orbit * 0.35).clamp(0.0, 0.9),
        ];

        Self {
            kernel_radius,
            ring_peaks,
            shell: if eco.membrane > 0.55 {
                ShellKind::Exponential
            } else {
                ShellKind::Polynomial
            },
            angular_lobes: (base_lobes + rng.gen_range(-0.6_f32..0.6_f32)).clamp(1.0, 6.0),
            asymmetry: (base_asym + eco.volatility * 0.08 + rng.gen_range(-0.05_f32..0.05_f32))
                .clamp(0.02, 0.45),
            spiral_twist: (base_twist + rng.gen_range(-0.5_f32..0.5_f32)).clamp(-2.5, 2.5),
            mu: (base_mu + (eco.metabolism - 0.04) * 0.4 + rng.gen_range(-0.015_f32..0.015_f32))
                .clamp(0.10, 0.28),
            sigma: (base_sigma + eco.fertility * 0.012 + rng.gen_range(-0.005_f32..0.005_f32))
                .clamp(0.020, 0.075),
            org_mu: (0.24 + rng.gen_range(-0.06_f32..0.06_f32)).clamp(0.12, 0.42),
            org_coupling: (0.22 + eco.orbit * 0.25 + rng.gen_range(-0.08_f32..0.08_f32))
                .clamp(0.05, 0.65),
            dt: 0.11,
            membrane_affinity: (0.50 + eco.membrane * 0.45).clamp(0.2, 1.3),
            nutrient_sensitivity: (0.65 + eco.hunger * 10.0).clamp(0.3, 1.5),
            regeneration: (0.35 + eco.fertility * 0.45).clamp(0.15, 1.0),
            biolum_hue: (hue + rng.gen_range(-0.05_f32..0.05_f32)).rem_euclid(1.0),
            accent_shift: rng.gen_range(0.08_f32..0.32_f32),
        }
    }

    /// Mutate morphological genome; occasional macro-mutations create saltational diversity.
    pub fn mutate(&self, rng: &mut StdRng, rate: f32) -> Self {
        let mut child = *self;
        let macro_jump = if rng.gen_bool(0.08) { 2.4 } else { 1.0 };
        let r = (rate * macro_jump).clamp(0.005, 0.35);

        child.mu = (child.mu + rng.gen_range(-r..r) * 0.16).clamp(0.09, 0.30);
        child.sigma = (child.sigma + rng.gen_range(-r..r) * 0.07).clamp(0.018, 0.085);
        child.org_mu = (child.org_mu + rng.gen_range(-r..r) * 0.20).clamp(0.10, 0.48);
        child.org_coupling = (child.org_coupling + rng.gen_range(-r..r) * 0.25).clamp(0.04, 0.75);
        child.asymmetry = (child.asymmetry + rng.gen_range(-r..r) * 0.25).clamp(0.01, 0.48);
        child.angular_lobes =
            (child.angular_lobes + rng.gen_range(-r..r) * 2.2).clamp(1.0, 6.0);
        child.spiral_twist =
            (child.spiral_twist + rng.gen_range(-r..r) * 1.4).clamp(-2.8, 2.8);
        child.membrane_affinity =
            (child.membrane_affinity + rng.gen_range(-r..r) * 0.4).clamp(0.15, 1.4);
        child.nutrient_sensitivity =
            (child.nutrient_sensitivity + rng.gen_range(-r..r) * 0.4).clamp(0.25, 1.6);
        child.regeneration = (child.regeneration + rng.gen_range(-r..r) * 0.3).clamp(0.1, 1.0);
        child.biolum_hue = (child.biolum_hue + rng.gen_range(-r..r) * 0.14).rem_euclid(1.0);
        child.accent_shift = (child.accent_shift + rng.gen_range(-r..r) * 0.12).clamp(0.04, 0.45);

        for peak in &mut child.ring_peaks {
            *peak = (*peak + rng.gen_range(-r..r) * 0.35).clamp(0.0, 1.5);
        }

        if rng.gen_bool((r * 0.4).min(0.25) as f64) {
            let delta: i32 = if rng.gen_bool(0.5) { 1 } else { -1 };
            child.kernel_radius = (child.kernel_radius as i32 + delta).clamp(3, 7) as u32;
        }
        if rng.gen_bool((r * 0.2).min(0.12) as f64) {
            child.shell = match child.shell {
                ShellKind::Exponential => ShellKind::Polynomial,
                ShellKind::Polynomial => ShellKind::Exponential,
            };
        }

        child
    }

    pub fn crossover(&self, other: &Self, rng: &mut StdRng) -> Self {
        let pick = |a: f32, b: f32, rng: &mut StdRng| if rng.gen_bool(0.5) { a } else { b };
        Self {
            kernel_radius: if rng.gen_bool(0.5) { self.kernel_radius } else { other.kernel_radius },
            ring_peaks: [
                pick(self.ring_peaks[0], other.ring_peaks[0], rng),
                pick(self.ring_peaks[1], other.ring_peaks[1], rng),
                pick(self.ring_peaks[2], other.ring_peaks[2], rng),
            ],
            shell: if rng.gen_bool(0.5) { self.shell } else { other.shell },
            angular_lobes: pick(self.angular_lobes, other.angular_lobes, rng),
            asymmetry: pick(self.asymmetry, other.asymmetry, rng),
            spiral_twist: pick(self.spiral_twist, other.spiral_twist, rng),
            mu: (self.mu + other.mu) * 0.5,
            sigma: (self.sigma + other.sigma) * 0.5,
            org_mu: pick(self.org_mu, other.org_mu, rng),
            org_coupling: pick(self.org_coupling, other.org_coupling, rng),
            dt: self.dt,
            membrane_affinity: pick(self.membrane_affinity, other.membrane_affinity, rng),
            nutrient_sensitivity: pick(self.nutrient_sensitivity, other.nutrient_sensitivity, rng),
            regeneration: pick(self.regeneration, other.regeneration, rng),
            biolum_hue: pick(self.biolum_hue, other.biolum_hue, rng),
            accent_shift: pick(self.accent_shift, other.accent_shift, rng),
        }
    }

    pub fn distance(&self, other: &Self) -> f32 {
        (self.mu - other.mu).abs() * 6.0
            + (self.sigma - other.sigma).abs() * 14.0
            + (self.asymmetry - other.asymmetry).abs() * 2.5
            + (self.angular_lobes - other.angular_lobes).abs() * 0.35
            + (self.spiral_twist - other.spiral_twist).abs() * 0.3
            + (self.membrane_affinity - other.membrane_affinity).abs() * 0.8
    }
}

/// Primitive Adaptive Intelligence: compact heritable recurrent neural controller.
/// 6 sensory inputs -> 6 recurrent hidden units -> 3 motor/secretion outputs.
/// Inputs: [nutrient, grad_x, grad_y, toxin, pheromone, energy_norm]
/// Outputs: [thrust_x, thrust_y, signal_secretion]
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct NeuralBrain {
    pub w_in: [[f32; 6]; 6],
    pub w_rec: [[f32; 6]; 6],
    pub w_out: [[f32; 6]; 3],
    pub bias: [f32; 6],
    pub state: [f32; 6],
    pub last_out: [f32; 3],
}

impl NeuralBrain {
    pub fn random(rng: &mut StdRng) -> Self {
        let mut w_in = [[0.0f32; 6]; 6];
        let mut w_rec = [[0.0f32; 6]; 6];
        let mut w_out = [[0.0f32; 6]; 3];
        let mut bias = [0.0f32; 6];

        for i in 0..6 {
            bias[i] = rng.gen_range(-0.25..0.25);
            for j in 0..6 {
                w_in[i][j] = rng.gen_range(-0.6..0.6);
                w_rec[i][j] = rng.gen_range(-0.35..0.35);
            }
        }
        // Slight initial chemotaxis prior on grad_x/grad_y so early protocells can forage
        w_in[0][1] += 0.55;
        w_in[1][2] += 0.55;
        for o in 0..3 {
            for j in 0..6 {
                w_out[o][j] = rng.gen_range(-0.5..0.5);
            }
        }
        w_out[0][0] += 0.5;
        w_out[1][1] += 0.5;

        Self {
            w_in,
            w_rec,
            w_out,
            bias,
            state: [0.0; 6],
            last_out: [0.0; 3],
        }
    }

    pub fn step(&mut self, inputs: [f32; 6]) -> [f32; 3] {
        let mut next_state = [0.0f32; 6];
        for i in 0..6 {
            let mut acc = self.bias[i];
            for j in 0..6 {
                acc += self.w_in[i][j] * inputs[j] + self.w_rec[i][j] * self.state[j];
            }
            next_state[i] = acc.tanh();
        }
        self.state = next_state;

        let mut out = [0.0f32; 3];
        for o in 0..3 {
            let mut acc = 0.0f32;
            for j in 0..6 {
                acc += self.w_out[o][j] * self.state[j];
            }
            out[o] = acc.tanh();
        }
        self.last_out = out;
        out
    }

    pub fn mutate(&self, rng: &mut StdRng, rate: f32) -> Self {
        let mut child = *self;
        child.state = [0.0; 6];
        let s = rate.clamp(0.01, 0.35);
        for i in 0..6 {
            if rng.gen_bool(0.35) {
                child.bias[i] = (child.bias[i] + rng.gen_range(-s..s)).clamp(-2.0, 2.0);
            }
            for j in 0..6 {
                if rng.gen_bool(0.25) {
                    child.w_in[i][j] = (child.w_in[i][j] + rng.gen_range(-s..s)).clamp(-2.5, 2.5);
                }
                if rng.gen_bool(0.25) {
                    child.w_rec[i][j] =
                        (child.w_rec[i][j] + rng.gen_range(-s..s)).clamp(-1.5, 1.5);
                }
            }
        }
        for o in 0..3 {
            for j in 0..6 {
                if rng.gen_bool(0.3) {
                    child.w_out[o][j] =
                        (child.w_out[o][j] + rng.gen_range(-s..s)).clamp(-2.5, 2.5);
                }
            }
        }
        child
    }
}

/// Precomputed anisotropic radial + spiral + angular Lenia kernel.
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

                let angle = (dy as f32).atan2(dx as f32) - heading + r * genome.spiral_twist;
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

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct BodyMetrics {
    pub mass: f32,
    pub membrane_mass: f32,
    pub organelle_mass: f32,
    pub com_dx: f32,
    pub com_dy: f32,
    pub elongation: f32,
    pub activity: f32,
    pub radius_norm: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LocalEnvInput {
    pub nutrient: f32,
    pub pheromone: f32,
    pub toxin: f32,
    pub nutrient_grad_x: f32,
    pub nutrient_grad_y: f32,
    pub mutagen: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeniaBody {
    pub cells: Vec<f32>,
    #[serde(skip)]
    pub next: Vec<f32>,
    #[serde(skip)]
    pub kernel: Option<MorphKernel>,
    pub metrics: BodyMetrics,
    pub heading: f32,
    pub pulse_phase: f32,
}

impl LeniaBody {
    pub fn new(genome: &MorphGenome, heading: f32, rng: &mut StdRng) -> Self {
        let mut cells = vec![0.0f32; BODY_CELLS];
        let center = (BODY_RES as f32 - 1.0) * 0.5;
        let seed_r = (BODY_RES as f32) * 0.25;

        for y in 0..BODY_RES {
            for x in 0..BODY_RES {
                let dx = x as f32 - center;
                let dy = y as f32 - center;
                let d = (dx * dx + dy * dy).sqrt();
                let angle = dy.atan2(dx) - heading + (d / seed_r) * genome.spiral_twist * 0.35;
                let lobe_mod = 1.0 + 0.24 * (angle * genome.angular_lobes).cos();
                let r_norm = d / (seed_r * lobe_mod).max(1.0);

                if r_norm < 1.45 {
                    let core = (-r_norm * r_norm * 1.35).exp();
                    let jitter = rng.gen_range(-0.06..0.06);
                    let cyto = (core + jitter).clamp(0.0, 1.0);
                    let ring = (-(r_norm - 0.85).powi(2) / 0.14).exp() * genome.membrane_affinity;
                    let organelle = (-(r_norm * r_norm) * 3.2).exp() * genome.org_coupling;
                    let idx = (y * BODY_RES + x) * BODY_CHANNELS;
                    cells[idx] = cyto;
                    cells[idx + 1] = ring.clamp(0.0, 1.0);
                    cells[idx + 2] = organelle.clamp(0.0, 1.0);
                }
            }
        }

        let kernel = MorphKernel::build(genome, heading);
        let mut body = Self {
            next: cells.clone(),
            cells,
            kernel: Some(kernel),
            metrics: BodyMetrics::default(),
            heading,
            pulse_phase: rng.gen_range(0.0..TAU),
        };
        body.recompute_metrics(0.0);
        body
    }

    pub fn ensure_runtime_buffers(&mut self, genome: &MorphGenome) {
        if self.next.len() != BODY_CELLS {
            self.next = vec![0.0; BODY_CELLS];
        }
        if self.kernel.is_none() {
            self.kernel = Some(MorphKernel::build(genome, self.heading));
        }
    }

    pub fn step_cpu(&mut self, genome: &MorphGenome, env: LocalEnvInput) -> BodyMetrics {
        self.ensure_runtime_buffers(genome);
        self.pulse_phase = (self.pulse_phase + 0.16) % TAU;

        let nutrient_boost = (env.nutrient * genome.nutrient_sensitivity).clamp(0.0, 1.2);
        let toxin_stress = env.toxin.clamp(0.0, 1.0);
        let eff_mu = (genome.mu - nutrient_boost * 0.018 + toxin_stress * 0.022).clamp(0.08, 0.34);
        let eff_dt = genome.dt * (0.72 + nutrient_boost * 0.55);

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
        let taps = &self.kernel.as_ref().unwrap().taps;

        for y in 0..BODY_RES {
            for x in 0..BODY_RES {
                let mut u_cyto = 0.0f32;
                let mut u_org = 0.0f32;
                for &(dx, dy, w) in taps {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && ny >= 0 && nx < BODY_RES as i32 && ny < BODY_RES as i32 {
                        let nidx = (ny as usize * BODY_RES + nx as usize) * BODY_CHANNELS;
                        u_cyto += self.cells[nidx] * w;
                        u_org += self.cells[nidx + 2] * w;
                    }
                }

                let idx = (y * BODY_RES + x) * BODY_CHANNELS;
                let cyto = self.cells[idx];
                let mem = self.cells[idx + 1];
                let org = self.cells[idx + 2];

                let rx = x as f32 - center;
                let ry = y as f32 - center;
                let dist = (rx * rx + ry * ry).sqrt();
                let boundary_fade = ((max_r - dist) / 2.5).clamp(0.0, 1.0);

                let chemo = if dist > 0.5 {
                    ((rx * gx + ry * gy) / dist) * grad_mag.min(1.0) * 0.14
                } else {
                    0.0
                };

                // Ch 0: Cytoplasm coupled with organelle energy support
                let g_cyto = growth_gaussian(u_cyto, eff_mu, genome.sigma)
                    + chemo
                    + u_org * genome.org_coupling * 0.12;
                let toxin_loss = toxin_stress * 0.065 * (1.0 - mem * 0.65);
                let regen = if cyto > 0.05 && cyto < 0.45 && nutrient_boost > 0.25 {
                    genome.regeneration * 0.015 * nutrient_boost
                } else {
                    0.0
                };
                let next_cyto = ((cyto + eff_dt * g_cyto + regen - toxin_loss) * boundary_fade)
                    .clamp(0.0, 1.0);

                // Ch 1: Organic Membrane at cytoplasm boundary
                let edge_signal = (4.0 * next_cyto * (1.0 - next_cyto)).clamp(0.0, 1.0);
                let target_mem =
                    (edge_signal * genome.membrane_affinity * (0.65 + 0.35 * (1.0 - u_cyto)))
                        .clamp(0.0, 1.0);
                let next_mem = (mem * 0.78 + target_mem * 0.22) * boundary_fade;

                // Ch 2: Internal Organelle / Bioluminescent Nucleus differentiated inside cytoplasm
                let g_org = growth_gaussian(u_cyto, genome.org_mu, genome.sigma * 1.35);
                let next_org = ((org + eff_dt * 0.65 * g_org * next_cyto) * boundary_fade)
                    .clamp(0.0, next_cyto);

                total_delta += (next_cyto - cyto).abs();
                self.next[idx] = next_cyto;
                self.next[idx + 1] = next_mem.clamp(0.0, 1.0);
                self.next[idx + 2] = next_org;
            }
        }

        std::mem::swap(&mut self.cells, &mut self.next);
        self.recenter_and_measure(total_delta / (BODY_RES * BODY_RES) as f32)
    }

    fn recenter_and_measure(&mut self, activity: f32) -> BodyMetrics {
        self.recompute_metrics(activity);

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
                        self.next[dst + 2] = self.cells[src + 2];
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
        let mut org_sum = 0.0f32;
        let mut wx = 0.0f32;
        let mut wy = 0.0f32;

        for y in 0..BODY_RES {
            for x in 0..BODY_RES {
                let idx = (y * BODY_RES + x) * BODY_CHANNELS;
                let c = self.cells[idx];
                mass_sum += c;
                mem_sum += self.cells[idx + 1];
                org_sum += self.cells[idx + 2];
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
            organelle_mass: (org_sum / area).clamp(0.0, 1.0),
            com_dx: com_dx.clamp(-1.0, 1.0),
            com_dy: com_dy.clamp(-1.0, 1.0),
            elongation,
            activity,
            radius_norm,
        };
    }

    pub fn divide(&mut self, daughter_genome: &MorphGenome, rng: &mut StdRng) -> Self {
        let mut daughter_cells = vec![0.0f32; BODY_CELLS];
        let half = BODY_RES / 2;

        for y in 0..BODY_RES {
            for x in 0..BODY_RES {
                let idx = (y * BODY_RES + x) * BODY_CHANNELS;
                if x >= half {
                    let dx = (x - half / 2).min(BODY_RES - 1);
                    let didx = (y * BODY_RES + dx) * BODY_CHANNELS;
                    for c in 0..BODY_CHANNELS {
                        daughter_cells[didx + c] = (self.cells[idx + c] * 0.92).clamp(0.0, 1.0);
                        self.cells[idx + c] *= 0.25;
                    }
                } else {
                    for c in 0..BODY_CHANNELS {
                        self.cells[idx + c] = (self.cells[idx + c] * 0.92).clamp(0.0, 1.0);
                    }
                }
            }
        }

        let heading = (self.heading + rng.gen_range(-0.6..0.6)).rem_euclid(TAU);
        let kernel = MorphKernel::build(daughter_genome, heading);
        self.recompute_metrics(0.05);

        let mut daughter = Self {
            next: daughter_cells.clone(),
            cells: daughter_cells,
            kernel: Some(kernel),
            metrics: BodyMetrics::default(),
            heading,
            pulse_phase: rng.gen_range(0.0..TAU),
        };
        daughter.recompute_metrics(0.05);
        daughter
    }

    /// Inflict real physical damage to the organism's cytoplasm & organelles.
    /// Returns the actual biomass lost (used for coherent predation energy transfer).
    pub fn apply_damage(&mut self, fraction: f32) -> f32 {
        let before = self.metrics.mass;
        let retain = (1.0 - fraction.clamp(0.0, 0.95)).max(0.05);
        for c in &mut self.cells {
            *c *= retain;
        }
        self.recompute_metrics(fraction);
        (before - self.metrics.mass).max(0.0)
    }
}
