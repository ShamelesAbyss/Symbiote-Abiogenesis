//! Headless high-resolution PNG screenshot renderer for Symbiote: Abiogenesis — GENOMORPH.
//!
//! Renders the toroidal abyssal EcoField, smooth 40x40x3 continuous Lenia bodies,
//! AND each organism's articulated procedural anatomy (segments, oscillating fins,
//! undulating tails, predatory jaws, filter veils, sensory antennae, armor carapaces).

use crate::convergence::eco_field::{ECO_CHANNELS, ECO_HEIGHT, ECO_WIDTH};
use crate::convergence::engine::{toroidal_delta, ConvergenceWorld};
use crate::convergence::genomorph::OrganKind;
use crate::convergence::lenia_body::{BODY_CHANNELS, BODY_RES};
use anyhow::Result;
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;

pub fn capture_showcase_screenshots(prefer_gpu: bool) -> Result<()> {
    fs::create_dir_all("assets/screenshots")?;

    let mut world = ConvergenceWorld::new(0xAB10_6E4E, prefer_gpu);
    for _ in 0..180 {
        world.step();
    }

    render_png(
        &world,
        1440,
        810,
        0.0,
        0.0,
        1.82,
        Path::new("assets/screenshots/ecosystem_overview.png"),
    )?;

    // Center microscopic zoom on the organism with the richest anatomical structure
    let (fx, fy) = world
        .organisms
        .iter()
        .max_by(|a, b| {
            let sa = a.phenotype.organs.len() as f32 * 18.0 + a.energy;
            let sb = b.phenotype.organs.len() as f32 * 18.0 + b.energy;
            sa.total_cmp(&sb)
        })
        .map(|o| (o.x, o.y))
        .unwrap_or((0.0, 0.0));

    render_png(
        &world,
        1440,
        810,
        fx,
        fy,
        6.2,
        Path::new("assets/screenshots/microscopic_cells.png"),
    )?;

    println!("Saved assets/screenshots/ecosystem_overview.png and assets/screenshots/microscopic_cells.png");
    Ok(())
}

pub fn render_png(
    world: &ConvergenceWorld,
    width: u32,
    height: u32,
    center_x: f32,
    center_y: f32,
    zoom: f32,
    path: &Path,
) -> Result<()> {
    let w = width as usize;
    let h = height as usize;
    let aspect = width as f32 / height.max(1) as f32;
    let mut rgb = vec![0u8; w * h * 3];

    rgb.par_chunks_mut(w * 3).enumerate().for_each(|(py, row)| {
        let v = py as f32 / height as f32;
        let ndc_y = (v - 0.5) * 2.0;
        let wy = center_y + ndc_y / zoom;

        for px in 0..w {
            let u = px as f32 / width as f32;
            let ndc_x = (u - 0.5) * 2.0;
            let wx = center_x + (ndc_x * aspect) / zoom;

            let vig = (1.0 - 0.24 * (ndc_x * ndc_x * 0.45 + ndc_y * ndc_y)).clamp(0.55, 1.0);
            let mut col = [0.003 * vig, 0.005 * vig, 0.011 * vig];

            let e = sample_eco_smooth(&world.eco_field.cells, wx, wy);
            let nutr_wisp = (e[0] - 0.36).max(0.0).powf(2.2);
            let pher_wisp = e[1].powf(1.5);
            let tox_wisp = e[2].powf(1.6);
            let root_fil = e[3].powf(2.4);

            col[0] += 0.008 * nutr_wisp + 0.14 * pher_wisp + 0.42 * tox_wisp + 0.03 * root_fil;
            col[1] += 0.055 * nutr_wisp + 0.05 * pher_wisp + 0.05 * tox_wisp + 0.14 * root_fil;
            col[2] += 0.085 * nutr_wisp + 0.26 * pher_wisp + 0.09 * tox_wisp + 0.28 * root_fil;

            let h_speck = hash2d((wx * 42.0).floor() as i32, (wy * 42.0).floor() as i32);
            if h_speck > 0.992 {
                let fx_s = (wx * 42.0).fract() - 0.5;
                let fy_s = (wy * 42.0).fract() - 0.5;
                let d_s = (fx_s * fx_s + fy_s * fy_s).sqrt();
                let glow = (1.0 - d_s * 5.0).max(0.0).powi(2) * 0.18;
                col[0] += glow * 0.4;
                col[1] += glow * 0.8;
                col[2] += glow * 1.1;
            }

            for org in &world.organisms {
                let radius = (0.055 + org.body.metrics.radius_norm * 0.058)
                    * (0.45 + org.maturity * 0.55)
                    * org.genomorph.base_scale;
                let bound_r = radius * 1.75;
                let dx = toroidal_delta(wx - org.x);
                let dy = toroidal_delta(wy - org.y);
                if dx.abs() > bound_r || dy.abs() > bound_r {
                    continue;
                }

                let energy_norm = (org.energy / 95.0).clamp(0.30, 1.18);
                let breath = 0.90 + 0.10 * org.body.pulse_phase.sin();
                let core_rgb =
                    hsv_to_rgb(org.morph_genome.biolum_hue, 0.90, 0.72 * breath * energy_norm);
                let mem_rgb = hsv_to_rgb(
                    (org.morph_genome.biolum_hue + 0.08).rem_euclid(1.0),
                    0.70,
                    1.15,
                );
                let org_rgb = hsv_to_rgb(
                    (org.morph_genome.biolum_hue + org.morph_genome.accent_shift).rem_euclid(1.0),
                    0.95,
                    1.45,
                );

                // Soft ambient bioluminescent halo
                let dist_norm = (dx * dx + dy * dy).sqrt() / bound_r;
                if dist_norm < 1.0 {
                    let halo =
                        (1.0 - dist_norm).powi(3) * 0.075 * org.body.metrics.mass * 3.0 * energy_norm;
                    for c in 0..3 {
                        col[c] += core_rgb[c] * halo;
                    }
                }

                // 1. Continuous 40x40x3 Lenia cellular body
                if dx.abs() <= radius && dy.abs() <= radius {
                    let lu = (dx / radius) * 0.5 + 0.5;
                    let lv = (dy / radius) * 0.5 + 0.5;
                    let (cyto, mem, organ) = sample_body_hermite(&org.body.cells, lu, lv);

                    if cyto > 0.015 || mem > 0.015 || organ > 0.015 {
                        let cyto_soft = smoothstep(0.03, 0.72, cyto);
                        let inner_glow = cyto.powf(1.6);
                        let alpha = (cyto_soft * 0.72).clamp(0.0, 0.82);
                        let rim = mem.powf(1.45) * 0.65;
                        let nucleus = organ.powf(1.3) * 1.05;

                        for c in 0..3 {
                            let body_col = core_rgb[c] * (0.20 + 0.85 * inner_glow);
                            col[c] = col[c] * (1.0 - alpha)
                                + body_col * alpha
                                + mem_rgb[c] * rim
                                + org_rgb[c] * nucleus;
                        }
                    }
                }

                // 2. Articulated procedural anatomy organs (rotated by organism heading)
                let cos_h = org.body.heading.cos();
                let sin_h = org.body.heading.sin();
                let lx = (dx * cos_h + dy * sin_h) / radius;
                let ly = (-dx * sin_h + dy * cos_h) / radius;

                for organ in &org.phenotype.organs {
                    let (dist, t) = sd_capsule_t(lx, ly, organ.x0, organ.y0, organ.x1, organ.y1);
                    let w_taper = organ.width * (1.0 - t * 0.42).max(0.25);
                    if dist > w_taper * 1.85 {
                        continue;
                    }
                    let fill = (1.0 - smoothstep(w_taper * 0.35, w_taper, dist)).clamp(0.0, 1.0);
                    let rim = (1.0 - smoothstep(0.0, w_taper * 0.55, (dist - w_taper * 0.75).abs()))
                        .clamp(0.0, 1.0);

                    let o_hue = (org.morph_genome.biolum_hue + organ.hue_shift).rem_euclid(1.0);
                    let (o_sat, o_val, o_alpha) = match organ.kind {
                        OrganKind::FeedingJaw => (0.95, 1.35 * energy_norm, 0.82),
                        OrganKind::SensoryAntenna => (0.75, 1.45 * energy_norm, 0.72),
                        OrganKind::ArmorCarapace => (0.45, 1.05 * energy_norm, 0.78),
                        OrganKind::FilterMembrane => (0.85, 0.85 * energy_norm, 0.48),
                        OrganKind::DigestiveVacuole | OrganKind::ToxinGland => {
                            (0.95, 1.25 * energy_norm, 0.65)
                        }
                        _ => (0.85, 0.95 * energy_norm, 0.68),
                    };
                    let o_col = hsv_to_rgb(o_hue, o_sat, o_val);
                    let a = fill * o_alpha;
                    for c in 0..3 {
                        col[c] = col[c] * (1.0 - a * 0.75)
                            + o_col[c] * a * (0.45 + 0.55 * (1.0 - t))
                            + o_col[c] * rim * 0.45;
                    }
                }
            }

            for c in 0..3 {
                let x = col[c].max(0.0);
                let mapped = ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14))
                    .powf(0.92)
                    .clamp(0.0, 1.0);
                row[px * 3 + c] = (mapped * 255.0) as u8;
            }
        }
    });

    let file = File::create(path)?;
    let w_buf = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w_buf, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&rgb)?;
    Ok(())
}

#[inline]
fn sd_capsule_t(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> (f32, f32) {
    let pax = px - ax;
    let pay = py - ay;
    let bax = bx - ax;
    let bay = by - ay;
    let h = ((pax * bax + pay * bay) / (bax * bax + bay * bay).max(1e-5)).clamp(0.0, 1.0);
    let dx = pax - bax * h;
    let dy = pay - bay * h;
    ((dx * dx + dy * dy).sqrt(), h)
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn hash2d(x: i32, y: i32) -> f32 {
    let n = (x.wrapping_mul(374_761_393) ^ y.wrapping_mul(668_265_263))
        .wrapping_mul(1_274_126_177);
    let u = ((n ^ (n >> 16)) as u32) & 0x00FF_FFFF;
    u as f32 / 16_777_215.0
}

fn sample_eco_smooth(cells: &[f32], wx: f32, wy: f32) -> [f32; 4] {
    let u = ((wx + 1.0) * 0.5).rem_euclid(1.0) * ECO_WIDTH as f32;
    let v = ((wy + 1.0) * 0.5).rem_euclid(1.0) * ECO_HEIGHT as f32;
    let x0 = (u.floor() as usize) % ECO_WIDTH;
    let y0 = (v.floor() as usize) % ECO_HEIGHT;
    let x1 = (x0 + 1) % ECO_WIDTH;
    let y1 = (y0 + 1) % ECO_HEIGHT;
    let fx = smoothstep(0.0, 1.0, u.fract());
    let fy = smoothstep(0.0, 1.0, v.fract());

    let mut out = [0.0f32; 4];
    for c in 0..4 {
        let c00 = cells[(y0 * ECO_WIDTH + x0) * ECO_CHANNELS + c];
        let c10 = cells[(y0 * ECO_WIDTH + x1) * ECO_CHANNELS + c];
        let c01 = cells[(y1 * ECO_WIDTH + x0) * ECO_CHANNELS + c];
        let c11 = cells[(y1 * ECO_WIDTH + x1) * ECO_CHANNELS + c];
        let top = c00 * (1.0 - fx) + c10 * fx;
        let bot = c01 * (1.0 - fx) + c11 * fx;
        out[c] = top * (1.0 - fy) + bot * fy;
    }
    out
}

fn sample_body_hermite(cells: &[f32], lu: f32, lv: f32) -> (f32, f32, f32) {
    let u = lu.clamp(0.0, 0.999) * (BODY_RES - 1) as f32;
    let v = lv.clamp(0.0, 0.999) * (BODY_RES - 1) as f32;
    let x0 = u.floor() as usize;
    let y0 = v.floor() as usize;
    let x1 = (x0 + 1).min(BODY_RES - 1);
    let y1 = (y0 + 1).min(BODY_RES - 1);
    let fx = smoothstep(0.0, 1.0, u.fract());
    let fy = smoothstep(0.0, 1.0, v.fract());

    let b00 = (y0 * BODY_RES + x0) * BODY_CHANNELS;
    let b10 = (y0 * BODY_RES + x1) * BODY_CHANNELS;
    let b01 = (y1 * BODY_RES + x0) * BODY_CHANNELS;
    let b11 = (y1 * BODY_RES + x1) * BODY_CHANNELS;

    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let top = cells[b00 + c] * (1.0 - fx) + cells[b10 + c] * fx;
        let bot = cells[b01 + c] * (1.0 - fx) + cells[b11 + c] * fx;
        out[c] = top * (1.0 - fy) + bot * fy;
    }
    (out[0], out[1], out[2])
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let hp = h.rem_euclid(1.0) * 6.0;
    let x = c * (1.0 - ((hp % 2.0) - 1.0).abs());
    let (r, g, b) = if hp < 1.0 {
        (c, x, 0.0)
    } else if hp < 2.0 {
        (x, c, 0.0)
    } else if hp < 3.0 {
        (0.0, c, x)
    } else if hp < 4.0 {
        (0.0, x, c)
    } else if hp < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = v - c;
    [r + m, g + m, b + m]
}
