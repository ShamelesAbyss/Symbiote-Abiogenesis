//! Headless high-resolution PNG screenshot renderer for Symbiote: Project Abiogenesis.
//!
//! Uses the exact same bioluminescent shading equations as `OBSERVATORY_WGSL`
//! so documentation screenshots and CI visual checks reflect the real simulation.

use crate::convergence::eco_field::{ECO_CHANNELS, ECO_HEIGHT, ECO_WIDTH};
use crate::convergence::engine::ConvergenceWorld;
use crate::convergence::lenia_body::{BODY_CHANNELS, BODY_RES};
use anyhow::Result;
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;

pub fn capture_showcase_screenshots(prefer_gpu: bool) -> Result<()> {
    fs::create_dir_all("assets/screenshots")?;

    let mut world = ConvergenceWorld::new(0xAB10_6E4E, prefer_gpu);
    for _ in 0..220 {
        world.step();
    }

    render_png(
        &world,
        1280,
        720,
        0.0,
        0.0,
        1.0,
        Path::new("assets/screenshots/ecosystem_overview.png"),
    )?;

    // Center microscopic zoom on the highest-energy thriving organism
    let (fx, fy) = world
        .organisms
        .iter()
        .max_by(|a, b| a.energy.total_cmp(&b.energy))
        .map(|o| (o.x, o.y))
        .unwrap_or((0.0, 0.0));

    render_png(
        &world,
        1280,
        720,
        fx,
        fy,
        5.5,
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

            let mut col = [0.012f32, 0.018, 0.032];
            if wx.abs() <= 1.0 && wy.abs() <= 1.0 {
                let e = sample_eco_bilinear(&world.eco_field.cells, wx, wy);
                col[0] += 0.03 * e[0] + 0.14 * e[1] + 0.32 * e[2] + 0.10 * e[3].powf(1.3);
                col[1] += 0.16 * e[0] + 0.06 * e[1] + 0.05 * e[2] + 0.34 * e[3].powf(1.3);
                col[2] += 0.18 * e[0] + 0.24 * e[1] + 0.08 * e[2] + 0.68 * e[3].powf(1.3);
            }

            for org in &world.organisms {
                let radius = 0.055 + org.body.metrics.radius_norm * 0.055;
                let dx = wx - org.x;
                let dy = wy - org.y;
                if dx.abs() > radius || dy.abs() > radius {
                    continue;
                }
                let lu = (dx / radius) * 0.5 + 0.5;
                let lv = (dy / radius) * 0.5 + 0.5;
                let (cyto, mem) = sample_body_bilinear(&org.body.cells, lu, lv);
                if cyto > 0.01 || mem > 0.01 {
                    let energy_norm = (org.energy / 100.0).clamp(0.1, 1.4);
                    let breath = 0.88 + 0.12 * org.body.pulse_phase.sin();
                    let core_rgb = hsv_to_rgb(org.morph_genome.biolum_hue, 0.72, 0.95 * breath * energy_norm);
                    let mem_rgb = hsv_to_rgb((org.morph_genome.biolum_hue + 0.14).fract(), 0.45, 1.25);

                    let alpha = (cyto * 0.92).clamp(0.0, 0.95);
                    for c in 0..3 {
                        let c_val = core_rgb[c] * (0.45 + 0.85 * cyto);
                        col[c] = col[c] * (1.0 - alpha) + c_val * alpha + mem_rgb[c] * mem * 0.65;
                    }
                }
            }

            for c in 0..3 {
                let mapped = (col[c] / (1.0 + col[c] * 0.35)).powf(0.92).clamp(0.0, 1.0);
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

fn sample_eco_bilinear(cells: &[f32], wx: f32, wy: f32) -> [f32; 4] {
    let u = ((wx + 1.0) * 0.5).clamp(0.0, 0.999) * (ECO_WIDTH - 1) as f32;
    let v = ((wy + 1.0) * 0.5).clamp(0.0, 0.999) * (ECO_HEIGHT - 1) as f32;
    let x0 = u.floor() as usize;
    let y0 = v.floor() as usize;
    let x1 = (x0 + 1).min(ECO_WIDTH - 1);
    let y1 = (y0 + 1).min(ECO_HEIGHT - 1);
    let fx = u.fract();
    let fy = v.fract();

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

fn sample_body_bilinear(cells: &[f32], lu: f32, lv: f32) -> (f32, f32) {
    let u = lu.clamp(0.0, 0.999) * (BODY_RES - 1) as f32;
    let v = lv.clamp(0.0, 0.999) * (BODY_RES - 1) as f32;
    let x0 = u.floor() as usize;
    let y0 = v.floor() as usize;
    let x1 = (x0 + 1).min(BODY_RES - 1);
    let y1 = (y0 + 1).min(BODY_RES - 1);
    let fx = u.fract();
    let fy = v.fract();

    let b00 = (y0 * BODY_RES + x0) * BODY_CHANNELS;
    let b10 = (y0 * BODY_RES + x1) * BODY_CHANNELS;
    let b01 = (y1 * BODY_RES + x0) * BODY_CHANNELS;
    let b11 = (y1 * BODY_RES + x1) * BODY_CHANNELS;

    let c_top = cells[b00] * (1.0 - fx) + cells[b10] * fx;
    let c_bot = cells[b01] * (1.0 - fx) + cells[b11] * fx;
    let m_top = cells[b00 + 1] * (1.0 - fx) + cells[b10 + 1] * fx;
    let m_bot = cells[b01 + 1] * (1.0 - fx) + cells[b11 + 1] * fx;
    (c_top * (1.0 - fy) + c_bot * fy, m_top * (1.0 - fy) + m_bot * fy)
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
