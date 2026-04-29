//! gui_audit — pixel-level static audit of a GUI screenshot.
//!
//! Reads a PNG (the user's already-captured screenshot — winshot /
//! desk_capture / printscreen / whatever), walks every pixel exactly
//! once, and emits a JSON report of measurable layout / readability
//! signals: aspect ratio, dominant palette, max-pair WCAG contrast,
//! edge density, column-alignment peaks, and margin balance. Plus a
//! `blockers` array of rule violations the caller can use as a
//! ship-gate.
//!
//! This complements the CP-protocol verifier (which proves the widget
//! tree and state are correct) and the VLM critique (which speaks to
//! taste). gui_audit is deterministic — same PNG in, same JSON out —
//! and is meant to fail cheap when contrast is broken or the canvas
//! is empty, before we burn a model call on the VLM step.
//!
//! Usage:
//!     gui_audit --in path/to/shot.png [--out report.json]
//!     gui_audit --in shot.png --palette 12 --json-pretty
//!
//! Exit codes: 0 OK, 1 IO/parse failure, 2 blockers fired (so a
//! shell wrapper can `gui_audit ... || exit_with_block`).

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use serde::Serialize;

#[derive(Default)]
struct Args {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    palette_n: usize,
    pretty: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        palette_n: 8,
        ..Default::default()
    };
    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--in" | "-i" => {
                a.input = Some(PathBuf::from(it.next().ok_or("--in needs value")?));
            }
            "--out" | "-o" => {
                a.output = Some(PathBuf::from(it.next().ok_or("--out needs value")?));
            }
            "--palette" => {
                a.palette_n = it
                    .next()
                    .ok_or("--palette needs value")?
                    .parse()
                    .map_err(|e| format!("--palette parse: {e}"))?;
            }
            "--json-pretty" => a.pretty = true,
            "--help" | "-h" => {
                println!(
                    "gui_audit --in <png> [--out <json>] [--palette N] [--json-pretty]"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }
    if a.input.is_none() {
        return Err("--in <png> is required".into());
    }
    Ok(a)
}

#[derive(Serialize)]
struct Report {
    source: String,
    width: u32,
    height: u32,
    aspect_ratio: f64,
    pixel_count: u64,
    palette: Vec<PaletteEntry>,
    contrast: ContrastReport,
    edge_density: f64,
    layout: LayoutMetrics,
    margins: Margins,
    blockers: Vec<String>,
    // Free-form scores in [0,1] so the VLM/skill layer has a
    // single-knob summary without re-deriving from the metrics.
    scores: Scores,
}

#[derive(Serialize)]
struct PaletteEntry {
    hex: String,
    rgb: [u8; 3],
    frac: f64,
}

#[derive(Serialize)]
struct ContrastReport {
    /// Top palette pair by frequency × ratio — the contrast you
    /// actually look at when you open the screen.
    primary: ContrastPair,
    /// Worst pair among the top-N palette colors (any-vs-any).
    worst: ContrastPair,
}

#[derive(Serialize)]
struct ContrastPair {
    fg: String,
    bg: String,
    ratio: f64,
    wcag_aa_normal: bool,  // ≥ 4.5
    wcag_aa_large: bool,   // ≥ 3.0
    wcag_aaa_normal: bool, // ≥ 7.0
}

#[derive(Serialize)]
struct LayoutMetrics {
    column_peaks_x: Vec<u32>,
    column_alignment_score: f64, // peaks_kept / peaks_seen
    row_peaks_y: Vec<u32>,
}

#[derive(Serialize)]
struct Margins {
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
    /// 1.0 = perfectly balanced (all four equal); 0.0 = totally
    /// lopsided. Cheap proxy for "centered layout?".
    balance: f64,
}

#[derive(Serialize)]
struct Scores {
    readability: f64,    // contrast-driven
    structure: f64,      // column peak density
    breathability: f64,  // edge_density inverted around target band
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("gui_audit: {e}");
            return ExitCode::from(1);
        }
    };
    let input = args.input.unwrap();
    let img = match image::ImageReader::open(&input)
        .and_then(|r| Ok(r.with_guessed_format()?))
        .map_err(|e| format!("open: {e}"))
        .and_then(|r| r.decode().map_err(|e| format!("decode: {e}")))
    {
        Ok(v) => v.to_rgb8(),
        Err(e) => {
            eprintln!("gui_audit: {e}");
            return ExitCode::from(1);
        }
    };
    let (w, h) = img.dimensions();
    let pixel_count = w as u64 * h as u64;

    // ---- 1. quantized palette histogram (8^3 buckets) ----
    let mut hist: HashMap<u32, u64> = HashMap::with_capacity(512);
    for px in img.pixels() {
        let [r, g, b] = px.0;
        let key = ((r as u32 >> 5) << 6) | ((g as u32 >> 5) << 3) | (b as u32 >> 5);
        *hist.entry(key).or_insert(0) += 1;
    }
    // bucket → mean RGB (use bucket center; 32-step buckets).
    let mut buckets: Vec<(u32, u64)> = hist.into_iter().collect();
    buckets.sort_by(|a, b| b.1.cmp(&a.1));
    let palette: Vec<PaletteEntry> = buckets
        .iter()
        .take(args.palette_n)
        .map(|(k, count)| {
            let r = (((k >> 6) & 0x7) as u8) * 32 + 16;
            let g = (((k >> 3) & 0x7) as u8) * 32 + 16;
            let b = ((k & 0x7) as u8) * 32 + 16;
            PaletteEntry {
                hex: format!("#{r:02x}{g:02x}{b:02x}"),
                rgb: [r, g, b],
                frac: *count as f64 / pixel_count as f64,
            }
        })
        .collect();

    // ---- 2. contrast (WCAG 2.1) ----
    // primary = palette[0] (background) vs palette[1] (most likely text).
    let primary = if palette.len() >= 2 {
        wcag_pair(palette[1].rgb, palette[0].rgb)
    } else {
        // Single colour = no contrast at all.
        wcag_pair([0, 0, 0], [255, 255, 255]).with_ratio_zeroed()
    };
    let mut worst = primary.clone();
    for i in 0..palette.len() {
        for j in i + 1..palette.len() {
            let p = wcag_pair(palette[i].rgb, palette[j].rgb);
            if p.ratio < worst.ratio {
                worst = p;
            }
        }
    }

    // ---- 3. edge density + column / row peak projection ----
    // Single sweep: |Δx luma| + |Δy luma| > threshold.
    let threshold: u32 = 32;
    let mut edge_count: u64 = 0;
    let mut col_proj = vec![0u32; w as usize];
    let mut row_proj = vec![0u32; h as usize];
    let raw = img.as_raw(); // [r,g,b,r,g,b,...]
    let stride = (w as usize) * 3;
    let luma_at = |x: u32, y: u32| -> u32 {
        let i = y as usize * stride + x as usize * 3;
        // Rec.709 weights, scaled to integer.
        (raw[i] as u32 * 2126 + raw[i + 1] as u32 * 7152 + raw[i + 2] as u32 * 722) / 10000
    };
    for y in 0..h {
        for x in 0..w {
            let l = luma_at(x, y);
            let dx = if x + 1 < w {
                luma_at(x + 1, y).abs_diff(l)
            } else {
                0
            };
            let dy = if y + 1 < h {
                luma_at(x, y + 1).abs_diff(l)
            } else {
                0
            };
            if dx + dy > threshold {
                edge_count += 1;
                col_proj[x as usize] += 1;
                row_proj[y as usize] += 1;
            }
        }
    }
    let edge_density = edge_count as f64 / pixel_count as f64;

    let column_peaks_x = peaks(&col_proj, 6);
    let row_peaks_y = peaks(&row_proj, 6);
    let column_alignment_score = if col_proj.is_empty() {
        0.0
    } else {
        let mean = col_proj.iter().map(|&v| v as f64).sum::<f64>() / col_proj.len() as f64;
        let above = col_proj.iter().filter(|&&v| v as f64 > mean).count() as f64;
        // Lots of evenly-distributed activity → low score; sharp peaks
        // concentrated in few columns → high score (aligned UI).
        1.0 - (above / col_proj.len() as f64).min(1.0)
    };

    // ---- 4. margins (using top palette colour as background) ----
    let bg = palette
        .first()
        .map(|p| p.rgb)
        .unwrap_or([255, 255, 255]);
    let bg_tol: u32 = 24;
    let is_bg = |x: u32, y: u32| -> bool {
        let i = y as usize * stride + x as usize * 3;
        let dr = (raw[i] as i32 - bg[0] as i32).unsigned_abs();
        let dg = (raw[i + 1] as i32 - bg[1] as i32).unsigned_abs();
        let db = (raw[i + 2] as i32 - bg[2] as i32).unsigned_abs();
        dr + dg + db <= bg_tol
    };
    let row_is_blank = |y: u32| -> bool {
        let mut bg_n = 0u32;
        for x in 0..w {
            if is_bg(x, y) {
                bg_n += 1;
            }
        }
        bg_n as f64 / w as f64 > 0.95
    };
    let col_is_blank = |x: u32| -> bool {
        let mut bg_n = 0u32;
        for y in 0..h {
            if is_bg(x, y) {
                bg_n += 1;
            }
        }
        bg_n as f64 / h as f64 > 0.95
    };
    let top = (0..h).take_while(|&y| row_is_blank(y)).count() as u32;
    let bottom = (0..h).rev().take_while(|&y| row_is_blank(y)).count() as u32;
    let left = (0..w).take_while(|&x| col_is_blank(x)).count() as u32;
    let right = (0..w).rev().take_while(|&x| col_is_blank(x)).count() as u32;
    let bal_arr = [top, bottom, left, right];
    let margin_max = *bal_arr.iter().max().unwrap_or(&0) as f64;
    let margin_min = *bal_arr.iter().min().unwrap_or(&0) as f64;
    let balance = if margin_max < 1.0 {
        1.0
    } else {
        margin_min / margin_max
    };

    // ---- 5. blockers + scores ----
    let mut blockers = Vec::new();
    if pixel_count < 10_000 {
        blockers.push(format!(
            "image_too_small: {w}x{h} pixel_count={pixel_count} (suspect blank capture)"
        ));
    }
    if edge_density < 0.005 {
        blockers.push(format!(
            "near_empty_canvas: edge_density={edge_density:.4} (<0.5% — likely white screen)"
        ));
    }
    if primary.ratio < 4.5 {
        blockers.push(format!(
            "low_primary_contrast: ratio={:.2} (WCAG AA normal text needs ≥4.5)",
            primary.ratio
        ));
    }
    if let Some(top) = palette.first() {
        // 0.97 was picked empirically: a busy egui jukebox view with the
        // default Light theme sits at ~0.95 background coverage and is
        // not actually blank. Above 0.97 you really are looking at a
        // splash screen / not-yet-painted canvas.
        if top.frac > 0.97 {
            blockers.push(format!(
                "single_colour_dominant: top palette {} covers {:.1}% of frame",
                top.hex,
                top.frac * 100.0
            ));
        }
    }
    let readability = (primary.ratio / 7.0).clamp(0.0, 1.0); // AAA = 1.0
    let structure = column_alignment_score;
    // Healthy egui-style UIs sit around 4-12% edge density; outside
    // that band feels either blank or busy.
    let breathability = {
        let band_lo = 0.04;
        let band_hi = 0.12;
        if edge_density < band_lo {
            (edge_density / band_lo).clamp(0.0, 1.0)
        } else if edge_density > band_hi {
            (1.0 - (edge_density - band_hi) / band_hi).clamp(0.0, 1.0)
        } else {
            1.0
        }
    };

    let report = Report {
        source: input.display().to_string(),
        width: w,
        height: h,
        aspect_ratio: w as f64 / h.max(1) as f64,
        pixel_count,
        palette,
        contrast: ContrastReport { primary, worst },
        edge_density,
        layout: LayoutMetrics {
            column_peaks_x,
            column_alignment_score,
            row_peaks_y,
        },
        margins: Margins {
            top,
            bottom,
            left,
            right,
            balance,
        },
        blockers: blockers.clone(),
        scores: Scores {
            readability,
            structure,
            breathability,
        },
    };

    let json = if args.pretty {
        serde_json::to_string_pretty(&report)
    } else {
        serde_json::to_string(&report)
    };
    let json = match json {
        Ok(s) => s,
        Err(e) => {
            eprintln!("gui_audit: serialize: {e}");
            return ExitCode::from(1);
        }
    };
    if let Some(out) = args.output {
        if let Some(parent) = out.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Err(e) = fs::write(&out, &json) {
            eprintln!("gui_audit: write {}: {e}", out.display());
            return ExitCode::from(1);
        }
    } else {
        println!("{json}");
    }
    if !blockers.is_empty() {
        return ExitCode::from(2);
    }
    ExitCode::SUCCESS
}

fn peaks(proj: &[u32], min_gap: usize) -> Vec<u32> {
    if proj.is_empty() {
        return Vec::new();
    }
    let n = proj.len() as f64;
    let mean: f64 = proj.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var: f64 = proj.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let stddev = var.sqrt();
    let cutoff = mean + stddev; // 1σ above mean
    let mut out: Vec<u32> = Vec::new();
    let mut last_kept: Option<usize> = None;
    for (i, &v) in proj.iter().enumerate() {
        if v as f64 > cutoff {
            match last_kept {
                Some(prev) if i - prev < min_gap => {
                    if v > proj[prev] {
                        if let Some(last) = out.last_mut() {
                            *last = i as u32;
                        }
                        last_kept = Some(i);
                    }
                }
                _ => {
                    out.push(i as u32);
                    last_kept = Some(i);
                }
            }
        }
    }
    out
}

fn wcag_pair(a: [u8; 3], b: [u8; 3]) -> ContrastPair {
    let la = wcag_luminance(a);
    let lb = wcag_luminance(b);
    let (lo, hi) = if la < lb { (la, lb) } else { (lb, la) };
    let ratio = (hi + 0.05) / (lo + 0.05);
    let (fg, bg) = if la < lb { (a, b) } else { (b, a) };
    ContrastPair {
        fg: hex_of(fg),
        bg: hex_of(bg),
        ratio,
        wcag_aa_normal: ratio >= 4.5,
        wcag_aa_large: ratio >= 3.0,
        wcag_aaa_normal: ratio >= 7.0,
    }
}

fn wcag_luminance(rgb: [u8; 3]) -> f64 {
    fn ch(c: u8) -> f64 {
        let cs = c as f64 / 255.0;
        if cs <= 0.03928 {
            cs / 12.92
        } else {
            ((cs + 0.055) / 1.055).powf(2.4)
        }
    }
    0.2126 * ch(rgb[0]) + 0.7152 * ch(rgb[1]) + 0.0722 * ch(rgb[2])
}

fn hex_of(rgb: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])
}

impl Clone for ContrastPair {
    fn clone(&self) -> Self {
        ContrastPair {
            fg: self.fg.clone(),
            bg: self.bg.clone(),
            ratio: self.ratio,
            wcag_aa_normal: self.wcag_aa_normal,
            wcag_aa_large: self.wcag_aa_large,
            wcag_aaa_normal: self.wcag_aaa_normal,
        }
    }
}

impl ContrastPair {
    fn with_ratio_zeroed(mut self) -> Self {
        self.ratio = 1.0; // 1:1 = no contrast at all
        self.wcag_aa_normal = false;
        self.wcag_aa_large = false;
        self.wcag_aaa_normal = false;
        self
    }
}
