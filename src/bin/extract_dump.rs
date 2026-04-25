//! One-shot debug helper: dump decompose() + fit_b() + extract_t60() +
//! extract_attack() output for a given WAV. Used to pin golden values
//! for the issue #3 extractors.
//!
//! Usage: `cargo run --release --bin extract_dump -- <path.wav> <f0_hz>`

use std::env;
use std::path::Path;

use hound::WavReader;
use keysynth::extract::attack::extract_attack;
use keysynth::extract::decompose::{decompose, Partial};
use keysynth::extract::inharmonicity::fit_b;
use keysynth::extract::t60::extract_t60;

fn read_wav_mono(path: &Path) -> Option<(Vec<f32>, f32)> {
    let mut r = WavReader::open(path).ok()?;
    let spec = r.spec();
    let sr = spec.sample_rate as f32;
    let channels = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().filter_map(Result::ok).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / max)
                .collect()
        }
    };
    if channels <= 1 {
        Some((samples, sr))
    } else {
        let mut mono = Vec::with_capacity(samples.len() / channels);
        for frame in samples.chunks_exact(channels) {
            let sum: f32 = frame.iter().sum();
            mono.push(sum / channels as f32);
        }
        Some((mono, sr))
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: extract_dump <path.wav> <f0_hz>");
        std::process::exit(2);
    }
    let path = Path::new(&args[1]);
    let f0: f32 = args[2].parse().expect("bad f0_hz");
    let (sig, sr) = read_wav_mono(path).expect("could not read wav");
    println!(
        "=== {} ({:.0} Hz, {:.2} s) ===",
        path.display(),
        sr,
        sig.len() as f32 / sr
    );

    let partials: Vec<Partial> = decompose(&sig, sr, f0, 16);
    println!("\n--- decompose: {} partials ---", partials.len());
    for p in &partials {
        println!(
            "  n={:>2}  freq={:>9.3} Hz  init_db={:>+7.2}",
            p.n, p.freq_hz, p.init_db
        );
    }

    let fit = fit_b(&partials);
    println!("\n--- fit_b ---");
    println!("  B          = {:.4e}", fit.b);
    println!("  R^2        = {:.4}", fit.r_squared);
    println!("  n_used     = {}", fit.n_used);

    let t60 = extract_t60(&sig, sr, &partials);
    println!("\n--- extract_t60 ---");
    for (i, t) in t60.seconds.iter().enumerate() {
        let n = partials.get(i).map(|p| p.n).unwrap_or(0);
        if *t < 0.0 {
            println!("  n={:>2}  T60=  -- (no decay)", n);
        } else {
            println!("  n={:>2}  T60={:>6.2} s", n, t);
        }
    }

    let att = extract_attack(&sig, sr, 100.0);
    println!("\n--- extract_attack (100 ms window) ---");
    println!("  time_to_peak_s        = {:.4}", att.time_to_peak_s);
    println!("  peak_db               = {:.2}", att.peak_db);
    println!("  post_peak_slope_db_s  = {:.2}", att.post_peak_slope_db_s);
}
