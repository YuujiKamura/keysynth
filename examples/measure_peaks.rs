use keysynth::synth::{make_voice, midi_to_freq, Engine, ModalLut, MODAL_LUT};

fn main() {
    let (lut, src) = ModalLut::auto_load(None);
    let _ = MODAL_LUT.set(lut);
    eprintln!("LUT source: {src}");

    const SR: f32 = 44100.0;
    let render = |engine: Engine, note: u8, secs: f32| -> f32 {
        let freq = midi_to_freq(note);
        let mut v = make_voice(engine, SR, freq, 100);
        let n = (SR * secs) as usize;
        let mut buf = vec![0.0_f32; n];
        v.render_add(&mut buf);
        buf.iter().fold(0.0_f32, |a, &x| a.max(x.abs()))
    };

    for engine in [
        Engine::PianoModal,
        Engine::PianoLite,
        Engine::PianoThick,
        Engine::Piano,
        Engine::Piano5AM,
        Engine::Square,
        Engine::Ks,
    ] {
        let p = render(engine, 53, 1.0);
        let db = 20.0 * p.max(1e-9).log10();
        eprintln!("engine={engine:?} note=53 raw_peak={p:.5} ({db:.1} dBFS)");
    }
}
