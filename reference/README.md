# Reference recordings for analyse

The `analyse` binary needs a *real piano recording* as ground truth. Sample
libraries (SoundFonts) are not real pianos -- they're stitched-together
multi-velocity samples with crossfades, which makes them an unreliable
reference.

## University of Iowa Musical Instrument Samples (MIS) — recommended

The Iowa MIS dataset (Lawrence Fritts, University of Iowa Electronic Music
Studios) provides isolated single-note recordings of a Steinway concert
grand, free for educational and research use.

- **Source**: https://theremin.music.uiowa.edu/MISpiano.html
- **License**: free for non-commercial educational/research use
- **Files we want**: `Piano.ff.C4.aif` (and ideally `.mf.C4.aif`,
  `.pp.C4.aif` for velocity coverage)

### Acquisition steps

1. Visit the URL above
2. Download the appropriate ZIP (e.g. "Piano - C4 to C5") or individual AIF
3. Convert AIF → WAV mono 44100 Hz 16-bit using one of:
   - `ffmpeg -i Piano.ff.C4.aif -ac 1 -ar 44100 -sample_fmt s16 reference/iowa_piano_ff_c4.wav`
   - `sox Piano.ff.C4.aif -c 1 -r 44100 -b 16 reference/iowa_piano_ff_c4.wav`
4. Place under `reference/` (gitignored).
5. Verify SHA-256 against `reference/expected_hashes.txt` (see below).

### Expected files

| Path | MIDI note | Velocity | Source filename |
|---|---|---|---|
| `reference/iowa_piano_ff_c4.wav` | 60 | ff (~110) | Piano.ff.C4.aif |
| `reference/iowa_piano_mf_c4.wav` | 60 | mf (~80) | Piano.mf.C4.aif |
| `reference/iowa_piano_pp_c4.wav` | 60 | pp (~40) | Piano.pp.C4.aif |

The ff (fortissimo) C4 is the canonical baseline used throughout the
analyse documentation.

## MAPS dataset (alternative)

For multi-piano coverage and ISO-grade benchmarking:

- **Source**: https://hal.archives-ouvertes.fr/hal-00544155/
- **License**: see dataset README (free for academic use)
- **Use**: more comprehensive but heavier; defer to MIS for single-note work.

## Why these and not GeneralUser GS / FluidR3

SoundFonts are pre-rendered samples wrapped in a synthesis engine. Round-
tripping a SoundFont through rustysynth and calling the result "the
reference" is circular -- you've measured how well your synth matches
*another synth*, not how well it matches a real piano. The Iowa MIS files
are direct stereo close-mic recordings of an actual instrument played by
a person.

## Why nothing is committed here

Sample recordings are typically distributed under licences that prohibit
redistribution within derived repositories (even when the underlying use
is free). They also balloon git history. We document the URL + expected
hash; users fetch and place files locally.
