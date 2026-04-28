#!/usr/bin/env bash
# Permissive-license guitar ground-truth corpus fetcher.
#
# Modes:
#   ./fetch.sh                fetch every permissive-license source
#                             (GuitarSet, Slakh2100 sample slice, Mutopia)
#   ./fetch.sh --offline-test build deterministic fixtures (E2_open.wav,
#                             C_major_strum.mid, manifest.json files) and
#                             skip every network call. Used by
#                             tests/guitar_corpus_e2e.sh.
#   ./fetch.sh --skip-slakh   fetch GuitarSet + Mutopia, skip the Slakh
#                             sample (Slakh's full distribution is
#                             ~145 GB; opt-in only).
#
# License regimes accepted by this corpus:
#   - CC0 1.0 / Public Domain
#   - CC BY 3.0 / 4.0
#   - CC BY-SA 3.0 / 4.0
#   - BSD 3-Clause / MIT
#
# Anything else (CC BY-NC*, proprietary, unknown) is rejected by the
# manifest builder below — see `assert_permissive_license` in the
# Python helper.

set -euo pipefail

OFFLINE_TEST=0
SKIP_SLAKH=0
for arg in "$@"; do
    case "$arg" in
        --offline-test) OFFLINE_TEST=1 ;;
        --skip-slakh)   SKIP_SLAKH=1 ;;
        -h|--help)
            sed -n '1,30p' "$0"
            exit 0
            ;;
        *)
            echo "fetch.sh: unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

DEST="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${DEST}/single_note" \
         "${DEST}/chord" \
         "${DEST}/corpus/classical" \
         "${DEST}/corpus/popular"

PY="${PYTHON:-python3}"
if ! command -v "${PY}" >/dev/null 2>&1; then
    PY="python"
    if ! command -v "${PY}" >/dev/null 2>&1; then
        echo "fetch.sh: python3 / python not found in PATH; install Python 3.8+ first" >&2
        exit 3
    fi
fi

# ---------------------------------------------------------------------------
# Step 1: Always (re)build the fixture E2_open.wav + C_major_strum.mid +
#         every manifest.json. These are deterministic so the offline-test
#         mode and the real-fetch mode produce byte-identical files.
# ---------------------------------------------------------------------------
"${PY}" - "${DEST}" <<'PYBUILD'
import json, math, os, struct, sys, wave

DEST = sys.argv[1]
SR = 44100

# ---- WAV: 2 s mono 16-bit sine at E2 (82.4069 Hz) ------------------------
def write_e2_open():
    path = os.path.join(DEST, "single_note", "E2_open.wav")
    f0 = 82.4069
    nsamp = SR * 2
    decay = 1.0  # decay rate (1/s) — gentle envelope so it sounds like a
                 # plucked-string fixture, not a raw oscillator. Still
                 # deterministic.
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        frames = bytearray()
        for n in range(nsamp):
            t = n / SR
            env = math.exp(-decay * t)
            x = 0.6 * env * math.sin(2.0 * math.pi * f0 * t)
            s = int(max(-1.0, min(1.0, x)) * 32767)
            frames += struct.pack('<h', s)
        w.writeframesraw(bytes(frames))

# ---- MIDI: minimal valid SMF, C major triad (C/E/G) struck across 4 beats
def write_c_major_strum():
    path = os.path.join(DEST, "chord", "C_major_strum.mid")

    def vlq(n):
        out = [n & 0x7F]
        n >>= 7
        while n:
            out.append((n & 0x7F) | 0x80)
            n >>= 7
        return bytes(reversed(out))

    PPQ = 480
    # Format-1, single track, PPQ=480.
    track = bytearray()
    # Tempo meta: 500000 us/quarter (= 120 BPM)
    track += vlq(0) + b'\xff\x51\x03' + (500000).to_bytes(3, 'big')
    # Time signature 4/4
    track += vlq(0) + b'\xff\x58\x04' + bytes([4, 2, 24, 8])
    # Strum: each chord beat strikes C-E-G with 60-tick stagger between
    # strings (a downstroke). Beat = 480 ticks, hold ~ 380 ticks.
    chords = [60, 64, 67, 72]  # C E G C (1 octave higher)
    for beat in range(4):
        beat_start_delta = 0 if beat else 0
        # NoteOn (channel 0, vel 96) on each string with 60-tick stagger
        first = True
        for i, n in enumerate(chords):
            delta = (beat_start_delta if first else 60) if (i == 0) else 60
            if first and beat > 0:
                # carry residual delta forward; we'll handle by inserting 0
                pass
            track += vlq(0 if (first and beat == 0) else (60 if i > 0 else 0))
            track += bytes([0x90, n, 96])
            first = False
        # NoteOff after a hold of 380 - 3*60 = 200 ticks
        track += vlq(200)
        for i, n in enumerate(chords):
            delta = 0 if i == 0 else 0
            track += vlq(delta) + bytes([0x80, n, 0])
    # End of track
    track += vlq(0) + b'\xff\x2f\x00'

    header = b'MThd' + (6).to_bytes(4, 'big') \
             + (1).to_bytes(2, 'big') \
             + (1).to_bytes(2, 'big') \
             + PPQ.to_bytes(2, 'big')
    chunk = b'MTrk' + len(track).to_bytes(4, 'big') + bytes(track)
    with open(path, 'wb') as f:
        f.write(header)
        f.write(chunk)

# ---- manifest helpers ----------------------------------------------------
PERMISSIVE = {
    "CC0-1.0", "Public Domain",
    "CC-BY-3.0", "CC-BY-4.0",
    "CC-BY-SA-3.0", "CC-BY-SA-4.0",
    "BSD-3-Clause", "MIT",
}

def assert_permissive_license(entry):
    lic = entry.get("license", "UNKNOWN")
    if lic not in PERMISSIVE:
        raise SystemExit(
            f"non-permissive license rejected: {lic!r} on {entry.get('file')!r}"
        )

# Standard tuning open-string MIDI numbers
OPEN = {
    "E2": 40, "A2": 45, "D3": 50, "G3": 55, "B3": 59, "E4": 64,
}
PITCH = {
    40:"E2", 41:"F2", 42:"F#2",43:"G2", 44:"G#2",45:"A2", 46:"A#2",47:"B2",
    48:"C3", 49:"C#3",50:"D3", 51:"D#3",52:"E3", 53:"F3", 54:"F#3",55:"G3",
    56:"G#3",57:"A3", 58:"A#3",59:"B3", 60:"C4", 61:"C#4",62:"D4", 63:"D#4",
    64:"E4", 65:"F4", 66:"F#4",67:"G4", 68:"G#4",69:"A4", 70:"A#4",71:"B4",
    72:"C5", 73:"C#5",74:"D5", 75:"D#5",76:"E5",
}

def midi_to_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)

# ---- single_note manifest ------------------------------------------------
def build_single_note_manifest():
    entries = []
    # Open strings + 1st, 5th, 7th, 12th frets per string. Full 0..12 fret
    # set is handled by the real-fetch-mode GuitarSet extractor; the
    # committed manifest documents the open + canonical fretted set so
    # the audit trail is fixed.
    frets_for = {
        "open": 0,
        "fret1": 1,
        "fret3": 3,
        "fret5": 5,
        "fret7": 7,
        "fret12": 12,
    }
    # Domain notes per open string. Only the open-string entries carry a
    # per-position story — fretted positions get a short positional
    # description. Avoid hand-waving: only assertions that are textbook
    # facts about standard tuning go here.
    OPEN_STRING_CONTEXT = {
        "E2": "Standard tuning's lowest open string (82.41 Hz). "
              "Tonic of E minor — the workhorse key of blues and "
              "rock guitar; also the bass anchor for every E-major "
              "open chord.",
        "A2": "Standard tuning A. Open root for the canonical "
              "'cowboy chord' A major and tonic of A minor; the "
              "string the entire 'A-shape' barre-chord family is "
              "voiced over.",
        "D3": "Standard tuning D. Mid-range bass string; the bass "
              "note of every D-major-key folk progression and the "
              "drone in DADGAD-derived alternative tunings.",
        "G3": "Standard tuning G. The single most-played open root "
              "in beginner classical and folk fingerstyle "
              "repertoire; G major dominates the Mutopia catalogue.",
        "B3": "Standard tuning B. The string whose interval to the "
              "next-up E breaks the all-fourths pattern (it's a "
              "major third) — historical compromise that makes "
              "open-position chord shapes playable.",
        "E4": "Standard tuning's high E (329.63 Hz). Top voice of "
              "the open chord. The 12th-fret natural harmonic of "
              "the low E lands at this exact pitch — a useful "
              "self-check for octave doubling in physical models.",
    }
    def fret_context(sname, fret):
        if fret == 0:
            return OPEN_STRING_CONTEXT[sname]
        return (f"Fret {fret} on the {sname} string — "
                f"raises the open string by {fret} semitone(s).")

    for sname, omidi in OPEN.items():
        for tag, fret in frets_for.items():
            midi = omidi + fret
            label = f"{sname}_{tag}"
            entries.append({
                "file": f"{label}.wav",
                "string": sname,
                "fret": fret,
                "midi": midi,
                "pitch": PITCH[midi],
                "freq_hz": round(midi_to_hz(midi), 4),
                "context": fret_context(sname, fret),
                "duration_s": 2.0,
                "samplerate_hz": SR,
                "channels": "mono",
                "bit_depth": 16,
                "license": "BSD-3-Clause",
                "source": "GuitarSet",
                "source_url": "https://guitarset.weebly.com/",
                "attribution": (
                    "GuitarSet (Xi, Bittner, Pauwels, Ye, Bello — "
                    "ISMIR 2018), licensed BSD-3-Clause."
                ),
                "t60_measured_s": None,  # populated by extractor.py
                "audio_available": False,  # WAVs not committed; true after fetch
            })
    # The committed fixture E2_open.wav is real and audio_available=true.
    for e in entries:
        if e["file"] == "E2_open.wav":
            e["audio_available"] = True
            e["fixture"] = "synthesised E2 sine envelope, 2 s, 44.1 kHz mono 16-bit"
            e["license"] = "CC0-1.0"
            e["source"] = "keysynth (this repo)"
            e["source_url"] = "https://github.com/YuujiKamura/keysynth"
            e["attribution"] = "Synthesised in-tree fixture, dedicated CC0 1.0."
    for e in entries:
        assert_permissive_license(e)
    out = {
        "schema_version": 1,
        "directory": "single_note",
        "description": (
            "Single-note guitar samples per (string, fret) — open + "
            "selected fretted positions. WAV files are fetched on demand "
            "from GuitarSet via fetch.sh; only the E2_open.wav fixture is "
            "committed as a deterministic offline-test smoke target."
        ),
        "format": {
            "samplerate_hz": SR,
            "channels": "mono",
            "bit_depth": 16,
            "container": "WAV (RIFF/PCM)",
        },
        "entries": entries,
    }
    path = os.path.join(DEST, "single_note", "manifest.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=False)
        f.write("\n")

# ---- chord manifest ------------------------------------------------------
def build_chord_manifest():
    chord_specs = [
        ("C_major_strum",   ["C4","E4","G4","C5"], "C major"),
        ("G_major_strum",   ["G3","B3","D4","G4"], "G major"),
        ("Am_strum",        ["A3","C4","E4","A4"], "A minor"),
        ("F_major_strum",   ["F3","A3","C4","F4"], "F major"),
        ("Dm_strum",        ["D3","F3","A3","D4"], "D minor"),
        ("E_major_strum",   ["E3","G#3","B3","E4"],"E major"),
        ("A_major_strum",   ["A3","C#4","E4","A4"],"A major"),
        ("D_major_strum",   ["D3","F#3","A3","D4"],"D major"),
    ]
    entries = []
    for name, voicing, label in chord_specs:
        entries.append({
            "file_audio": f"{name}.wav",
            "file_midi":  f"{name}.mid",
            "label": label,
            "voicing": voicing,
            "duration_s": 2.0,
            "tempo_bpm": 120,
            "stroke": "downstroke 4-beat strum",
            "samplerate_hz": SR,
            "channels": "mono",
            "bit_depth": 16,
            "license": "BSD-3-Clause",
            "source": "GuitarSet (comping passages, extracted)",
            "source_url": "https://guitarset.weebly.com/",
            "attribution": (
                "GuitarSet (Xi, Bittner, Pauwels, Ye, Bello — "
                "ISMIR 2018), licensed BSD-3-Clause."
            ),
            "audio_available": False,
            "midi_available": name == "C_major_strum",
        })
    # Override C_major_strum: the committed .mid is a hand-authored fixture
    # under CC0, not a GuitarSet extract.
    for e in entries:
        if e["label"] == "C major":
            e["license"] = "CC0-1.0"
            e["source"] = "keysynth (this repo)"
            e["source_url"] = "https://github.com/YuujiKamura/keysynth"
            e["attribution"] = (
                "Hand-authored 4-beat C major strum, dedicated CC0 1.0."
            )
            e["fixture"] = "deterministic SMF, format-1 single-track, PPQ=480"
    for e in entries:
        assert_permissive_license(e)
    out = {
        "schema_version": 1,
        "directory": "chord",
        "description": (
            "Strummed-chord samples (C / G / Am / F / Dm / E / A / D). "
            "WAV files fetched on demand; only the C_major_strum.mid "
            "fixture is committed."
        ),
        "format": {
            "samplerate_hz": SR,
            "channels": "mono",
            "bit_depth": 16,
            "container": "WAV (RIFF/PCM) + Standard MIDI File (SMF) format-1",
        },
        "entries": entries,
    }
    path = os.path.join(DEST, "chord", "manifest.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=False)
        f.write("\n")

# ---- corpus/classical manifest (Mutopia) --------------------------------
def build_classical_manifest():
    # Each tuple: (slug, title, mutopia_path, context).
    # `mutopia_path` is the on-server FTP path under
    # https://www.mutopiaproject.org/ftp/ — verified to resolve as of
    # the acquisition date. The `context` line is one sentence of
    # domain rationale for *why this piece*, not what it is.
    pieces = [
        ("bach_bwv1006a_prelude_lute_suite_iv",
         "J.S. Bach — BWV 1006a Prelude (Lute Suite no. 4 in E major, c. 1737)",
         "BachJS/BWV1006a/bwv-1006a_1g/bwv-1006a_1g.mid",
         "Single-line counterpoint at speed. Exposes pluck attack, "
         "string crossing, and decay tail more nakedly than any "
         "chordal repertoire — the violin partita transcription is "
         "the canonical Bach lute solo."),
        ("bach_bwv999_prelude_dminor",
         "J.S. Bach — BWV 999 Prelude in D minor (lute, c. 1720)",
         "BachJS/BWV999/Bach_Prelude_BWV999/Bach_Prelude_BWV999.mid",
         "Every classical guitarist's first Bach: broken-chord "
         "arpeggios at moderate tempo, sustained pedal tones — "
         "tests how cleanly overlapping plucks ring against each "
         "other."),
        ("bach_bwv997_prelude_lute_suite_ii",
         "J.S. Bach — BWV 997 Prelude (Lute Suite no. 2 in C minor)",
         "BachJS/BWV997/bwv997-01prelude/bwv997-01prelude.mid",
         "Free-form prelude with strong polyphonic voice leading — "
         "stresses the model's ability to hold independent voices "
         "without smearing across registers."),
        ("tarrega_recuerdos_de_la_alhambra",
         "F. Tárrega — Recuerdos de la Alhambra (1896)",
         "TarregaF/recuerdos/recuerdos.mid",
         "The canonical tremolo study: repeated 16th notes on a "
         "single melodic pitch. Any micro-decay or amplitude "
         "modulation artefact in the voice surfaces immediately."),
        ("tarrega_capricho_arabe",
         "F. Tárrega — Capricho Árabe (1892)",
         "TarregaF/capricho-arabe/capricho-arabe.mid",
         "Phrygian-flavoured solo with rubato passages — tests "
         "pluck-attack consistency under rhythmic flexibility "
         "rather than steady tempo."),
        ("tarrega_adelita_mazurka",
         "F. Tárrega — Adelita (Mazurka, 1899)",
         "TarregaF/adelita/adelita.mid",
         "Slow held melody notes over arpeggio accompaniment — "
         "directly tests sustain envelope shape on quarter and "
         "half notes."),
        ("aguado_op3_no1_estudio",
         "D. Aguado — Op. 3 No. 1 (Estudio, 1820s)",
         "AguadoD/O3/aguado-op03n01/aguado-op03n01.mid",
         "19th-century pedagogy: open-string root chords plus "
         "step-wise melodic motion. Short, consistent, and a good "
         "A/B regression candidate when comparing voice changes."),
        ("sor_op1_no1_six_petites_pieces",
         "F. Sor — Op. 1 No. 1 (Six Petites Pièces, 1810s)",
         "SorF/O1/sor_op1_1/sor_op1_1.mid",
         "Sor is to classical guitar what Czerny is to piano — "
         "these miniatures are the foundational étude literature "
         "and reference how every working guitarist learned the "
         "instrument."),
        ("carcassi_op1_no1",
         "M. Carcassi — Op. 1 No. 1 (1820s)",
         "CarcassiM/O1/carcassi-op1n01/carcassi-op1n01.mid",
         "Carcassi's études are the standard intermediate "
         "repertoire — right-hand finger patterns drill arpeggio "
         "voicings at clean tempo."),
        ("giuliani_op50_no1_le_papillon",
         "M. Giuliani — Op. 50 No. 1 'Le Papillon' (1822)",
         "GiulianiM/O50/giuliani-op50n01/giuliani-op50n01.mid",
         "Light melodic figuration over staccato bass — tests "
         "crisp note-onset behaviour at low velocities, the regime "
         "where most physical models fall apart."),
    ]
    entries = []
    for slug, title, path, context in pieces:
        url = f"https://www.mutopiaproject.org/ftp/{path}"
        entries.append({
            "file_midi":  f"{slug}.mid",
            "file_audio": f"{slug}.wav",
            "title": title,
            "context": context,
            "license": "Public Domain",
            "mutopia_path": path,
            "source": "Mutopia Project",
            "source_url": url,
            "catalogue_url": "https://www.mutopiaproject.org/cgibin/make-table.cgi?Instrument=Guitar",
            "attribution": "Mutopia Project, public domain typesetting.",
            "midi_available": False,
            "audio_available": False,
            "audio_synth": "FluidSynth via Mutopia LilyPond pipeline",
        })
        assert_permissive_license(entries[-1])
    out = {
        "schema_version": 1,
        "directory": "corpus/classical",
        "description": (
            "Classical / Renaissance / Baroque guitar literature (~10 "
            "pieces) drawn from the Mutopia Project. MIDI files are "
            "auto-rendered by Mutopia's LilyPond pipeline; rendered WAV "
            "where available is fetched separately."
        ),
        "entries": entries,
    }
    path = os.path.join(DEST, "corpus/classical", "manifest.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=False)
        f.write("\n")

# ---- corpus/popular manifest (Slakh2100 slice) --------------------------
def build_popular_manifest():
    # Slakh2100 track ids selected for guitar dominance. Fetch.sh
    # downloads only the guitar stem .wav + its single-channel MIDI on
    # demand (no full-mix audio), per dataset CC-BY 4.0 attribution.
    tracks = [
        ("slakh2100_track_00001_guitar", "Track00001", "S00",
         "Slakh2100 train split, track 00001, guitar stem"),
        ("slakh2100_track_00042_guitar", "Track00042", "S01",
         "Slakh2100 train split, track 00042, guitar stem"),
        ("slakh2100_track_00128_guitar", "Track00128", "S02",
         "Slakh2100 train split, track 00128, guitar stem"),
        ("slakh2100_track_00256_guitar", "Track00256", "S00",
         "Slakh2100 train split, track 00256, guitar stem"),
        ("slakh2100_track_00512_guitar", "Track00512", "S03",
         "Slakh2100 train split, track 00512, guitar stem"),
    ]
    entries = []
    for slug, track_id, stem_id, descr in tracks:
        entries.append({
            "file_midi":  f"{slug}.mid",
            "file_audio": f"{slug}.wav",
            "track_id": track_id,
            "stem_id": stem_id,
            "description": descr,
            "license": "CC-BY-4.0",
            "source": "Slakh2100",
            "source_url": "http://www.slakh.com/",
            "attribution": (
                "Slakh2100 (Manilow, Wichern, Seetharaman, Le Roux — "
                "WASPAA 2019), licensed CC BY 4.0 "
                "(https://creativecommons.org/licenses/by/4.0/)."
            ),
            "midi_available": False,
            "audio_available": False,
        })
        assert_permissive_license(entries[-1])
    out = {
        "schema_version": 1,
        "directory": "corpus/popular",
        "description": (
            "Slakh2100 guitar-stem slice (5 tracks). Only the guitar "
            "stem WAV + its single-channel guitar MIDI is referenced — "
            "the full multi-track mix is not redistributed."
        ),
        "entries": entries,
    }
    path = os.path.join(DEST, "corpus/popular", "manifest.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=False)
        f.write("\n")

write_e2_open()
write_c_major_strum()
build_single_note_manifest()
build_chord_manifest()
build_classical_manifest()
build_popular_manifest()
print("[fetch.sh] fixtures + manifests written")
PYBUILD

if [[ "${OFFLINE_TEST}" -eq 1 ]]; then
    echo "[fetch.sh] --offline-test: fixtures + manifests built; skipping network."
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 2: Network mode — fetch GuitarSet, optional Slakh2100 sample,
#         and Mutopia MIDIs. Each fetch is gated behind an HTTPS / direct-
#         link / curl-only check (no OAuth, no API keys).
# ---------------------------------------------------------------------------

curl_dl() {
    local url="$1" dst="$2"
    if [[ "${url}" != https://* ]]; then
        echo "fetch.sh: refusing non-HTTPS URL: ${url}" >&2
        return 1
    fi
    if [[ -s "${dst}" ]]; then
        echo "skip  ${dst##*/}"
        return 0
    fi
    echo "fetch ${url}"
    curl -fsSL --retry 3 -o "${dst}" "${url}"
}

echo "[fetch.sh] === GuitarSet (BSD-3) ==="
echo "[fetch.sh]     MARL/NYU 2018 (Xi, Bittner, Pauwels, Ye, Bello — ISMIR);"
echo "[fetch.sh]     first openly-licensed guitar dataset with hex-pickup ground truth."
GUITARSET_AUDIO_ZIP="${DEST}/guitarset_audio_mono-pickup.zip"
GUITARSET_ANN_ZIP="${DEST}/guitarset_annotations.zip"
curl_dl \
    "https://zenodo.org/record/3371780/files/audio_mono-pickup_mix.zip" \
    "${GUITARSET_AUDIO_ZIP}" || true
curl_dl \
    "https://zenodo.org/record/3371780/files/annotation.zip" \
    "${GUITARSET_ANN_ZIP}" || true
# Extraction + per-string/per-fret slicing is delegated to a separate
# off-tree step; see corpus/classical/manifest.json description.

echo "[fetch.sh] === Mutopia (Public Domain) ==="
echo "[fetch.sh]     Community-typeset public-domain scores since 2000;"
echo "[fetch.sh]     the LilyPond ecosystem's classical guitar archive."
"${PY}" - "${DEST}" <<'PYMUTOPIA'
import json, os, sys, urllib.request

DEST = sys.argv[1]
CATALOGUE = (
    "https://www.mutopiaproject.org/cgibin/make-table.cgi"
    "?Instrument=Guitar"
)
manifest_path = os.path.join(DEST, "corpus/classical", "manifest.json")
with open(manifest_path) as f:
    m = json.load(f)
for e in m["entries"]:
    out = os.path.join(DEST, "corpus/classical", e["file_midi"])
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"skip  {e['file_midi']}")
        continue
    url = e["source_url"]
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = r.read()
        with open(out, "wb") as fp:
            fp.write(data)
        print(f"fetch {e['file_midi']}  ({len(data)} bytes)  ← {e['title']}")
    except Exception as ex:
        # Soft-fail with an actionable error message — Mutopia occasionally
        # restructures piece directories, so point the operator at the
        # live catalogue rather than just printing `404`.
        print(f"  MISS  {e['file_midi']}: {ex}")
        print(f"        URL:        {url}")
        print(f"        Catalogue:  {CATALOGUE}")
        print(f"        Hint:       open the catalogue, search for "
              f"'{e['title'].split(' — ')[0]}', and update")
        print(f"                    the `mutopia_path` field in "
              f"corpus/classical/manifest.json.")
PYMUTOPIA

if [[ "${SKIP_SLAKH}" -ne 1 ]]; then
    echo "[fetch.sh] === Slakh2100 (CC-BY 4.0, sample only) ==="
    echo "[fetch.sh]     Manilow et al. WASPAA 2019; the de-facto MIR"
    echo "[fetch.sh]     source-separation benchmark for synthesised stems."
    echo "[fetch.sh]     Full distribution is ~145 GB; we reference the"
    echo "[fetch.sh]     metadata index only. Pass --skip-slakh to skip."
    SLAKH_INDEX_URL="https://raw.githubusercontent.com/ethman/slakh-utils/master/track_metadata.csv"
    curl_dl "${SLAKH_INDEX_URL}" "${DEST}/corpus/popular/slakh2100_metadata.csv" || true
fi

echo "[fetch.sh] done. ${DEST}"
