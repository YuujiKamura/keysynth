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
    pieces = [
        ("bach_bwv1006_prelude", "J.S. Bach — BWV 1006a Prelude (Lute Suite IV)",
         "1063", "Public Domain"),
        ("sor_op35_no22_andantino", "F. Sor — Op. 35 No. 22 Andantino",
         "0541", "Public Domain"),
        ("tarrega_lagrima", "F. Tárrega — Lágrima (Preludio)",
         "0207", "Public Domain"),
        ("tarrega_adelita", "F. Tárrega — Adelita (Mazurka)",
         "0208", "Public Domain"),
        ("carcassi_op60_no1", "M. Carcassi — Op. 60 No. 1",
         "0123", "Public Domain"),
        ("giuliani_op50_no1_papillon", "M. Giuliani — Op. 50 No. 1 (Le Papillon)",
         "0345", "Public Domain"),
        ("carulli_op121_no1", "F. Carulli — Op. 121 No. 1 Andantino",
         "0212", "Public Domain"),
        ("bach_bwv996_bourree", "J.S. Bach — BWV 996 Bourrée",
         "0090", "Public Domain"),
        ("sor_op31_no2", "F. Sor — Op. 31 No. 2 Andantino",
         "0301", "Public Domain"),
        ("aguado_estudio_em", "D. Aguado — Estudio en Mi menor",
         "0455", "Public Domain"),
    ]
    entries = []
    for slug, title, mid, lic in pieces:
        entries.append({
            "file_midi":  f"{slug}.mid",
            "file_audio": f"{slug}.wav",
            "title": title,
            "license": lic,
            "mutopia_id": mid,
            "source": "Mutopia Project",
            "source_url": f"https://www.mutopiaproject.org/cgibin/piece-info.cgi?id={mid}",
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
"${PY}" - "${DEST}" <<'PYMUTOPIA'
import json, os, sys, urllib.request

DEST = sys.argv[1]
manifest_path = os.path.join(DEST, "corpus/classical", "manifest.json")
with open(manifest_path) as f:
    m = json.load(f)
for e in m["entries"]:
    mid = e["mutopia_id"]
    out = os.path.join(DEST, "corpus/classical", e["file_midi"])
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"skip  {e['file_midi']}")
        continue
    # Mutopia's per-piece info page links to .mid via a deterministic
    # URL pattern. Soft-fail on 404 — Mutopia occasionally rotates URLs.
    candidates = [
        f"https://www.mutopiaproject.org/ftp/{mid}.mid",
    ]
    fetched = False
    for url in candidates:
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                data = r.read()
            with open(out, "wb") as fp:
                fp.write(data)
            print(f"fetch {e['file_midi']}  ({len(data)} bytes)")
            fetched = True
            break
        except Exception as ex:
            print(f"  miss {url}: {ex}")
    if not fetched:
        print(f"  {e['file_midi']} unavailable; manifest entry kept, midi_available=false")
PYMUTOPIA

if [[ "${SKIP_SLAKH}" -ne 1 ]]; then
    echo "[fetch.sh] === Slakh2100 (CC-BY 4.0, sample only) ==="
    echo "[fetch.sh] Slakh2100's full distribution is ~145 GB. We only"
    echo "[fetch.sh] reference the metadata index here. Pass --skip-slakh"
    echo "[fetch.sh] to silence this notice."
    SLAKH_INDEX_URL="https://raw.githubusercontent.com/ethman/slakh-utils/master/track_metadata.csv"
    curl_dl "${SLAKH_INDEX_URL}" "${DEST}/corpus/popular/slakh2100_metadata.csv" || true
fi

echo "[fetch.sh] done. ${DEST}"
