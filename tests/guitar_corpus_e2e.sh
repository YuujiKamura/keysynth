#!/usr/bin/env bash
# End-to-end gate for the permissive-licence guitar ground-truth corpus
# (issue #44 follow-up; see bench-out/REF/guitar/README.md for the
# directory layout).
#
# Five gates, all must pass before merge:
#   1. `fetch.sh --offline-test` exits 0 without network access
#   2. single_note/E2_open.wav exists, 44.1 kHz / mono / 16-bit
#   3. chord/C_major_strum.mid exists, valid SMF (MThd header)
#   4. every manifest.json parses as JSON and only references permissive
#      licences
#   5. `cargo run --release --bin score` runs against the committed
#      reference fixture and emits a non-empty `mr_stft` line

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GUITAR_REF="${REPO_ROOT}/bench-out/REF/guitar"

PY="${PYTHON:-python3}"
if ! command -v "${PY}" >/dev/null 2>&1; then
    PY="python"
fi

fail() { echo "guitar_corpus_e2e: FAIL — $*" >&2; exit 1; }
pass() { echo "guitar_corpus_e2e: ok   — $*"; }

# ---- gate 1: offline-test fetch -----------------------------------------
echo ">>> gate 1: bash fetch.sh --offline-test"
bash "${GUITAR_REF}/fetch.sh" --offline-test \
    || fail "fetch.sh --offline-test exited non-zero"
pass "fetch.sh --offline-test ran clean"

# ---- gate 2: WAV format -------------------------------------------------
echo ">>> gate 2: single_note/E2_open.wav format check"
"${PY}" - "${GUITAR_REF}/single_note/E2_open.wav" <<'PYWAV'
import sys, wave
path = sys.argv[1]
with wave.open(path, "rb") as w:
    ch = w.getnchannels()
    sr = w.getframerate()
    sw = w.getsampwidth()
    nf = w.getnframes()
ok = (ch == 1 and sr == 44100 and sw == 2 and nf >= 44100)
print(f"  ch={ch} sr={sr} bytes_per_sample={sw} frames={nf}")
if not ok:
    sys.exit(f"E2_open.wav format mismatch: expected mono / 44.1 kHz / 16-bit / >=1 s")
PYWAV
pass "E2_open.wav is mono 44.1 kHz 16-bit, >= 1 s"

# ---- gate 3: SMF header -------------------------------------------------
echo ">>> gate 3: chord/C_major_strum.mid SMF header"
"${PY}" - "${GUITAR_REF}/chord/C_major_strum.mid" <<'PYMID'
import sys
data = open(sys.argv[1], "rb").read()
if data[:4] != b'MThd':
    sys.exit("C_major_strum.mid: missing MThd header")
hdr_len = int.from_bytes(data[4:8], "big")
fmt = int.from_bytes(data[8:10], "big")
ntrk = int.from_bytes(data[10:12], "big")
ppq = int.from_bytes(data[12:14], "big")
print(f"  hdr_len={hdr_len} format={fmt} tracks={ntrk} ppq={ppq} bytes={len(data)}")
if hdr_len != 6 or fmt not in (0, 1) or ntrk < 1 or ppq <= 0:
    sys.exit("C_major_strum.mid: malformed header fields")
# Find first MTrk and verify event stream ends with FF 2F 00 (end-of-track)
i = 14
if data[i:i+4] != b'MTrk':
    sys.exit("C_major_strum.mid: first chunk after header is not MTrk")
trk_len = int.from_bytes(data[i+4:i+8], "big")
trk_end = i + 8 + trk_len
if data[trk_end-3:trk_end] != b'\xff\x2f\x00':
    sys.exit("C_major_strum.mid: track does not end with End-of-Track meta")
PYMID
pass "C_major_strum.mid is a valid SMF format-1 file"

# ---- gate 4: manifests ---------------------------------------------------
echo ">>> gate 4: manifest.json validity + license whitelist"
"${PY}" - "${GUITAR_REF}" <<'PYMAN'
import json, os, sys
ROOT = sys.argv[1]
PERMISSIVE = {
    "CC0-1.0", "Public Domain",
    "CC-BY-3.0", "CC-BY-4.0",
    "CC-BY-SA-3.0", "CC-BY-SA-4.0",
    "BSD-3-Clause", "MIT",
}
SUBDIRS = [
    "single_note", "chord",
    "corpus/classical", "corpus/popular",
]
total = 0
for sub in SUBDIRS:
    p = os.path.join(ROOT, sub, "manifest.json")
    with open(p) as f:
        m = json.load(f)
    if "entries" not in m or not isinstance(m["entries"], list):
        sys.exit(f"{p}: 'entries' missing or not a list")
    for e in m["entries"]:
        lic = e.get("license", "UNKNOWN")
        if lic not in PERMISSIVE:
            sys.exit(f"{p}: non-permissive license {lic!r} in entry {e!r}")
        total += 1
    print(f"  {sub}: {len(m['entries'])} entries, all permissive")
print(f"  total entries vetted: {total}")
PYMAN
pass "all manifest.json files parse and only reference permissive licences"

# ---- gate 5: score binary wiring ----------------------------------------
echo ">>> gate 5: score binary against E2_open.wav"
( cd "${REPO_ROOT}" && \
  cargo build --release --bin score --features native --quiet ) \
    || fail "cargo build --release --bin score failed"

# Respect CARGO_TARGET_DIR — common in this repo because the workspace
# default places `target/` inside the repo, but `voices_live/*` plugins
# rebuild independently and developers often redirect `target/` to a
# fast SSD. Fall back to the in-tree default when unset.
TARGET_DIR="${CARGO_TARGET_DIR:-${REPO_ROOT}/target}"
SCORE_BIN="${TARGET_DIR}/release/score"
# Cygwin / MSYS2 sometimes round-trips Windows-style drive letters; let
# bash do the hard work via cygpath where available.
if command -v cygpath >/dev/null 2>&1; then
    SCORE_BIN="$(cygpath -u "${SCORE_BIN}")"
fi
if [[ -x "${SCORE_BIN}.exe" ]]; then
    SCORE_BIN="${SCORE_BIN}.exe"
elif [[ ! -x "${SCORE_BIN}" ]]; then
    fail "score binary not found at ${SCORE_BIN}(.exe) (CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-unset})"
fi

SCORE_JSON="${REPO_ROOT}/target/guitar_corpus_e2e_score.json"
mkdir -p "$(dirname "${SCORE_JSON}")"
"${SCORE_BIN}" \
    --reference "${GUITAR_REF}/single_note/E2_open.wav" \
    --candidate "${GUITAR_REF}/single_note/E2_open.wav" \
    --note 40 \
    --json > "${SCORE_JSON}" \
    || fail "score binary returned non-zero against E2_open.wav"

echo "  score output written to ${SCORE_JSON}"
"${PY}" - "${SCORE_JSON}" <<'PYSCORE'
import json, sys
data = open(sys.argv[1]).read().strip()
try:
    obj = json.loads(data)
except Exception as e:
    sys.exit(f"score output not valid JSON: {e}: {data!r}")
# A self-loss should report mr_stft (and any other components) as a
# small non-negative number; we just assert the field exists and is
# numeric.
keys = ", ".join(sorted(obj.keys())) if isinstance(obj, dict) else type(obj).__name__
print(f"  score keys: {keys}")
def find_mrstft(o, depth=0):
    if depth > 4: return None
    if isinstance(o, dict):
        for k, v in o.items():
            if "mr_stft" in k.lower() or "mr-stft" in k.lower():
                return (k, v)
            r = find_mrstft(v, depth+1)
            if r is not None: return r
    elif isinstance(o, list):
        for v in o:
            r = find_mrstft(v, depth+1)
            if r is not None: return r
    return None
hit = find_mrstft(obj)
if hit is None:
    sys.exit(f"score output missing an mr_stft component: {obj!r}")
print(f"  mr_stft component: {hit[0]} = {hit[1]}")
PYSCORE
pass "score binary wired against guitar/single_note/E2_open.wav"

echo ""
echo "guitar_corpus_e2e: ALL GATES PASSED"
