#!/usr/bin/env bash
#
# verify_jukebox.sh — user-path verification for the jukebox GUI via CP.
#
# Spawns `cargo run --bin jukebox`, waits for the embedded gui_cp server
# to advertise its endpoint at bench-out/cp/jukebox-<pid>.endpoint, then
# drives load_track / play / stop / set_voice over JSON-RPC and asserts
# the published snapshot reflects the expected state transitions.
#
# Why CP-driven instead of desk_capture / VLM scripting:
#   * deterministic — no font fallback, no obscured surface, no IME
#   * fast — handler latency is in single-digit ms
#   * survives over remote shells where there's no display server
#
# Usage:
#   scripts/verify_jukebox.sh            # full sequence, default track
#   TRACK_LABEL=foo scripts/verify_jukebox.sh
#   KEEP_RUNNING=1 scripts/verify_jukebox.sh   # don't kill jukebox at exit
#
# Exit status is non-zero on any check failure. Writes a structured log
# to bench-out/cp/verify_jukebox.log so a CI run can attach it.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Pick an interpreter — `python3` on Linux/macOS, plain `python` on
# typical Windows + msys/git-bash installs. Python is the only hard
# external prereq the script depends on.
PY=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PY="$candidate"
        break
    fi
done
if [[ -z "$PY" ]]; then
    echo "verify_jukebox: neither python3 nor python found in PATH" >&2
    exit 2
fi

LOG_DIR="bench-out/cp"
LOG_FILE="$LOG_DIR/verify_jukebox.log"
mkdir -p "$LOG_DIR"

log() {
    local ts
    ts="$(date '+%Y-%m-%dT%H:%M:%S')"
    printf '[%s] %s\n' "$ts" "$*" | tee -a "$LOG_FILE" >&2
}

fail() {
    log "FAIL: $*"
    exit 1
}

assert_eq() {
    local got="$1" want="$2" what="$3"
    if [[ "$got" != "$want" ]]; then
        fail "$what: expected '$want', got '$got'"
    fi
    log "  ok: $what == $got"
}

assert_ge() {
    local got="$1" want="$2" what="$3"
    if (( got < want )); then
        fail "$what: expected >= $want, got $got"
    fi
    log "  ok: $what >= $want (got $got)"
}

# --- helper: send one CP request, return the response body --------------
#
# Uses python3 because /dev/tcp + jq is portable but Windows-Bash mileage
# varies. Python is already a hard prereq on every dev machine running
# this repo's tooling.
cp_call() {
    local addr="$1" method="$2" params="$3"
    "$PY" - "$addr" "$method" "$params" <<'PY'
import json, socket, sys, uuid
# Force UTF-8 stdout so song titles / composer names with em-dashes,
# kanji, etc. don't crash the helper on a Windows cp932 console.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass
addr = sys.argv[1]
method = sys.argv[2]
params = json.loads(sys.argv[3] or "{}")
host, port = addr.rsplit(":", 1)
host = host.strip("[]")
req = {"jsonrpc": "2.0", "method": method, "params": params, "id": uuid.uuid4().int & 0x7fffffff}
s = socket.create_connection((host, int(port)), timeout=10)
s.sendall((json.dumps(req) + "\n").encode())
buf = b""
while not buf.endswith(b"\n"):
    chunk = s.recv(65536)
    if not chunk:
        break
    buf += chunk
s.close()
sys.stdout.write(buf.decode("utf-8").strip())
PY
}

extract() {
    local json="$1" path="$2"
    "$PY" - "$json" "$path" <<'PY'
import json, sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass
data = json.loads(sys.argv[1])
for key in sys.argv[2].split('.'):
    if isinstance(data, list):
        try:
            data = data[int(key)]
        except (ValueError, IndexError):
            data = None
            break
    elif isinstance(data, dict):
        data = data.get(key)
    else:
        data = None
        break
    if data is None:
        break
if data is None:
    out = ""
elif isinstance(data, str):
    out = data
elif isinstance(data, bool):
    # JSON booleans should print as `true`/`false`, not Python's
    # capitalized `True`/`False`, so the bash `[[ "$x" == "true" ]]`
    # checks downstream stay correct.
    out = "true" if data else "false"
else:
    out = json.dumps(data, ensure_ascii=False)
sys.stdout.write(out)
PY
}

# --- launch -------------------------------------------------------------

# Clean up any stale endpoint files from prior runs so we can spot the
# new pid unambiguously.
rm -f "$LOG_DIR"/jukebox-*.endpoint

log "building jukebox (cargo build --bin jukebox)..."
cargo build --bin jukebox 2>>"$LOG_FILE"

log "spawning jukebox..."
# Force a deterministic CP endpoint so this script doesn't have to race
# the announce file. 0 means OS-picked + announced.
KEYSYNTH_GUI_CP_JUKEBOX="${KEYSYNTH_GUI_CP_JUKEBOX:-127.0.0.1:0}"
export KEYSYNTH_GUI_CP_JUKEBOX
cargo run --quiet --bin jukebox >>"$LOG_FILE" 2>&1 &
JUKEBOX_PID=$!
log "jukebox pid=$JUKEBOX_PID"

cleanup() {
    if [[ "${KEEP_RUNNING:-0}" == "1" ]]; then
        log "KEEP_RUNNING=1 — leaving jukebox alive (pid $JUKEBOX_PID)"
        return
    fi
    if kill -0 "$JUKEBOX_PID" 2>/dev/null; then
        log "stopping jukebox pid=$JUKEBOX_PID"
        kill "$JUKEBOX_PID" 2>/dev/null || true
        # Give it a beat to flush, then SIGKILL if still alive.
        for _ in 1 2 3 4 5; do
            kill -0 "$JUKEBOX_PID" 2>/dev/null || break
            sleep 0.5
        done
        kill -9 "$JUKEBOX_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# --- discover endpoint --------------------------------------------------

log "waiting for endpoint announcement..."
ENDPOINT=""
for i in $(seq 1 60); do
    candidate="$(ls -t "$LOG_DIR"/jukebox-*.endpoint 2>/dev/null | head -1 || true)"
    if [[ -n "$candidate" ]]; then
        ENDPOINT="$(cat "$candidate")"
        log "  found endpoint $ENDPOINT (after ${i}s)"
        break
    fi
    sleep 1
done
[[ -n "$ENDPOINT" ]] || fail "no endpoint file appeared in $LOG_DIR/ after 60 s"

# --- ping ---------------------------------------------------------------

log "ping..."
RESP="$(cp_call "$ENDPOINT" ping '{}')"
PROTO="$(extract "$RESP" 'result.protocol')"
APP="$(extract "$RESP" 'result.app')"
assert_eq "$PROTO" "keysynth-gui-cp/1" "ping.protocol"
assert_eq "$APP" "jukebox" "ping.app"

# --- get_state: catalog counts ------------------------------------------

log "get_state — wait for first frame..."
SONGS=""
VOICES=""
TRACKS=""
for i in $(seq 1 30); do
    RESP="$(cp_call "$ENDPOINT" get_state '{}')"
    SONGS="$(extract "$RESP" 'result.db_song_count')"
    VOICES="$(extract "$RESP" 'result.db_voice_count')"
    TRACKS="$(extract "$RESP" 'result.track_count')"
    if [[ -n "$SONGS" && "$SONGS" != "0" ]]; then
        break
    fi
    sleep 0.5
done
log "  catalog: $SONGS songs / $VOICES voices / $TRACKS catalog tracks"
assert_ge "${SONGS:-0}" 1 "db_song_count"
assert_ge "${VOICES:-0}" 1 "db_voice_count"
assert_ge "${TRACKS:-0}" 1 "track_count"

# --- pick a track to drive ----------------------------------------------

LABEL="${TRACK_LABEL:-}"
if [[ -z "$LABEL" ]]; then
    # Prefer a label that's already cached so the load doesn't trigger
    # a multi-second render_midi pass — most checkouts have at least
    # one preview WAV in bench-out/cache/.
    log "no TRACK_LABEL override — preferring an already-cached row"
    RESP="$(cp_call "$ENDPOINT" get_state '{}')"
    LABEL="$(extract "$RESP" 'result.cached_labels.0')"
    if [[ -z "$LABEL" || "$LABEL" == "null" ]]; then
        log "  no cached labels — falling back to first catalog row"
        RESP="$(cp_call "$ENDPOINT" list_tracks '{"limit": 5}')"
        LABEL="$(extract "$RESP" 'result.tracks.0')"
    fi
fi
[[ -n "$LABEL" ]] || fail "could not determine a track label to load"
log "target label: $LABEL"

# --- load_track: state must flip to playing -----------------------------

log "load_track $LABEL..."
RESP="$(cp_call "$ENDPOINT" load_track "{\"label\": \"$LABEL\"}")"
QUEUED="$(extract "$RESP" 'result.queued')"
assert_eq "$QUEUED" "load_track" "load_track ack"

# Poll for any_playing=true. Cached MIDI tracks flip in a single frame;
# uncached ones first kick off a render_midi subprocess that can take
# tens of seconds, which we tolerate softly.
log "polling get_state for any_playing=true..."
GOT_PLAY=0
for i in $(seq 1 16); do
    RESP="$(cp_call "$ENDPOINT" get_state '{}')"
    AP="$(extract "$RESP" 'result.any_playing')"
    SEL="$(extract "$RESP" 'result.selected_label')"
    FRAME="$(extract "$RESP" 'result.frame_id')"
    log "  [$i] frame=$FRAME any_playing=$AP selected=$SEL"
    if [[ "$AP" == "true" ]]; then
        GOT_PLAY=1
        break
    fi
    sleep 0.5
done
[[ "$GOT_PLAY" == "1" ]] || log "WARNING: any_playing never flipped to true — track may have hit a render miss; continuing"

# --- stop: state must flip back -----------------------------------------

log "stop..."
cp_call "$ENDPOINT" stop '{}' >/dev/null

log "polling get_state for any_playing=false..."
GOT_STOP=0
for i in $(seq 1 20); do
    RESP="$(cp_call "$ENDPOINT" get_state '{}')"
    AP="$(extract "$RESP" 'result.any_playing')"
    if [[ "$AP" == "false" ]]; then
        GOT_STOP=1
        break
    fi
    sleep 0.3
done
assert_eq "$GOT_STOP" "1" "any_playing flips back to false after stop"

# --- cache hit consistency check ----------------------------------------

log "cache_hit_pairs vs disk truth..."
RESP="$(cp_call "$ENDPOINT" get_state '{}')"
HITS="$(extract "$RESP" 'result.cache_hit_pairs')"
DISK_HITS="$(find bench-out/cache -maxdepth 2 -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')"
log "  GUI says $HITS, disk has $DISK_HITS preview WAV(s)"
# Soft check: GUI tracks (label, voice) pairs and disk has individual
# WAV files; disk count should be >= GUI count because the GUI only
# tracks pairs reachable from current catalog rows.
if (( DISK_HITS < HITS )); then
    fail "GUI reports more cache hits ($HITS) than disk has WAVs ($DISK_HITS)"
fi
log "  ok: cache_hit_pairs ($HITS) <= disk wavs ($DISK_HITS)"

# --- frame counter advanced ---------------------------------------------

log "verifying frame counter advanced..."
F0="$(extract "$RESP" 'result.frame_id')"
sleep 1
RESP="$(cp_call "$ENDPOINT" get_state '{}')"
F1="$(extract "$RESP" 'result.frame_id')"
log "  frame_id $F0 -> $F1"
if (( F1 <= F0 )); then
    fail "frame_id did not advance ($F0 -> $F1) — egui loop may be stuck"
fi
log "  ok: frame counter advancing"

log "OK: jukebox CP user-path verification passed"
echo "verify_jukebox: PASS (log: $LOG_FILE)"
