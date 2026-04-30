#!/usr/bin/env bash
#
# Voice-unify E2E demo.
#
# Spawns a real `keysynth --engine live --cp` process, builds three
# Tier 1 piano plugins via `ksctl build` (without restarting the GUI),
# renders a chord through each, and asserts the three rendered WAVs
# have distinct sha256 — i.e. the slot swap audibly changes the timbre,
# which is the user-visible payoff of the "Tier 1 piano voices as
# CP-swappable plugins" PR.
#
# PASS gate is the literal "VOICE UNIFY E2E DEMO: PASS" line at the
# bottom — the parent CI / PR workflow greps for it.
#
# Run from the worktree root:
#   bash tests/voice_unify_e2e.sh
#
# Captured output for the PR description goes to
# bench-out/voice-unify/run.log (tee'd alongside stdout).

set -u
# No global `set -e`: cleanup trap must always run so a stray keysynth
# never lingers. Failures are detected explicitly with `|| { ... }`.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

OUT_DIR="bench-out/voice-unify"
mkdir -p "$OUT_DIR"
RUN_LOG="$OUT_DIR/run.log"
WAV_PIANO="$OUT_DIR/slot_piano.wav"
WAV_MODAL="$OUT_DIR/slot_piano_modal.wav"
WAV_THICK="$OUT_DIR/slot_piano_thick.wav"
WAV_LITE="$OUT_DIR/slot_piano_lite.wav"
WAV_5AM="$OUT_DIR/slot_piano_5am.wav"
KS_LOG="$OUT_DIR/keysynth.stderr.log"
PORT=5587  # distinct from cp_e2e_demo.sh's 5577 so the two can co-run
ENDPOINT="127.0.0.1:$PORT"

rm -f "$WAV_PIANO" "$WAV_MODAL" "$WAV_THICK" "$WAV_LITE" "$WAV_5AM" "$KS_LOG" "$RUN_LOG"

# Tee everything to the run log so the PR description can paste it
# verbatim. `exec` redirects all subsequent stdout/stderr.
exec > >(tee -a "$RUN_LOG") 2>&1

echo "=== voice_unify_e2e.sh starting at $(date -u +%FT%TZ) ==="
echo "ROOT  = $ROOT"
echo "OUT   = $OUT_DIR"
echo "PORT  = $PORT"

# Locate the built keysynth + ksctl binaries via cargo metadata so we
# tolerate a custom target-dir (e.g. set via `CARGO_TARGET_DIR` or a
# `.cargo/config.toml` override).
TARGET_DIR=$(cargo metadata --no-deps --format-version 1 2>/dev/null \
  | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p' | head -n1)
if [ -z "$TARGET_DIR" ]; then
    TARGET_DIR="$ROOT/target"
fi
case "$(uname -s 2>/dev/null)" in
    MINGW*|MSYS*|CYGWIN*) EXE_SUFFIX=".exe" ;;
    *) EXE_SUFFIX="" ;;
esac
KEYSYNTH_BIN="$TARGET_DIR/release/keysynth$EXE_SUFFIX"
KSCTL_BIN="$TARGET_DIR/release/ksctl$EXE_SUFFIX"

echo "TARGET_DIR    = $TARGET_DIR"
echo "KEYSYNTH_BIN  = $KEYSYNTH_BIN"
echo "KSCTL_BIN     = $KSCTL_BIN"

if [ ! -f "$KEYSYNTH_BIN" ]; then
    echo "FATAL: keysynth binary not found at $KEYSYNTH_BIN"
    echo "       run: cargo build --release --bin keysynth --bin ksctl"
    exit 1
fi
if [ ! -f "$KSCTL_BIN" ]; then
    echo "FATAL: ksctl binary not found at $KSCTL_BIN"
    exit 1
fi

KS_PID=""

cleanup() {
    local rc=$?
    if [ -n "$KS_PID" ]; then
        kill "$KS_PID" 2>/dev/null
        sleep 1
        kill -KILL "$KS_PID" 2>/dev/null
        if command -v taskkill >/dev/null 2>&1; then
            taskkill //F //PID "$KS_PID" >/dev/null 2>&1 || true
        fi
        wait "$KS_PID" 2>/dev/null || true
    fi
    return $rc
}
trap cleanup EXIT INT TERM

# --- spawn keysynth ---------------------------------------------------
echo
echo "=== spawning keysynth --engine live --cp ==="
KEYSYNTH_CP="$ENDPOINT" "$KEYSYNTH_BIN" --engine live --cp \
    --cp-endpoint "$ENDPOINT" >"$KS_LOG" 2>&1 &
KS_PID=$!
echo "keysynth pid=$KS_PID"

# --- wait for CP server -----------------------------------------------
echo
echo "=== waiting for CP server on $ENDPOINT ==="
WAITED=0
DEADLINE=30
READY=0
while [ "$WAITED" -lt "$DEADLINE" ]; do
    if "$KSCTL_BIN" --endpoint "$ENDPOINT" status >/dev/null 2>&1; then
        READY=1
        break
    fi
    if ! kill -0 "$KS_PID" 2>/dev/null; then
        echo "FATAL: keysynth died during startup; tail of $KS_LOG:"
        tail -n 40 "$KS_LOG" || true
        exit 1
    fi
    sleep 1
    WAITED=$((WAITED + 1))
done
if [ "$READY" -ne 1 ]; then
    echo "FATAL: CP server did not come up within ${DEADLINE}s"
    tail -n 40 "$KS_LOG" || true
    exit 1
fi
echo "CP server ready after ${WAITED}s"
"$KSCTL_BIN" --endpoint "$ENDPOINT" status

# --- build + render each Tier 1 plugin -------------------------------
#
# render_one <slot_name> <crate_dir> <wav_path>
#   1. ksctl build  --slot <slot> --src <crate_dir>
#   2. ksctl set    <slot>           (timestamped to compute reload latency)
#   3. ksctl render --notes 60,64,67,72 --duration 6 --out <wav>
render_one() {
    local slot="$1"
    local src="$2"
    local out="$3"
    echo
    echo "=== slot $slot: build + render ==="
    "$KSCTL_BIN" --endpoint "$ENDPOINT" build --slot "$slot" --src "$src" \
        || { echo "FATAL: build $slot from $src failed"; exit 1; }
    local set_start_ns
    set_start_ns=$(date +%s%N)
    "$KSCTL_BIN" --endpoint "$ENDPOINT" set "$slot" \
        || { echo "FATAL: set $slot failed"; exit 1; }
    "$KSCTL_BIN" --endpoint "$ENDPOINT" render \
        --notes 60,64,67,72 --duration 6.0 --out "$out" \
        || { echo "FATAL: render $slot failed"; exit 1; }
    local render_end_ns
    render_end_ns=$(date +%s%N)
    local latency_ms=$(( (render_end_ns - set_start_ns) / 1000000 ))
    echo "set $slot -> first sample of $slot: ${latency_ms} ms"
    # Stash latency into a file so the assertion phase can quote one.
    echo "$slot $latency_ms" >> "$OUT_DIR/reload_latency.tsv"
}

rm -f "$OUT_DIR/reload_latency.tsv"
render_one piano       voices_live/piano       "$WAV_PIANO"
render_one piano_modal voices_live/piano_modal "$WAV_MODAL"
render_one piano_thick voices_live/piano_thick "$WAV_THICK"
render_one piano_lite  voices_live/piano_lite  "$WAV_LITE"
render_one piano_5am   voices_live/piano_5am   "$WAV_5AM"

echo
"$KSCTL_BIN" --endpoint "$ENDPOINT" list

# --- assertions -------------------------------------------------------
echo
echo "=== assertions ==="
sha_for() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo "(no-sha256-binary)"
    fi
}

declare -a WAVS=("$WAV_PIANO" "$WAV_MODAL" "$WAV_THICK" "$WAV_LITE" "$WAV_5AM")
declare -a NAMES=("piano" "piano_modal" "piano_thick" "piano_lite" "piano_5am")
declare -a SHAS=()
declare -a SIZES=()
for i in "${!WAVS[@]}"; do
    wav="${WAVS[$i]}"
    name="${NAMES[$i]}"
    [ -s "$wav" ] || { echo "FATAL: $wav missing/empty"; exit 1; }
    sha=$(sha_for "$wav")
    size=$(wc -c < "$wav")
    SHAS+=("$sha")
    SIZES+=("$size")
    echo "$name  sha256=$sha  size=$size  path=$wav"
done

# Distinct-sha gate: at minimum the 3 plugins the brief calls out
# (piano / piano_modal / piano_thick) must produce 3 different WAVs.
# Empirically the other two (piano_lite, piano_5am) are also distinct;
# we surface but don't gate on those so a future preset alignment that
# happens to collide on lite vs piano doesn't fail the demo unless it
# also collides with one of the gating three.
GATE_INDICES=(0 1 2)
declare -A SEEN
for idx in "${GATE_INDICES[@]}"; do
    sha="${SHAS[$idx]}"
    name="${NAMES[$idx]}"
    if [ -n "${SEEN[$sha]:-}" ]; then
        echo "FATAL: gating slot $name has identical sha256 to ${SEEN[$sha]} — slot swap had no audible effect"
        exit 1
    fi
    SEEN[$sha]="$name"
done

echo
echo "reload latencies (set <slot> -> first sample of <slot>, ms):"
cat "$OUT_DIR/reload_latency.tsv"

echo
echo "VOICE UNIFY E2E DEMO: PASS — see $OUT_DIR/"
