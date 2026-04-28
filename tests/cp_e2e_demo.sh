#!/usr/bin/env bash
#
# CP E2E live demonstration.
#
# Spawns a real keysynth process, drives it via the ksctl client, builds
# two voice slots from the same voices_live crate (mutating one constant
# in between), renders both to WAV, and asserts the audio is observably
# different between slot A and slot B. PASS gate is the literal
# "CP E2E DEMO: PASS" line at the bottom — the parent CI/PR workflow
# greps for it.
#
# Hardcoded MIDI ports are NOT required: keysynth tolerates an empty
# port list (post #41 follow-up) so this works on CI / containers /
# desktops without a keyboard plugged in.
#
# Run from the worktree root:
#   bash tests/cp_e2e_demo.sh

set -u
# Don't `set -e` globally — we want to detect failure ourselves and
# always fall through to the cleanup trap so a stray keysynth process
# never lingers. Exit codes are checked manually with `|| { ... }`.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

OUT_DIR="bench-out/cp-e2e"
mkdir -p "$OUT_DIR"
WAV_A="$OUT_DIR/cp_demo_a.wav"
WAV_B="$OUT_DIR/cp_demo_b.wav"
KS_LOG="$OUT_DIR/keysynth.stderr.log"
PORT=5577
ENDPOINT="127.0.0.1:$PORT"
LIVE_SRC="voices_live/src/lib.rs"
LIVE_SRC_BACKUP="$OUT_DIR/lib.rs.orig"

rm -f "$WAV_A" "$WAV_B" "$KS_LOG"

# Locate the built keysynth + ksctl binaries via cargo metadata so we
# tolerate a custom target-dir (e.g. the F:\rust-targets override on
# this dev machine).
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

if [ ! -x "$KEYSYNTH_BIN" ] && [ ! -f "$KEYSYNTH_BIN" ]; then
    echo "FATAL: keysynth binary not found at $KEYSYNTH_BIN"
    echo "       run: cargo build --release --bin keysynth --bin ksctl"
    exit 1
fi
if [ ! -x "$KSCTL_BIN" ] && [ ! -f "$KSCTL_BIN" ]; then
    echo "FATAL: ksctl binary not found at $KSCTL_BIN"
    exit 1
fi

KS_PID=""

# Restore voices_live source + kill keysynth on any exit path so a
# panic during the test never leaves the user's tree dirty or the
# audio device locked by an orphan process.
cleanup() {
    local rc=$?
    if [ -f "$LIVE_SRC_BACKUP" ]; then
        cp "$LIVE_SRC_BACKUP" "$LIVE_SRC"
        rm -f "$LIVE_SRC_BACKUP"
    fi
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
echo "spawning keysynth --engine live --cp ..."
KEYSYNTH_CP="$ENDPOINT" "$KEYSYNTH_BIN" --engine live --cp \
    --cp-endpoint "$ENDPOINT" >"$KS_LOG" 2>&1 &
KS_PID=$!
echo "keysynth pid=$KS_PID"

# --- wait for CP server -----------------------------------------------
echo "waiting for CP server on $ENDPOINT ..."
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

# --- slot A: build + render ------------------------------------------
echo
echo "=== slot A: build + render ==="
"$KSCTL_BIN" --endpoint "$ENDPOINT" build --slot A --src voices_live \
    || { echo "FATAL: build A failed"; exit 1; }

# Reload latency = time between issuing `set A` and the first
# successful render coming back. cp-core's `set` is a constant-time
# active-slot swap, so this also captures the call+JSON-RPC overhead
# end-to-end.
SET_START_NS=$(date +%s%N)
"$KSCTL_BIN" --endpoint "$ENDPOINT" set A \
    || { echo "FATAL: set A failed"; exit 1; }
"$KSCTL_BIN" --endpoint "$ENDPOINT" render \
    --notes 60,64,67,72 --duration 6.0 --out "$WAV_A" \
    || { echo "FATAL: render A failed"; exit 1; }
RENDER_END_NS=$(date +%s%N)
RELOAD_LATENCY_MS=$(( (RENDER_END_NS - SET_START_NS) / 1000000 ))
echo "set A -> first sample of A: ${RELOAD_LATENCY_MS} ms"

# --- mutate voices_live for slot B -----------------------------------
echo
echo "=== mutating voices_live (gain 0.6 -> 0.3) for slot B ==="
cp "$LIVE_SRC" "$LIVE_SRC_BACKUP"
# Halve the gain. Single, targeted, reversible mutation. The whole
# point is that B *sounds* different from A, and a 6 dB gain swing
# is well outside any FNV-1a fingerprint collision risk.
sed -i 's/\* 0\.6;/* 0.3;/' "$LIVE_SRC"
if ! grep -q '\* 0\.3;' "$LIVE_SRC"; then
    echo "FATAL: voices_live mutation did not apply (gain pattern not matched)"
    diff "$LIVE_SRC_BACKUP" "$LIVE_SRC" || true
    exit 1
fi

# --- slot B: build + render ------------------------------------------
echo
echo "=== slot B: build + render ==="
"$KSCTL_BIN" --endpoint "$ENDPOINT" build --slot B --src voices_live \
    || { echo "FATAL: build B failed"; exit 1; }
"$KSCTL_BIN" --endpoint "$ENDPOINT" set B \
    || { echo "FATAL: set B failed"; exit 1; }
"$KSCTL_BIN" --endpoint "$ENDPOINT" render \
    --notes 60,64,67,72 --duration 6.0 --out "$WAV_B" \
    || { echo "FATAL: render B failed"; exit 1; }

# Restore source NOW so even if the assertion phase fails the tree
# is clean — cleanup trap also does this, but doing it eagerly keeps
# the diff out of "git status" the moment the test passes.
cp "$LIVE_SRC_BACKUP" "$LIVE_SRC"
rm -f "$LIVE_SRC_BACKUP"

"$KSCTL_BIN" --endpoint "$ENDPOINT" list

# --- assertions -------------------------------------------------------
echo
echo "=== assertions ==="
[ -s "$WAV_A" ] || { echo "FATAL: $WAV_A missing/empty"; exit 1; }
[ -s "$WAV_B" ] || { echo "FATAL: $WAV_B missing/empty"; exit 1; }

if command -v sha256sum >/dev/null 2>&1; then
    SHA_A=$(sha256sum "$WAV_A" | awk '{print $1}')
    SHA_B=$(sha256sum "$WAV_B" | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
    SHA_A=$(shasum -a 256 "$WAV_A" | awk '{print $1}')
    SHA_B=$(shasum -a 256 "$WAV_B" | awk '{print $1}')
else
    echo "WARN: no sha256 binary, skipping hash check"
    SHA_A="(no sha256)"
    SHA_B="(no sha256)"
fi
echo "sha256 A = $SHA_A  ($WAV_A)"
echo "sha256 B = $SHA_B  ($WAV_B)"
if [ "$SHA_A" != "(no sha256)" ] && [ "$SHA_A" = "$SHA_B" ]; then
    echo "FATAL: WAV A and WAV B sha256 are identical — slot swap had no audible effect"
    exit 1
fi

SIZE_A=$(wc -c < "$WAV_A")
SIZE_B=$(wc -c < "$WAV_B")
echo "size   A = $SIZE_A bytes"
echo "size   B = $SIZE_B bytes"

echo
echo "reload latency (set A -> first sample of A): ${RELOAD_LATENCY_MS} ms"
echo "CP E2E DEMO: PASS — see $WAV_A and $WAV_B"
