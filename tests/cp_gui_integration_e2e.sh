#!/usr/bin/env bash
#
# CP <-> GUI voice browser bridge E2E.
#
# Proves that a CP-managed slot added via ksctl is reachable through
# the same audio path the GUI's voice-browser click would use:
#
#   1. Spawn `keysynth --engine live --cp` (the GUI process; a window
#      opens but we never click anything — the app's update() loop
#      runs `library.refresh_cp_slots(&reloader.list_slots())` every
#      frame, which is the in-process state the tests/cp_gui_voice_lib
#      Rust test asserts on).
#   2. ksctl build --slot piano --src voices_live/piano    → loads
#      the cdylib into the Reloader pool.
#   3. ksctl list                                          → ksctl
#      surfaces the same `Reloader::list_slots()` the GUI's library
#      refresh reads from.
#   4. ksctl set piano + render                            → audio
#      path: render → make_voice(Engine::Live) → reloader.make_voice
#      → active slot (piano cdylib) → Tier 1 PianoVoice. This is the
#      exact path the GUI's apply_slot for a Category::CpSlot triggers
#      (engine = Live + reloader.set_active(name)).
#   5. ksctl set default + render                          → swap to
#      the watcher's toy-sine slot, render again, expect different
#      sha256.
#   6. ksctl unload piano                                  → CP slot
#      gone; ksctl list confirms. The GUI's next frame would also
#      drop the browser entry via refresh_cp_slots.
#
# PASS gate is the literal "CP-GUI BRIDGE PASS" line at the bottom.
#
# Run from worktree root:
#   bash tests/cp_gui_integration_e2e.sh

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

OUT_DIR="bench-out/cp-gui"
mkdir -p "$OUT_DIR"
RUN_LOG="$OUT_DIR/run.log"
WAV_PIANO="$OUT_DIR/cpgui_piano.wav"
WAV_DEFAULT="$OUT_DIR/cpgui_default.wav"
KS_LOG="$OUT_DIR/keysynth.stderr.log"
PORT=5597  # distinct from cp_e2e (5577) and voice_unify (5587)
ENDPOINT="127.0.0.1:$PORT"

rm -f "$WAV_PIANO" "$WAV_DEFAULT" "$KS_LOG" "$RUN_LOG"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "=== cp_gui_integration_e2e.sh starting at $(date -u +%FT%TZ) ==="

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

if [ ! -f "$KEYSYNTH_BIN" ] || [ ! -f "$KSCTL_BIN" ]; then
    echo "FATAL: keysynth/ksctl binaries missing — run: cargo build --release --bin keysynth --bin ksctl"
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

echo
echo "=== spawn keysynth --engine live --cp (GUI process) ==="
KEYSYNTH_CP="$ENDPOINT" "$KEYSYNTH_BIN" --engine live --cp \
    --cp-endpoint "$ENDPOINT" >"$KS_LOG" 2>&1 &
KS_PID=$!
echo "keysynth pid=$KS_PID"

echo
echo "=== wait for CP server on $ENDPOINT ==="
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
    exit 1
fi
echo "CP server ready after ${WAITED}s"

# --- baseline list ----------------------------------------------------
echo
echo "=== baseline ksctl list ==="
"$KSCTL_BIN" --endpoint "$ENDPOINT" list

# --- step 1: ksctl build piano slot ----------------------------------
echo
echo "=== step 1: ksctl build piano slot ==="
"$KSCTL_BIN" --endpoint "$ENDPOINT" build --slot piano --src voices_live/piano \
    || { echo "FATAL: build piano failed"; exit 1; }

# --- step 2: assert piano shows in list ------------------------------
#
# The same `Reloader::list_slots()` that this RPC reads is what
# `KeysynthApp::update()` calls every frame to refresh the browser's
# Category::CpSlot section. So if `ksctl list` sees piano, the GUI
# library would too on its next repaint tick.
echo
echo "=== step 2: ksctl list after build ==="
LIST_OUT=$("$KSCTL_BIN" --endpoint "$ENDPOINT" list)
echo "$LIST_OUT"
if ! echo "$LIST_OUT" | grep -qE '(^|[[:space:]])piano([[:space:]]|$)'; then
    echo "FATAL: piano slot did not appear in ksctl list — CP-GUI bridge broken at the data source"
    exit 1
fi
echo "OK: piano slot visible to GUI library refresh"

# --- step 3: render through CP slot ----------------------------------
#
# `ksctl set piano` flips the Reloader's active-slot pointer — the
# exact same call the GUI's apply_slot for Category::CpSlot makes:
#
#     reloader.set_active(&slot.label)
#
# So a render right after `set` exercises the CP-slot audio path the
# GUI click would have triggered.
echo
echo "=== step 3: ksctl set piano + render ==="
"$KSCTL_BIN" --endpoint "$ENDPOINT" set piano \
    || { echo "FATAL: set piano failed"; exit 1; }
"$KSCTL_BIN" --endpoint "$ENDPOINT" render \
    --notes 60,64,67,72 --duration 5.0 --out "$WAV_PIANO" \
    || { echo "FATAL: render piano failed"; exit 1; }

# --- step 4: swap to default + render --------------------------------
echo
echo "=== step 4: ksctl set default + render ==="
"$KSCTL_BIN" --endpoint "$ENDPOINT" set default \
    || { echo "FATAL: set default failed"; exit 1; }
"$KSCTL_BIN" --endpoint "$ENDPOINT" render \
    --notes 60,64,67,72 --duration 5.0 --out "$WAV_DEFAULT" \
    || { echo "FATAL: render default failed"; exit 1; }

# --- step 5: ksctl unload + assert it's gone -------------------------
echo
echo "=== step 5: ksctl unload piano ==="
"$KSCTL_BIN" --endpoint "$ENDPOINT" unload --slot piano \
    || { echo "FATAL: unload piano failed"; exit 1; }
LIST_AFTER_UNLOAD=$("$KSCTL_BIN" --endpoint "$ENDPOINT" list)
echo "$LIST_AFTER_UNLOAD"
if echo "$LIST_AFTER_UNLOAD" | grep -qE '(^|[[:space:]])piano([[:space:]]|$)'; then
    echo "FATAL: piano slot still visible after unload"
    exit 1
fi
echo "OK: piano slot gone after ksctl unload"

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

[ -s "$WAV_PIANO" ]   || { echo "FATAL: $WAV_PIANO missing/empty"; exit 1; }
[ -s "$WAV_DEFAULT" ] || { echo "FATAL: $WAV_DEFAULT missing/empty"; exit 1; }

SHA_PIANO=$(sha_for "$WAV_PIANO")
SHA_DEFAULT=$(sha_for "$WAV_DEFAULT")
SIZE_PIANO=$(wc -c < "$WAV_PIANO")
SIZE_DEFAULT=$(wc -c < "$WAV_DEFAULT")
echo "piano    sha256=$SHA_PIANO   size=$SIZE_PIANO   path=$WAV_PIANO"
echo "default  sha256=$SHA_DEFAULT size=$SIZE_DEFAULT path=$WAV_DEFAULT"

if [ "$SHA_PIANO" = "$SHA_DEFAULT" ]; then
    echo "FATAL: piano and default WAV sha256 are identical — slot swap had no audible effect"
    exit 1
fi

echo
echo "CP-GUI BRIDGE PASS — see $OUT_DIR/"
