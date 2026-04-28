#!/usr/bin/env bash
# E2E Gate for Steel Phase 1
set -e

# Detect binary path
KSREPL="F:/rust-targets/release/ksrepl.exe"
if [ ! -f "$KSREPL" ]; then
    KSREPL="target/release/ksrepl.exe"
fi
if [ ! -f "$KSREPL" ]; then
    KSREPL="target/release/ksrepl"
fi

echo "Gate 1: (+ 1 2) -> 3"
OUT1=$(echo "(+ 1 2)" | "$KSREPL")
echo "Output: $OUT1"
if [ "$OUT1" != "3" ]; then
    echo "Gate 1 failed"
    exit 1
fi

echo "Gate 2: (define x 10) (* x x) -> 100"
OUT2=$(echo "(define x 10) (* x x)" | "$KSREPL")
echo "Output: $OUT2"
if [ "$OUT2" != "100" ]; then
    echo "Gate 2 failed"
    exit 1
fi

echo "Gate 3: Invalid S-expression '(' -> Error (non-zero exit)"
set +e
echo "(" | "$KSREPL" 2>err.log
EXIT_CODE=$?
set -e
if [ $EXIT_CODE -eq 0 ]; then
    echo "Gate 3 failed: expected non-zero exit code"
    exit 1
fi
echo "Gate 3 passed: non-zero exit code $EXIT_CODE"
cat err.log
if ! grep -q "Parse" err.log; then
    echo "Gate 3 failed: expected Parse error in stderr"
    exit 1
fi

echo "Gate 4: keysynth regression check"
cargo check --release --bin keysynth

echo "ALL GATES PASSED"
