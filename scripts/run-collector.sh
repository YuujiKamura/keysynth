#!/usr/bin/env bash
# Daily / on-demand wrapper around the song-catalog pipeline.
#
#   Stage A — environment / paths.
#   Stage B — `tools/mutopia_collector.py` (and future siblings: imslp,
#             wikifonia). Each collector mutates
#             bench-out/songs/manifest.json + drops .mid into
#             bench-out/songs/.
#   Stage C — `tools/collector_validate.py` postprocesses the manifest:
#             license whitelist, ID-collision detection, optional
#             playable-MIDI render check, failure-log dump.
#
# Designed to be called from cron / GitHub Actions / local dev:
#
#   scripts/run-collector.sh                 # collect + validate (no render)
#   scripts/run-collector.sh --render-check  # also smoke-render every entry
#   scripts/run-collector.sh --apply         # rewrite manifest in place
#
# Stage B is invoked only when its script exists. That keeps this
# wrapper usable on branches that ship Stage C ahead of Stage B (and
# vice versa) without spurious failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    PYTHON="python"
fi

echo "[run-collector] repo=${REPO_ROOT} python=$(${PYTHON} --version 2>&1)"

# ---- Stage B: source collectors --------------------------------------------
collectors=(
    "tools/mutopia_collector.py"
    # OOS for now — placeholder slots so future stages have a clear
    # landing pad. Each collector is invoked only when its file is
    # actually a real (non-placeholder) script.
    "tools/imslp_collector.py"
    "tools/wikifonia_collector.py"
)

for c in "${collectors[@]}"; do
    if [[ ! -f "${c}" ]]; then
        echo "[run-collector] skip ${c} (not present yet)"
        continue
    fi
    # Treat scripts whose first kilobyte advertises themselves as
    # placeholders ("Stage C placeholder") as no-ops; an empty grep is
    # treated as a real implementation.
    if head -c 1024 "${c}" | grep -q "STAGE_C_PLACEHOLDER"; then
        echo "[run-collector] skip ${c} (placeholder)"
        continue
    fi
    echo "[run-collector] >>> ${c}"
    "${PYTHON}" "${c}"
done

# ---- Stage C: validate -----------------------------------------------------
echo "[run-collector] >>> tools/collector_validate.py $*"
set +e
"${PYTHON}" tools/collector_validate.py "$@"
status=$?
set -e

if [[ ${status} -eq 0 ]]; then
    echo "[run-collector] catalog clean"
elif [[ ${status} -eq 1 ]]; then
    echo "[run-collector] catalog has rejections — see tools/.collector-failures.json"
else
    echo "[run-collector] validator error (exit ${status})"
fi

exit ${status}
