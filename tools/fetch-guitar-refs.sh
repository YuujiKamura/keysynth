#!/usr/bin/env bash
# Download CC0 acoustic-steel-string-guitar reference WAVs used by the
# `tests/guitar_e2e.rs` numerical gates. Idempotent — skips files that
# already exist with non-zero size.
#
# Source: Discord SFZ GM Bank, patch 026-Acoustic Guitar (steel),
# Martin HD-28 Vintage Series samples by Jeff Learman.
# License: Creative Commons CC0 1.0 Universal (public domain).
# See bench-out/REF/guitar/LICENSE.txt for the verbatim CC0 attribution.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${REPO_ROOT}/bench-out/REF/guitar"
BASE="https://raw.githubusercontent.com/sfzinstruments/Discord-SFZ-GM-Bank/master/Discord%20GM/Melodic/026-Acoustic%20Guitar%20%28steel%29"

mkdir -p "${DEST}"

FILES=(
    "MartinGM2_040__E2_1.wav"
    "MartinGM2_046_Bb2_1.wav"
    "MartinGM2_052__E3_1.wav"
    "MartinGM2_055__G3_1.wav"
    "MartinGM2_058_Bb3_1.wav"
    "MartinGM2_061_Db4_1.wav"
    "MartinGM2_064__E4_1.wav"
)

for f in "${FILES[@]}"; do
    target="${DEST}/${f}"
    if [[ -s "${target}" ]]; then
        echo "skip  ${f}"
    else
        echo "fetch ${f}"
        curl -fsSL --retry 3 -o "${target}" "${BASE}/${f}"
    fi
done

echo "done. ${#FILES[@]} files in ${DEST}"
