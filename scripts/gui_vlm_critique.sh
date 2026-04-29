#!/usr/bin/env bash
# gui_vlm_critique.sh — feed a screenshot to Gemini CLI and get a
# structured JSON critique back.
#
# Pairs with `cargo run --bin gui_audit`: gui_audit gives the
# deterministic pixel signal (contrast, density, peaks, blockers),
# this script gives the taste signal — would a human want to use this
# screen, what's confusing, what would they change. Both feed into
# the gui-verify-pixel-vlm skill ship gate.
#
# Usage:
#   scripts/gui_vlm_critique.sh --in shot.png [--out crit.json] [--model gemini-2.5-flash]
#
# Conventions:
# - Uses the `gemini` CLI in headless mode (`-p`) with @file inline
#   image attachment. No API key — Gemini CLI is OAuth (Code Assist).
# - Prompt is locked to a JSON-only response shape so the caller can
#   `jq` the result without LLM-prose pre-processing.
# - On parse failure we still print whatever Gemini returned, just on
#   stderr, and exit non-zero so a CI ship gate can catch it.
#
# Exit codes: 0 OK, 1 bad args / IO, 2 gemini call failed,
#             3 response was not valid JSON.

set -euo pipefail

IN=""
OUT=""
MODEL=""

usage() {
  cat <<'EOF'
gui_vlm_critique.sh --in <png> [--out <json>] [--model <name>]

  --in     PNG to critique (required)
  --out    Write JSON to this path (default: stdout)
  --model  Override Gemini model (default: CLI default)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in)    IN="$2"; shift 2 ;;
    --out)   OUT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "gui_vlm_critique: unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$IN" ]]; then
  echo "gui_vlm_critique: --in <png> is required" >&2
  exit 1
fi
if [[ ! -f "$IN" ]]; then
  echo "gui_vlm_critique: file not found: $IN" >&2
  exit 1
fi
if ! command -v gemini >/dev/null 2>&1; then
  echo "gui_vlm_critique: gemini CLI not on PATH" >&2
  exit 1
fi

# Convert backslashes (Windows paths) so the @file reference in the
# prompt body is shell-safe.
IN_REL="${IN//\\//}"

PROMPT=$(cat <<EOF
You are a UX critic. The image attached is a screenshot of a desktop GUI.
Return ONLY a single JSON object with EXACTLY these fields and no prose
outside the JSON. Do not wrap in markdown fences.

{
  "willing_to_use": "high" | "medium" | "low",
  "willing_to_use_reason": "<one sentence>",
  "first_impression": "<one sentence — what you notice in the first 2s>",
  "unclear": ["<thing that is hard to understand>", ...],
  "improvements": [
    "<concrete fix #1>",
    "<concrete fix #2>",
    "<concrete fix #3>"
  ],
  "blockers": ["<anything that would stop you shipping>", ...]
}

The "improvements" array MUST have exactly 3 entries. The "blockers"
array can be empty if nothing is genuinely ship-blocking. Be specific
— "increase contrast" is not allowed, "the timestamp text is too
light against the grey row background" is what we want.

Image: @${IN_REL}
EOF
)

GEMINI_ARGS=(-p "$PROMPT")
if [[ -n "$MODEL" ]]; then
  GEMINI_ARGS=(--model "$MODEL" "${GEMINI_ARGS[@]}")
fi

# Capture the raw response. We tee to a tmp file too so a debugger
# can see what came back even if jq rejects it.
TMP_RAW=$(mktemp -t gui_vlm.XXXXXX.txt)
trap 'rm -f "$TMP_RAW"' EXIT

if ! gemini "${GEMINI_ARGS[@]}" >"$TMP_RAW" 2>/dev/null; then
  echo "gui_vlm_critique: gemini call failed" >&2
  cat "$TMP_RAW" >&2
  exit 2
fi

# Strip optional ``` / ```json fences and take the first {...} block.
RAW=$(cat "$TMP_RAW")
CLEAN=$(printf '%s' "$RAW" \
  | sed -E 's/^[[:space:]]*```(json)?[[:space:]]*//' \
  | sed -E 's/[[:space:]]*```[[:space:]]*$//' \
  | awk 'BEGIN{depth=0; started=0}
         {
           for (i=1; i<=length($0); i++) {
             c = substr($0,i,1);
             if (c=="{") { if (!started) started=1; depth++; }
             if (started) printf "%s", c;
             if (c=="}") { depth--; if (started && depth==0) { printf "\n"; exit 0; } }
           }
           if (started) printf "\n";
         }')

if [[ -z "$CLEAN" ]]; then
  echo "gui_vlm_critique: gemini returned no JSON object" >&2
  echo "--- raw response ---" >&2
  echo "$RAW" >&2
  exit 3
fi

# Validate via python (jq is not guaranteed on Windows-bash).
if ! printf '%s' "$CLEAN" | python -c 'import json,sys; json.loads(sys.stdin.read())' 2>/dev/null; then
  echo "gui_vlm_critique: response did not parse as JSON" >&2
  echo "--- cleaned ---" >&2
  echo "$CLEAN" >&2
  exit 3
fi

if [[ -n "$OUT" ]]; then
  mkdir -p "$(dirname "$OUT")" 2>/dev/null || true
  printf '%s\n' "$CLEAN" >"$OUT"
else
  printf '%s\n' "$CLEAN"
fi
