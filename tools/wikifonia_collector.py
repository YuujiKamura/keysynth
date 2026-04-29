#!/usr/bin/env python3
"""Wikifonia source-collector — STAGE_C_PLACEHOLDER.

Reserved file slot for a future stage that mines lead-sheet MusicXML
from Wikifonia archives, converts to Standard MIDI, and folds the
result into ``bench-out/songs/manifest.json``. The Wikifonia archive
itself was withdrawn upstream; mirrors carry mixed licensing, so the
real implementation will need a per-piece license fetcher feeding
``tools/collector_validate.py`` before any entries are committed.

The ``STAGE_C_PLACEHOLDER`` marker in this docstring tells
``scripts/run-collector.sh`` to skip the file in its Stage B sweep
until a real implementation lands.

Tracking: out of scope for the current Stage C hardening branch.
"""

import sys


def main() -> int:
    print("wikifonia_collector: placeholder — not yet implemented", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
