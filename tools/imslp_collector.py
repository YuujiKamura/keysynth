#!/usr/bin/env python3
"""IMSLP source-collector — STAGE_C_PLACEHOLDER.

Reserved file slot for a future stage that mines public-domain MIDI
renderings from IMSLP (https://imslp.org/) into
``bench-out/songs/manifest.json``. IMSLP redistribution rules are
per-edition (Petrucci-style scans are PD; performer-uploaded MIDI
sometimes carries CC-BY-NC) so the collector here will need a
case-by-case license-page parser before it ships.

The ``STAGE_C_PLACEHOLDER`` marker in this docstring tells
``scripts/run-collector.sh`` to skip the file in its Stage B sweep
until a real implementation lands.

Tracking: out of scope for the current Stage C hardening branch.
"""

import sys


def main() -> int:
    print("imslp_collector: placeholder — not yet implemented", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
