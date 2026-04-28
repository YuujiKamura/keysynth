# Guitar reference corpus

Two parallel layers, both restricted to permissive licences (CC0 / CC-BY /
CC-BY-SA / BSD / MIT / Public Domain — anything more restrictive is
rejected by `fetch.sh`'s manifest builder):

1. **Single-pitch CC0 sample set** (legacy, used by `tests/guitar_e2e.rs`)
   — the seven Discord SFZ GM Bank steel-string acoustic samples
   listed in [`LICENSE.txt`](./LICENSE.txt). Fetched by
   [`tools/fetch-guitar-refs.sh`](../../../tools/fetch-guitar-refs.sh).

2. **Permissive multi-source ground-truth corpus** (issue #44 follow-up)
   — single notes per (string, fret), strummed chord set, and a
   classical / popular corpus slice from public-domain and CC-BY
   sources. Fetched by [`fetch.sh`](./fetch.sh) in this directory.
   Validated by [`tests/guitar_corpus_e2e.sh`](../../../tests/guitar_corpus_e2e.sh).

This README documents layer (2). For (1) see the file-level table in
`LICENSE.txt`.

## Layer (2) — directory tree

```
bench-out/REF/guitar/
├── README.md                       (this file)
├── fetch.sh                        download + offline-test fixture builder
├── LICENSE.txt                     legacy CC0 layer (1) provenance
├── LICENSE-GuitarSet.txt           BSD-3-Clause, single_note + chord audio
├── LICENSE-Slakh2100.txt           CC BY 4.0, corpus/popular slice
├── LICENSE-Mutopia.txt             Public Domain, corpus/classical slice
├── single_note/
│   ├── manifest.json               (committed) 36 entries (6 strings × 6 frets)
│   └── E2_open.wav                 (committed) deterministic CC0 fixture
│   └── *.wav                       (fetched on demand from GuitarSet)
├── chord/
│   ├── manifest.json               (committed) 8 chord entries
│   ├── C_major_strum.mid           (committed) deterministic CC0 fixture
│   ├── *_strum.mid                 (fetched from GuitarSet annotations)
│   └── *_strum.wav                 (fetched on demand from GuitarSet)
└── corpus/
    ├── classical/
    │   ├── manifest.json           (committed) 10 Mutopia pieces
    │   ├── *.mid                   (fetched from Mutopia)
    │   └── *.wav                   (fetched / synthesised)
    └── popular/
        ├── manifest.json           (committed) 5 Slakh2100 guitar stems
        ├── *.mid                   (fetched from Slakh2100)
        └── *.wav                   (fetched on demand)
```

Everything in **(committed)** above travels with the repo. Everything
else lives behind `fetch.sh` and is gitignored. The single committed
WAV fixture (`single_note/E2_open.wav`, 176 444 bytes) exists so the
E2E gate can run without network access — see the offline-test mode
below.

Standard format for every audio file: **44.1 kHz · mono · 16-bit PCM**.
Standard format for every MIDI: **SMF format-1 · PPQ 480**.

## License summary

| Source        | License          | Coverage                         |
|---------------|------------------|----------------------------------|
| GuitarSet     | BSD-3-Clause     | `single_note/`, `chord/` audio   |
| Slakh2100     | CC BY 4.0        | `corpus/popular/` audio + MIDI   |
| Mutopia       | Public Domain    | `corpus/classical/` MIDI         |
| Discord GMBank| CC0-1.0          | layer (1) — see `LICENSE.txt`    |
| keysynth      | CC0-1.0          | committed offline-test fixtures  |

CC-BY-NC, proprietary, and unknown-licence sources are explicitly
rejected by the `assert_permissive_license` check inside
`fetch.sh`'s Python helper. See `LICENSE-*.txt` for verbatim license
text and per-source provenance.

## fetch.sh usage

```bash
# Build deterministic fixtures + every manifest.json without touching
# the network. Suitable for CI smoke-test and the E2E gate.
bash bench-out/REF/guitar/fetch.sh --offline-test

# Real download. Pulls GuitarSet (~5 GB), the Slakh2100 metadata
# index, and Mutopia per-piece MIDI. Idempotent — files already
# present with non-zero size are skipped.
bash bench-out/REF/guitar/fetch.sh

# Skip the optional Slakh2100 step (its full distribution is ~145 GB
# and the metadata index download is the only step run by default
# anyway, but this silences the notice).
bash bench-out/REF/guitar/fetch.sh --skip-slakh
```

`fetch.sh` requires:
- `bash`
- `python` or `python3` (3.8+; uses only `wave`, `struct`, `json`,
  `urllib.request` from the standard library)
- `curl`
- HTTPS connectivity to `zenodo.org`, `mutopiaproject.org`,
  `raw.githubusercontent.com` (real-mode only)

The Python helper enforces that every entry in every `manifest.json`
carries a license string from the permissive set. A non-permissive
entry causes `fetch.sh` to abort with a `non-permissive license
rejected` error.

## Wiring with the `score` binary

The standard layout is:

```bash
cargo run --release --bin score -- \
    --reference bench-out/REF/guitar/single_note/E2_open.wav \
    --candidate <your-candidate-render>.wav \
    --note 40
```

`--note 40` matches the E2 (open low E) MIDI number, which is the
pitch the committed fixture sounds at. For other reference targets,
look up the `midi` field in the per-directory `manifest.json` and
pass that value to `--note`.

The E2E gate `tests/guitar_corpus_e2e.sh` runs this command against
`E2_open.wav` for both `--reference` and `--candidate` (a
self-loss smoke test) so the binary's wiring against the corpus is
verified on every PR touching this tree.

## Parallel structure with the piano (Salamander) reference

| Family | License root          | Single-note fixture                | Wired into                                          |
|--------|-----------------------|-------------------------------------|------------------------------------------------------|
| Piano  | CC-BY 3.0 (Salamander)| `bench-out/REF/sfz_salamander_grand_v3_C4.wav` | `score --reference …Salamander…C4.wav --note 60` (default) |
| Guitar | BSD-3 + CC-BY + PD    | `bench-out/REF/guitar/single_note/E2_open.wav` | `score --reference …guitar/…E2_open.wav --note 40`         |

Each family carries (a) at least one committed reference target, (b)
a documented fetch path for the larger corpus, and (c) an E2E gate
that exercises the `score` binary against the family's reference.
The pattern this PR establishes — one `LICENSE-*.txt` per source,
one `manifest.json` per directory, `fetch.sh` with `--offline-test`
mode — should be repeated for any future non-piano family
(clarinet / drums / etc.) added to `voices_live/`.

## Provenance per source

### GuitarSet (BSD-3-Clause)

- Catalogue page: <https://guitarset.weebly.com/>
- Zenodo deposit: <https://zenodo.org/record/3371780>
- Repo:           <https://github.com/marl/GuitarSet>
- Acquisition date for this corpus: 2026-04-28

### Slakh2100 (CC BY 4.0)

- Catalogue page: <http://www.slakh.com/>
- Repo:           <https://github.com/ethman/slakh-utils>
- Acquisition date: 2026-04-28

### Mutopia Project (Public Domain)

- Catalogue page: <https://www.mutopiaproject.org/>
- About:          <https://www.mutopiaproject.org/about.html>
- Acquisition date: 2026-04-28

### keysynth (CC0-1.0, in-tree)

- Repo: <https://github.com/YuujiKamura/keysynth>
- The `single_note/E2_open.wav` and `chord/C_major_strum.mid`
  fixtures are deterministically synthesised by
  `bench-out/REF/guitar/fetch.sh --offline-test`. Re-running the
  script produces byte-identical output (no RNG, no time-of-day
  inputs).
