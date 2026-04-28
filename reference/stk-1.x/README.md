# Synthesis Toolkit (STK) — vendored Rust-port reference

This directory is a **read-only, vendored** subset of the Synthesis Toolkit
in C++ (STK). It is **not built and not linked** by this crate. We keep
these files in-tree purely as a reference while porting STK's acoustic
guitar physical model (`stk::Guitar` plus `stk::Twang` and supporting
filters/delays) to Rust under `voices::guitar_stk`. Do not modify the
files in `include/` or `src/` — they should remain byte-identical to
upstream so future contributors can diff against the original implementation.

## Upstream

- Repository: https://github.com/thestk/stk
- Branch: `master`
- Commit pinned: `6aacd357d76250bb7da2b1ddf675651828784bbc`
  (HEAD of `master` on 2025-03-29 — *Merge pull request #150
  from donarturo11/fix-msvc-compile-error*)

Files were fetched verbatim from
`https://raw.githubusercontent.com/thestk/stk/<sha>/<path>`.

## License

STK ships under an MIT-equivalent permissive license. Full text is in
`./LICENSE`. Header reproduced here for attribution:

> The Synthesis ToolKit in C++ (STK)
>
> Copyright (c) 1995-2023 Perry R. Cook and Gary P. Scavone
>
> Permission is hereby granted, free of charge, to any person obtaining
> a copy of this software and associated documentation files (the
> "Software"), to deal in the Software without restriction, including
> without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to
> the following conditions:
>
> The above copyright notice and this permission notice shall be
> included in all copies or substantial portions of the Software.
>
> Any person wishing to distribute modifications to the Software is
> asked to send the modifications to the original developer so that they
> can be incorporated into the canonical version. This is, however, not
> a binding provision of this license.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
> EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
> MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
> IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
> ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
> CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
> WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

When the Rust port lands, retain the copyright notice in any file
derived from these sources, alongside our own crate license.

## Files vendored

### Top-level
- `LICENSE`
- `README.md` (this file)

### `include/` — STK public headers
- `Guitar.h` — multi-string guitar instrument (port target)
- `Twang.h` — single-string Karplus-Strong waveguide used by `Guitar`
- `Plucked.h` — alternative single-string plucked-string model
- `Stk.h` — central STK base class (sample-rate registry, error handling)
- `Filter.h` — abstract filter base
- `Generator.h` — abstract generator base (parent of `Noise`)
- `Instrmnt.h` — abstract instrument base (parent of `Plucked`/`Guitar`)
- `Delay.h` `DelayA.h` `DelayL.h` — non-interpolating / allpass / linear delay lines
- `TapDelay.h` — multi-tap delay
- `OneZero.h` `OnePole.h` `BiQuad.h` `Fir.h` — basic IIR/FIR filters
- `Noise.h` — white-noise generator
- `SKINImsg.h` — STK control-message constants referenced by `Guitar.cpp`

### `src/` — corresponding C++ implementations
`Guitar.cpp`, `Twang.cpp`, `Plucked.cpp`, `Stk.cpp`,
`Delay.cpp`, `DelayA.cpp`, `DelayL.cpp`, `TapDelay.cpp`,
`OneZero.cpp`, `OnePole.cpp`, `BiQuad.cpp`, `Fir.cpp`, `Noise.cpp`.

## Headers intentionally NOT vendored

- `Network.h`, `RtAudio*`, `Mutex.h`, `Thread.h` — realtime I/O
  scaffolding; irrelevant to an offline / hosted-by-CPAL Rust port.
- `FileWvIn.h` (referenced from `Guitar.cpp` to load the `pluck1.raw`
  excitation sample). Pulling it in would drag the entire
  `WvIn` / `FileRead` / `Function` chain. The Rust port will load
  excitation samples through `hound` (or a baked-in static buffer),
  so the C++ file-I/O class is not needed as a porting reference.
  The `#include "FileWvIn.h"` line is left in `Guitar.cpp` as-is for
  fidelity to upstream and as a hint about where excitation samples
  enter the pipeline.

## Pointers

- Upstream survey of permissive guitar physical-model implementations:
  keysynth issue #52.
- Forthcoming Rust port: `voices::guitar_stk` (sibling task to this
  vendoring commit).
