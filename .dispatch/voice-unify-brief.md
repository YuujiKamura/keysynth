# Brief: keysynth voice 仕組みの一本化 — Tier 1 piano voice を CP slot 化

Reference: 朝の dispatch chain (#40 hot-reload + #41 cp-core + #42 MIDI-optional/E2E) の自然な延長。

## Working tree
- Worktree: `<tmp>/ks-voice-unify` (e.g. `%TEMP%\ks-voice-unify` on Windows, `/tmp/ks-voice-unify` on Unix)
- Branch: `claude/voice-plugin-migration` (already checked out, base origin/main = e966045)
- DO NOT touch any file outside this worktree.

## 背景 — 現状はクソ設計

User が正しく指摘した問題: keysynth は「使える音源」を表現する仕組みが **2 個並列に存在する**。

1. `src/synth.rs` の `Engine` enum (Square / Ks / Piano / PianoModal / PianoThick / PianoLite / Piano5AM / 等 13 variant) が **本体にハードコード**。新音源追加 = enum + make_voice + 全体再コンパイル
2. PR #40 で導入した `voices_live/` cdylib **プラグイン仕組み**。`Engine::Live` 1 variant のみ。`.dll` 後乗せできるが今 toy sine しか入ってない

つまり昨夜 Tier 1 で作った Stulov / Fletcher / 32-partial soundboard は (1) 側に焼き込まれてて、(2) の hot-reload 経路から触れない。「外から GUI 再起動なしで Tier 1 音色を swap」が物理的にできない。

## Goal

**Tier 1 piano voice を全部 voices_live プラグインに移植して、CP slot として swap 可能にする。**

これで user 体験が:
```
keysynth --engine live --cp &
ksctl build --slot stulov   --src voices_live/piano_stulov
ksctl build --slot modal    --src voices_live/piano_modal
ksctl build --slot longitud --src voices_live/piano_longitud
ksctl set stulov   ;   ksctl render --notes 60,64,67,72 --out a.wav
ksctl set modal    ;   ksctl render --notes 60,64,67,72 --out b.wav
ksctl set longitud ;   ksctl render --notes 60,64,67,72 --out c.wav
# GUI 一度も再起動してない、3 種類の Tier 1 音色を聴き比べた
```

## Scope (今回の PR)

**移植対象 (Tier 1 全部):**
- `Engine::Piano` (T1.3 Fletcher + 既存 KS+modal hybrid) → `voices_live/piano/`
- `Engine::PianoModal` (T1.4 32-partial LUT) → `voices_live/piano_modal/`
- `Engine::PianoThick` / `PianoLite` / `Piano5AM` → 各 `voices_live/piano_thick/` etc.

**移植しない (legacy、out of scope):**
- `Engine::Square / Ks / KsRich / Sub / Fm / Koto / SfPiano / SfzPiano` — 後の PR で同パターンで移植可能だが今回は対象外。enum に残置。

**「Tier 1 全部 plugin 化」が完了の定義。** 他 Engine variant は enum に残ったまま OK、 PR #40 の cdylib mechanism と enum dispatch が共存する形でも OK。「Tier 1 音色は plugin 経由で外から swap できる」が達成されれば本ブリーフ完了。

## Implementation strategy

### Phase 1: ABI 拡張
PR #40 の `voices_live` cdylib ABI は voice instance 1 個を返すだけの最小設計。Tier 1 voice は per-note state (KsString delay buffer / modal resonator bank / hammer params / Fletcher allpass coeff / soundboard mode bank) を持つので ABI を:
- `keysynth_live_new(sr: f32, freq: f32, vel: u8) -> *mut Voice` (既存)
- `keysynth_live_render_add(*mut Voice, *mut f32, n: usize)` (既存)
- `keysynth_live_trigger_release(*mut Voice)` (既存)
- `keysynth_live_is_done / is_releasing / drop` (既存)
- `keysynth_live_abi_version() -> u32` (既存)

これでも Tier 1 voice は実装できるはず (voice instance 内部に必要な state を抱える)。ABI 変更不要なら無し優先。

### Phase 2: Tier 1 移植
各 Tier 1 voice を `voices_live/<name>/` cdylib として再実装:
- `voices_live/piano/Cargo.toml` + `src/lib.rs` — `keysynth::voices::piano::PianoVoice` を内部で構築 + ABI export
- `voices_live/piano_modal/...` — `PianoModalVoice` 同様
- 等

ここで重要: **音色の実装ロジックは keysynth crate (lib) 側に残す**。voices_live/<name>/ は薄い wrapper。`use keysynth::voices::piano::PianoVoice; use keysynth::voices::hammer_stulov::*;` で借りてくる。Tier 1 ファイル (hammer_stulov.rs / string_inharmonicity.rs / longitudinal.rs) を一切編集しない (load-bearing)。

### Phase 3: CP デモ
`tests/voice_unify_e2e.sh` (新規):
- keysynth --engine live --cp 起動
- 3 個 plugin build & load (piano, piano_modal, piano_longitud)
- 3 回 set & render
- 3 WAV の sha256 互いに異なることを assert
- "VOICE UNIFY E2E DEMO: PASS" line 出力

## Hard rules

- `cargo fmt --check` clean
- `cargo check --bin keysynth --bin ksctl` clean
- `cargo check --no-default-features --features web --target wasm32-unknown-unknown --bin keysynth-web` clean (voices_live は workspace 外なので wasm32 影響しない)
- **PR open 前に tests/voice_unify_e2e.sh を実機で実行して PASS line を出す**。captured log を bench-out/voice-unify/run.log にコミット
- DO NOT modify Tier 1 source files: `src/voices/hammer_stulov.rs`, `src/voices/string_inharmonicity.rs`, `src/voices/longitudinal.rs`, `bench-out/REF/sfz_salamander_multi/modal_lut.json`. ロジックは `pub use` で plugin から借りる
- DO NOT remove existing `Engine::Piano / PianoModal / PianoThick / PianoLite / Piano5AM` enum variants (他コードが参照してる可能性、互換維持)。plugin 経路を **追加** で OK
- DO NOT modify `cp-core/`, `live_reload.rs`, `src/cp.rs` (#40/#41 substrate、stable)
- DO NOT touch `audio_kira.rs` (web backend)
- DO NOT add new public API to `Engine` enum
- DO NOT depend on jsonrpsee / rmcp / mcp-* / 他人の RPC SDK

## Out of scope

- Square / Ks / KsRich / Sub / Fm / Koto / SfPiano / SfzPiano の plugin 化 (後回し)
- T1.2 longitudinal の物理 redesign (Issue #39、別 PR territory)
- T2.1 mistuning beat (33020 が並行で進行中)
- Engine enum 完全廃止 (将来的にあり得るが、今 PR ではしない)

## Commit + PR

- Commits 自由に分割 OK。最低でも:
  - `feat(voices_live): plugin scaffolding for Tier 1 piano variants`
  - `feat(voices_live/piano): Stulov + Fletcher + KS+modal hybrid as cdylib`
  - `feat(voices_live/piano_modal): 32-partial LUT projection as cdylib`
  - `feat(voices_live/...): その他 Tier 1 variants`
  - `test(cp): E2E demo swapping all Tier 1 plugins without restart`
- PR title: `feat(voices_live): unify Tier 1 piano voices as CP-swappable plugins (#41 follow-up)`
- PR description must include:
  - 設計説明: なぜ enum + plugin の二重構造を一旦受け入れて plugin 側に Tier 1 を集約するか
  - tests/voice_unify_e2e.sh の captured log inline
  - 3+ WAV の sha256 hash + size
  - reload latency (ksctl set X → first sample of X) at least 1 measurement
  - shepherd v2.2 が auto-merge する想定なので CI green が land 条件

## Why now

User の元々の要望「外側から GUI 再起動せずに音源を追加・差替え」の本筋は **Tier 1 音色が swap 可能になって初めて達成** される。今の demo は toy sine の gain 変えただけで、user の期待してた体験 (Stulov vs Fletcher vs Modal の聴き比べ) になってない。これを直す PR。

shepherd v2.2 (`baoavo3md`) が CI green で auto-merge する。E2E demo PASS line 出るまで PR 開かない。

Good luck.
