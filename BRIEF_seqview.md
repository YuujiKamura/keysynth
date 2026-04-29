# Brief: Jukebox Sequencer View

## Worktree
- `C:\Users\yuuji\keysynth-team-seqview` (branch `team/sequencer-view`, off origin/main)
- 起動時 `cd C:\Users\yuuji\keysynth-team-seqview` してから始めろ

## Goal
現状の jukebox は MIDI を事前 render して WAV を再生する **静的アセット paradigm** に短絡してる。これを直すために **sequencer view** を新設、live 演奏 paradigm の窓を開ける。

## Design Base
**`src/bin/jukebox_lite_c.rs` (Codex 版) を base にコピー**して新 bin を作れ。Codex 版を選ぶ理由はユーザー判断:
> 「Codexが作ったGUI ぱっと見で別次元のセンスがある。配色とか見た目の構築に関してClaudeより数段マシ」

**配色/レイアウト感は Codex 版に従う**。Claude 風 (jukebox_lite.rs) の card grid を引きずるな。

## 新 bin
`src/bin/jukebox_lite_seq.rs` を作成。`Cargo.toml` の `[[bin]]` も追加。

## View Toggle
画面上部に 2 mode toggle:
- **List view**: 現状の jukebox_lite_c の dense table + detail panel をそのまま継承
- **Sequencer view**: BMS-style note grid (時間 × pitch)、live 演奏

切替は state の enum で持つ。

## Sequencer View 要件 (paradigm 違反禁止)

### 必須: Live playback path
- **`render_midi` subprocess を呼ぶな**。WAV pre-render path 禁止
- `voices_live` cdylib (workspace 内、#41/#46) を**直接 in-process で**呼ぶ
- MIDI イベントを timer で逐次発火、voices_live の `note_on`/`note_off` API を叩く
- 既存の `render_midi` binary 内部にすでに live sequencer ロジックがある (録音せずに再生する path)。それを参考にしろ

### Note grid 表示
- 縦軸 = pitch (MIDI note)、横軸 = time (bar/beat)
- 現在 playhead を縦線で示す、再生中に右に走らせる
- 音長は矩形

### 再生中に「触れる」要素 (これが live paradigm の本体)
- **Voice swap mid-playback**: voice ドロップダウン、変えた瞬間から音色変更 (voices_live の hot-swap)
- **Mute / Solo per track**: MIDI が複数 track 持つ場合、track ごとに M/S ボタン
- **Tempo control**: スライダーで再生中変更
- **Transport**: Play / Pause / Stop / Seek (note grid をクリックで seek)

### Anti-pattern
- ❌ subprocess で render_midi を呼んで WAV を焼いて cpal で再生
- ❌ `preview_cache` を経由
- ❌ 「mid-playback で voice 変えても次回再生から」の延期

## 参考 memory
- `~/.claude/projects/C--Users-yuuji/memory/feedback_static_asset_paradigm_bias.md` (今回の根拠)
- `~/.claude/projects/C--Users-yuuji/memory/feedback_subtractive_ui_no_taste.md` (UI 加算するな)
- `~/.claude/projects/C--Users-yuuji/memory/feedback_gui_has_correct_answer.md` (Fitts/Hick/Gestalt 原理)

## 完了条件
1. `cargo build --release --bin jukebox_lite_seq` 通る
2. 起動して List view ↔ Sequencer view 切替できる
3. Sequencer view で MIDI 選んで Play すると、note grid が動き、音が出る
4. 再生中に voice ドロップダウン変更 → **その瞬間から音色変わる** (hot-swap が効いてる証拠)
5. mute/solo が再生中に効く
6. tempo slider が再生中に効く
7. desk_capture もしくは winshot でスクショを `bench-out/gui-verify/jukebox_lite_seq_v1.png` に保存
8. commit、worktree 内のみ

## 制約
- push 禁止、PR 禁止、worktree 完結
- CLAUDE.md (~/CLAUDE.md と ~/.claude/CLAUDE.md) と memory/MEMORY.md に従え
- 行き詰まったら `BRIEF_seqview.md` を読み直せ
- voices_live API が分からなければ `Cargo.toml` の workspace で voices_live crate 探せ、見つからなければ `cargo metadata --format-version 1 | jq '.packages[] | select(.name|contains("voices"))'`
