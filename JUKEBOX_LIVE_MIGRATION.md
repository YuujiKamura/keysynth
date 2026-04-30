# Jukebox Live Migration — 事前レンダリング廃止 + 実時間 per-note 演奏

## 動機

現状の `jukebox_lite_seq` の MIDI 再生路:
```
MIDI file → render_midi (subprocess) → WAV file → cpal で WAV 再生
                                          ↓ cache miss なら待機
```

`chord_pad` は既に**実時間 per-note** で動いてる (PC キーボード / USB MIDI → `VoiceImpl::render_add` → cpal callback)。同じ `VoiceImpl` を持ってるのに片方は cache-and-replay、片方は live。理由が無い。

事前レンダリング路を凍結して、シーケンサーから直接 voice pool に dispatch する live 路に切り替える。

## 設計目標

1. **事前レンダリング路凍結**: `jukebox_lite_seq` の MIDI 再生から `render_midi` subprocess 呼出と `bench-out/preview/*.wav` cache lookup を排除。`render_midi` バイナリ自体は残置 (export 用途)。
2. **実時間 per-note**: cpal output stream callback で `MidiSequencer` の event を pop → `VoicePool` に dispatch → `VoiceImpl::render_add` でブロック合成。
3. **音色 live flip**: 再生中に channel ごと voice factory を入れ替えて、次の note_on から新 voice で鳴らす。
4. **per-channel routing**: MIDI 16 channel それぞれに独立の voice factory を割り当て可能。GM の典型 (ch10=drums) も意識。
5. **WAV/MP3 再生路は維持**: jukebox_lite_seq の既存 WAV/MP3 トラック再生は別経路 (`symphonia` decoder) なのでそのまま残す。今回触るのは MIDI 再生路のみ。

## アーキテクチャ

```
JukeboxLiteSeq (UI)
   ↓ holds
LiveMidiPlayer
   ├── MidiSequencer    (sample-accurate event source)
   ├── VoicePool        (polyphonic synth pool, per-channel routing)
   └── cpal::Stream     (output stream, callback drives sched + pool)
```

### `MidiSequencer` (`src/midi_sched.rs`, ~250 行)

```rust
pub struct MidiEvent { pub channel: u8, pub kind: MidiEventKind }
pub enum MidiEventKind {
    NoteOn  { note: u8, velocity: u8 },
    NoteOff { note: u8 },
    AllNotesOff,
}

pub struct MidiSequencer {
    events: Vec<(u64, MidiEvent)>, // (sample_offset, ev), sorted ascending
    cursor: AtomicU64,             // 現在の sample 位置
    total_samples: u64,
    sample_rate: u32,
    playing: AtomicBool,
    next_event_idx: AtomicUsize,   // events[i] 以降が未消化
}

impl MidiSequencer {
    pub fn from_smf(path: &Path, sample_rate: u32) -> Result<Self, String>;
    pub fn play(&self);
    pub fn pause(&self);
    pub fn stop(&self);                  // cursor=0 + all_notes_off 信号も
    pub fn seek(&self, samples: u64);
    pub fn cursor(&self) -> u64;
    pub fn total_samples(&self) -> u64;
    pub fn is_playing(&self) -> bool;

    /// cpal callback から呼ぶ: 現在 cursor から num_samples ぶん進めて、
    /// その期間に発火する events を順序通り返す。
    /// 戻り値の Vec はブロック内サンプル offset 付き (ローカル offset)。
    pub fn advance(&self, num_samples: u32) -> Vec<(u32, MidiEvent)>;
}
```

`midly` crate (deps 済) で SMF parse、tempo events を畳んで `(absolute_microseconds, event)` → `(absolute_samples, event)` に変換。

### `VoicePool` (`src/voice_pool.rs`, ~300 行)

```rust
pub trait VoiceFactory: Send + Sync {
    fn make_voice(&self, sample_rate: f32, freq: f32, velocity: f32) -> Box<dyn VoiceImpl + Send>;
    fn name(&self) -> &str;
    fn decay_model(&self) -> DecayModel;  // damper or plucked
}

struct ActiveVoice {
    channel: u8,
    note: u8,
    voice: Box<dyn VoiceImpl + Send>,
    started_at: u64,        // cursor sample at note_on (for stealing)
}

pub struct VoicePool {
    sample_rate: f32,
    voices: Mutex<Vec<ActiveVoice>>,         // 上限 32
    channel_factories: RwLock<[Arc<dyn VoiceFactory>; 16]>, // 16 ch 分
    voice_cap: usize,
}

impl VoicePool {
    pub fn new(sample_rate: f32, voice_cap: usize, default: Arc<dyn VoiceFactory>) -> Self;

    /// 音色 live flip: 既に鳴ってる voice は出し切る、次の note_on から新 voice。
    pub fn set_channel_factory(&self, channel: u8, factory: Arc<dyn VoiceFactory>);
    pub fn channel_factory_name(&self, channel: u8) -> String;

    pub fn dispatch(&self, ev: &MidiEvent);  // NoteOn/NoteOff/AllNotesOff
    pub fn process_block(&self, out: &mut [f32]);  // out[i] += sum(voices)
    pub fn active_count(&self) -> usize;
    pub fn all_notes_off(&self);
}
```

eviction policy (chord_pad と同じ): 上限到達時は releasing voice 優先で steal、無ければ FIFO。

`VoiceFactory` を `Arc<dyn VoiceFactory>` で持たせて、UI 層から `set_channel_factory(ch, new_arc)` で hot-swap。`channel_factories` は `RwLock` で audio callback は read のみ、UI 書込みは稀。

### `LiveMidiPlayer` (`src/bin/jukebox_lite_seq.rs` 内 or 新規 `live_midi_player.rs`)

```rust
pub struct LiveMidiPlayer {
    sched: Arc<MidiSequencer>,
    pool: Arc<VoicePool>,
    _stream: cpal::Stream,    // RAII; drop でストリーム停止
}

impl LiveMidiPlayer {
    pub fn new(smf: &Path, default_factory: Arc<dyn VoiceFactory>) -> Result<Self, String>;
    pub fn play(&self) { self.sched.play(); }
    pub fn pause(&self) { self.sched.pause(); }
    pub fn stop(&self) { self.sched.stop(); self.pool.all_notes_off(); }
    pub fn seek(&self, samples: u64) { self.sched.seek(samples); self.pool.all_notes_off(); }

    /// per-channel voice flip. 既存 voice は鳴り続け、次 note から新 voice。
    pub fn set_channel_voice(&self, ch: u8, factory: Arc<dyn VoiceFactory>) {
        self.pool.set_channel_factory(ch, factory);
    }
    pub fn channel_factory_name(&self, ch: u8) -> String { self.pool.channel_factory_name(ch) }
}

// cpal callback の中:
//   let events = sched.advance(num_samples);
//   for (offset, ev) in events {
//       // sample-accurate にしたいなら out を offset で分割して合成、
//       // 簡略化なら block 先頭で全 dispatch + 1 回 process_block。
//       pool.dispatch(&ev);
//   }
//   pool.process_block(out);
```

最初は block 先頭で全 dispatch + 1 回 `process_block` で良い (jitter は ≤ block_size = 数 ms、人間の耳で問題出ない)。sample-accurate 版は将来課題。

### UI 層 (`src/bin/jukebox_lite_seq.rs` 修正)

- 既存の MIDI 再生パス (cache lookup → render_midi spawn → WAV 待機) を削除。
- `LiveMidiPlayer` を保持し、MIDI track クリック時に new + play。
- 16 channel × voice ピッカー UI (drop down) を sidebar かパネル下部に追加。current voice 名表示 + クリックで voice 一覧から flip。
- voice 一覧は既存 `voices_live` discovery (`voice_lib::discover_plugin_voices`) から取得。

## 凍結対象 (UI のみから外す)

- `bench-out/preview/*.wav` の読み書き (jukebox_lite_seq の MIDI 再生路から)
- `render_midi` subprocess 呼出 (jukebox_lite_seq から)
- `preview_cache` モジュール参照 (jukebox_lite_seq から; モジュール本体と他 bin での使用は維持)

`render_midi` バイナリ・`preview_cache` モジュール・他 bin (`render_song`, `render_chord`, etc.) は**そのまま残す**。export / 解析用途で使うので削除しない。

## チーム実行プラン

### Phase 1: SCHED + POOL 並列 (2 agents)

| Agent | 担当ファイル | base branch |
|---|---|---|
| **SCHED** | `src/midi_sched.rs` (新規) + `src/lib.rs` (`pub mod midi_sched;`) | `team/jukebox-live-substrate` |
| **POOL** | `src/voice_pool.rs` (新規) + `src/lib.rs` (`pub mod voice_pool;`) | `team/jukebox-live-substrate` |

両者とも `src/lib.rs` を触るが追加行が違うのでマージは衝突しない (3-way merge で解決)。それでも衝突回避のため、hub が integrate 時に `src/lib.rs` を手動マージする。

完了条件 (各 agent):
1. `cargo build --release --features native` 成功
2. 新規 warning ゼロ
3. unit test 最低 3 件 (event ordering / voice allocation / voice stealing 等)
4. 変更が宣言した範囲のみ

### Phase 2: UI 統合 (1 agent)

| Agent | 担当ファイル | base branch |
|---|---|---|
| **UI** | `src/bin/jukebox_lite_seq.rs` (修正、既存 ~4271 行に手術) | Phase 1 統合済 substrate |

完了条件:
1. `cargo build --release --bin jukebox_lite_seq --features native` 成功
2. smoke test: 起動 5 秒以内 window visible (regression check)
3. 旧 cache path がコードから消えていること (`grep -nE 'preview_cache|spawn.*render_midi|cache_lookup' src/bin/jukebox_lite_seq.rs` の結果が再生関連で 0)
4. UI に per-channel voice picker (16ch ぶん) が存在
5. live flip API (`set_channel_voice`) が UI からトリガー可能

### Phase 3: 統合 + audible smoke test (Hub)

- 4 commits を `team/jukebox-live-substrate` に merge
- ビルド全部通す
- 起動 → MIDI クリック → 即時音 (cache wait なし) を**人間が**確認
  (audible は自動化困難。CP server の active voice count > 0 を起動後 2 秒以内に確認することで近似テスト)
- PR

### 規律

- worktree 隔離 (`keysynth-live-sched`, `keysynth-live-pool`, `keysynth-live-ui`)
- `CARGO_TARGET_DIR=<scratch-target>/jb-live-<wt>` (worktree-isolated build dir; pick any fast disk, e.g. `target/jb-live-<wt>`)
- foreground 強制、自己修正 3 回上限、`git add <path>` のみ
- 不変条件:
  - `VoicePool::process_block` は audio callback で呼ぶので **アロケート禁止** (Vec push 等は `dispatch` 側のみ、それも `Mutex::lock` の内側で限定)
  - `MidiSequencer::advance` も同様
  - `Mutex::try_lock` で contention 時は dispatch スキップ (callback を絶対に block しない)

## 参考

- `chord_pad` (`src/bin/chord_pad.rs` 84-180 行付近): polyphonic voice 管理 + steal eviction の参照実装
- `VoiceImpl` trait (`src/synth.rs:326`): `render_add(&mut [f32])`, `trigger_release()`, `is_done()`, `is_releasing()`
- `voice_lib::discover_plugin_voices`: 利用可能 voice 列挙
- `live_reload::LiveFactory`: voices_live cdylib hot-reload (UI 層で voice 一覧表示用)
- `midly` crate: SMF parse
