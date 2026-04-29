# Jukebox Lite Rebuild — Architecture Spec

## 動機

現状の `jukebox_lite` / `jukebox_lite_c` / `jukebox_lite_g` 3 変種は、各 AI が独立に 1200〜2200 行の bin を書いた結果、egui の暗黙前提 (first frame で必ず paint 完了する・状態変化で repaint を要求する) を踏み外す壊れ方をしている。

具体例 (Gemini 版 `jukebox_lite_g`):
- ウィンドウ作成後 `WS_VISIBLE` が立たないまま `frame_id=1` で停止
- `request_repaint_after` が条件分岐 (再生中のみ) で first paint 失敗時に静止
- CP コマンドは push されるが `ctx.request_repaint()` を呼ばないため main thread が起きない

**結論**: 共通コア層が無く、各変種が壊れやすい同じ罠を独自コードで踏んでいる。コア層を抽出して UI 層を薄くする。

## 設計方針

### 層分離

```
┌──────────────────────────────────────────┐
│ UI Layer (per variant)                    │
│  - jukebox_lite     (Claude: card grid)   │
│  - jukebox_lite_c   (Codex: dense list)   │
│  - jukebox_lite_g   (Gemini: foobar2k)    │
│  各 ~400 行以内、render() のみ            │
└──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│ jukebox_core (新規 module: src/jukebox_core.rs) │
│  - JukeboxState        (single source of truth) │
│  - JukeboxLibrary      (track scan + library_db)│
│  - JukeboxAudio        (cpal mixer + decoder)   │
│  - JukeboxControl      (CP server + repaint)    │
│  - JukeboxApp trait    (UI hook)                │
│  - default update()    (repaint backstop)       │
└──────────────────────────────────────────┘
                    │
                    ▼
   keysynth lib (既存: library_db, play_log, gui_cp, synth, …)
```

### コア層の責務 (`src/jukebox_core.rs`)

```rust
pub struct JukeboxCore {
    pub library: JukeboxLibrary,       // tracks, song_index, voice_count
    pub audio: JukeboxAudio,            // mixer, sample_rate, pending_render
    pub selection: JukeboxSelection,    // tile, search, sort, selected_label
    pub history: JukeboxHistory,        // play_log, favorites, recent
    pub control: JukeboxControl,        // cp_state, cp_handle, repaint_signal
}

impl JukeboxCore {
    pub fn new(dirs: Vec<PathBuf>, app_name: &'static str) -> Result<Self, String>;
    pub fn tick(&mut self, ctx: &egui::Context);  // drain CP, poll mixer, sync state
    pub fn play(&mut self, label: &str);
    pub fn stop(&mut self);
    pub fn rescan(&mut self);
    pub fn visible_tracks(&self) -> Vec<&Track>;
}

pub trait JukeboxApp {
    fn render(&mut self, ctx: &egui::Context, core: &mut JukeboxCore);
}

// default eframe::App impl that all variants share
pub struct JukeboxRunner<A: JukeboxApp> { core: JukeboxCore, app: A }

impl<A: JukeboxApp> eframe::App for JukeboxRunner<A> {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.core.tick(ctx);
        self.app.render(ctx, &mut self.core);
        ctx.request_repaint_after(Duration::from_millis(500));  // backstop
    }
}
```

### バグを構造で殺す不変条件

1. **First-paint 必達**: `JukeboxRunner::update` 末尾に**無条件** `request_repaint_after(500ms)` → first paint が完了し eframe が `ShowWindow` を呼ぶ
2. **CP コマンドで repaint**: `JukeboxControl` 内の callback は `egui_ctx.request_repaint()` をクローン経由で呼ぶ → main thread を起こす
3. **エラー伝播**: `JukeboxCore::new` が `Result` で全失敗を返す。panic は禁止 (silent first-paint failure を防ぐ)
4. **state 単一所有**: 各 UI は `&mut JukeboxCore` を経由してのみ state を変更する。`Mutex` 重複ロックを禁止

### UI 層の責務 (per variant)

各 bin は 1 ファイル、200〜400 行目安。

```rust
struct JukeboxLiteG { /* UI-only state: scroll pos, hover etc */ }

impl JukeboxApp for JukeboxLiteG {
    fn render(&mut self, ctx: &egui::Context, core: &mut JukeboxCore) {
        // ここで egui::SidePanel/CentralPanel/TopBottomPanel を組む
        // core.library.tracks / core.audio.is_playing() / core.selection.tile を読む
        // クリック時は core.play(label) / core.stop() / core.selection.set_tile(...) を呼ぶ
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dirs = vec![PathBuf::from("bench-out/songs"), PathBuf::from("bench-out/CHIPTUNE")];
    let core = JukeboxCore::new(dirs, "jukebox_lite_g")?;
    let app = JukeboxLiteG::default();
    let runner = JukeboxRunner::new(core, app);
    eframe::run_native("JUKEBOX G",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1100.0, 700.0])
                .with_title("JUKEBOX G")
                .with_visible(true),  // 明示
            ..Default::default()
        },
        Box::new(|cc| {
            keysynth::ui::setup_japanese_fonts(&cc.egui_ctx);
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(runner))
        }),
    ).map_err(|e| format!("eframe: {e}").into())
}
```

## チーム実行プラン

### 入口・役割・キュー (5要素)

| 要素 | 内容 |
|---|---|
| **入口** | この `JUKEBOX_REBUILD.md` |
| **キュー** | Phase 1 (CORE) → Phase 2 (UI x3 並列) → Phase 3 (統合) |
| **役割** | CORE agent / UI-A agent / UI-C agent / UI-G agent / Hub (Claude) |
| **完了条件** | (1) `cargo build --release --bin <name>` 通過 (2) 起動後 5 秒以内に `IsWindowVisible` = true (3) frame_id が 5 秒で 10+ |
| **停止条件** | ビルド3回失敗 or smoke test失敗 → hub に報告して停止 (自己修正ループは禁止) |

### Phase 1: CORE 抽出 (1 agent, foreground)

- worktree base: `team/jukebox-rebuild`
- 触るファイル: `src/lib.rs` (mod 追加), `src/jukebox_core.rs` (新規)
- 既存 `src/bin/jukebox_lite*.rs` には触らない
- commit: `feat(jukebox_core): extract shared core (state/audio/cp/runner)`

### Phase 2: UI 書き直し (3 agents, parallel)

各エージェントは別 worktree。base は Phase 1 commit。

| Agent | 担当ファイル | 既存スタイル参照 |
|---|---|---|
| UI-A | `src/bin/jukebox_lite.rs` | YouTube-style card grid |
| UI-C | `src/bin/jukebox_lite_c.rs` | foobar2000-style dense list |
| UI-G | `src/bin/jukebox_lite_g.rs` | Spotify-style with sidebar |

各 agent commit: `refactor(jukebox_lite_X): rewrite on jukebox_core`

### Phase 3: 統合 + smoke test (Hub)

- 4 worktree の commit を `team/jukebox-rebuild` に順次 merge
- `cargo build --release --bin jukebox_lite --bin jukebox_lite_c --bin jukebox_lite_g`
- 3 バイナリそれぞれを別ターゲットに起動して `IsWindowVisible == true` を 5 秒以内に確認
- PR を `team/jukebox-rebuild` → `main` に作成

### 規律

- **同一ファイルを2 agent で並列改変禁止** (worktree-isolation で物理的に分ける)
- **`git add .` 禁止** (`git add <path>` のみ)
- **brief は1 step = 1 tool call**
- **build artifact は worktree 別 prefix** (`F:/rust-targets/<wt名>`) で衝突回避
- agent は完了条件を満たさない場合 hub に報告して停止 (自己修正は最大3回)
