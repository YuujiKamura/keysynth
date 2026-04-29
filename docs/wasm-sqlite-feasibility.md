# Web (wasm32) で SQLite — フィージビリティ調査

Issue #66 OOS の "web (wasm32) で SQLite (`rusqlite` の wasm 対応は別途検証)" を裏取り
した調査メモ。コード変更なし、推奨案 1 つを根拠つきで提示する。

調査日: 2026-04-29 / branch `team/wasm-sqlite` (off `origin/main` @ 47ced27)

## 前提 — keysynth 側で web に出したい SQLite ワークロード

`src/library_db/mod.rs` から見える事実:

1. `bench-out/library.db` は **reproducible build artifact** で commit される
   (= MIDI manifest と voice metadata から `LibraryDb::rebuild()` で材料化される
   read-only catalog)。
2. `bench-out/play_log.db` (plays table) は **machine-local / gitignore**。
   ユーザの再生履歴のみ。
3. native ビルドは `rusqlite = { version = "0.31", features = ["bundled"] }`
   で OS の libsqlite3 を踏まない (`Cargo.toml:131`)。

web に持っていきたいのは現実的には **(1) library.db の read-only クエリ** で十分。
`bench-out/songs/manifest.json` と同じ「公開成果物バイナリ」扱いで wasm bundle に
焼き込めば、jukebox のフィルタ・並び替え・タグ検索が web 版でも生きる。

(3) のように現状 `rusqlite` は `native` feature 配下でしか有効化されないので、
web で SQLite を使うには `[target.'cfg(target_arch = "wasm32")'.dependencies]`
にバックエンドを追加する必要がある。

## 候補 3 つの 2026 年 Q2 時点ステータス

### A. `rusqlite[bundled]` をそのまま wasm32-unknown-unknown 向けに

歴史的に通らない: stdio.h 不在・libc が無いためリンカが fail
([rusqlite#827][rusqlite-827], [rusqlite#603][rusqlite-603])。
trevyn による [PR #1010][pr-1010] は close、後続 PR (例 [#935][pr-935]) も
紆余曲折。**現時点で `rusqlite[bundled]` 単体での wasm32-unknown-unknown
ネイティブビルドは production-ready ではない**。

ただし重要な進展として、rusqlite 本体が **wasm32-unknown-unknown ターゲット時に
内部で `sqlite-wasm-rs` を libsqlite3-sys の代わりに使う**経路が入っている
([sqlite-wasm-rs README][swr-crates] / [docs.rs][swr-docs])。つまり
"rusqlite の API はそのまま、wasm 向けには裏で sqlite-wasm-rs に委譲" という
ハイブリッドが正規化されつつある。この経路は keysynth 側 (rusqlite 0.31)
では未対応バージョンの可能性が高く、**移行時は rusqlite の minor バージョン
アップが前提**になる。

### B. [`sqlite-wasm-rs`][swr-crates] を直接使う

* wasm32-unknown-unknown 向け libsqlite3 バインディング (Rust 製)。
* `precompiled` feature で **emscripten 不要** の wasm bundle が手に入る
  (= keysynth の既存 `trunk build` パイプラインに余計なツールチェインを
  足さなくていい)。
* VFS が複数選べる:
  * `memdb` (in-memory)
  * `opfs-sahpool` (OPFS sync access handle pool — sqlite-wasm を移植)
  * `relaxed-idb` (IndexedDB のブロック保存)
* MSRV 1.82、`SQLITE_THREADSAFE=0` でビルド (= シングルスレッド前提)。
* 2026-01 時点で actively maintained。
* ただし API は `sqlite_wasm_rs::*` で **rusqlite と互換ではない**。
  使うなら native と web で `cfg(target_arch)` でラップする adapter が要る。

### C. sqlite.org 公式 [`sqlite-wasm`][official-wasm] (JS bridge 経由)

* sqlite.org が出している ES Module 版 (npm `@sqlite.org/sqlite-wasm`)。
* OPFS 永続化サポート — ただし以下の制約:
  * **Worker context 限定** (メインスレッド不可)
  * サーバ側に `Cross-Origin-Opener-Policy: same-origin` と
    `Cross-Origin-Embedder-Policy: require-corp` の COOP/COEP ヘッダ必須
    ([persistence.md][official-persistence], [chrome.dev 記事][chrome-blog])。
* keysynth-web は GitHub Pages 配信 → COOP/COEP ヘッダを足すのが面倒
  (静的ホスティングだとカスタムヘッダ設定不可、Service Worker で偽装する
  workaround はあるが追加複雑性)。
* Rust → JS → wasm の二重バインディングになり、`rusqlite` API との乖離も
  最大。

## 推奨 — 案 B (`sqlite-wasm-rs[precompiled]`) を memdb VFS で

### 結論

**`sqlite-wasm-rs` を `precompiled` + `memdb` で組む**。OPFS 永続化は v1 では
やらない。

```toml
[target.'cfg(target_arch = "wasm32")'.dependencies]
sqlite-wasm-rs = { version = "*", default-features = false, features = ["precompiled"] }
```

native (`rusqlite`) と wasm (`sqlite-wasm-rs`) の API 差は `src/library_db/`
配下に薄い adapter trait を切って吸収する (`Connection::prepare` /
`Statement::query_map` 相当だけ抽象化すれば足りる)。

### 根拠

1. **読み取り専用 catalog なら永続化レイヤは要らない**。
   `bench-out/library.db` バイトを `include_bytes!` で wasm に焼き込み、
   起動時に memdb に deserialize して終わり。SF2 を `include_bytes!` で
   焼いてる既存パターン (`Cargo.toml:65-69`) と完全に同じ思想。
   永続化が要るのは play_log だけだが、これは Issue #66 のスコープ外。

2. **GitHub Pages 制約と整合する**。COOP/COEP ヘッダ不要 = OPFS 経路を踏まない
   ので、案 C の `Cross-Origin-Embedder-Policy` 問題が消える。`trunk build`
   で出る単一 wasm + 静的ファイルのままデプロイできる。

3. **emscripten 依存を持ち込まない** (`precompiled` feature の利点)。
   既存 `trunk` + `wasm32-unknown-unknown` ターゲットの CI に追加工程ゼロ。

4. **rusqlite との将来移行パスが残る**。前述の通り rusqlite 上流が
   sqlite-wasm-rs 経路を取り込む方向に動いているので、後で rusqlite を
   bump して native と web で同一 API に揃え直す道は開いている
   (今やる必要は無い)。

5. **`SQLITE_THREADSAFE=0` で実害が無い**。keysynth-web は cpal-wasm 経由で
   AudioWorklet スレッド + main thread の 2 本だが、library.db は UI
   スレッド側 (jukebox UI) からしか読まれない。MIDI 再生スレッドが
   library.db を叩くワークロードは現状無い。

### 不採用根拠

* **案 A (rusqlite[bundled] そのまま)**: 現バージョン (0.31) では wasm32 で
  ビルドが通らない。上流の wasm32 サポートが安定するまで待つコストが高い
  ($keysynth は trunk + GitHub Pages デプロイを既に出している)。

* **案 C (sqlite.org 公式 JS module)**: GitHub Pages の COOP/COEP 問題と、
  Rust 側の API が `wasm-bindgen` 経由の JS 呼び出しになり `rusqlite` 風味
  から最も遠い。OPFS 永続化が真に要るユースケース (= ユーザ録音や手書き
  譜面の保存) が立ち上がった段階で再評価すれば良い。

## 実装スコープ (この issue/PR の範囲外、後続 task の参考)

1. `Cargo.toml` に `[target.'cfg(target_arch = "wasm32")'.dependencies]`
   セクションで `sqlite-wasm-rs` を追加。`web` feature のゲートに
   入れるかは要検討 (Pages サイズ予算次第)。
2. `src/library_db/mod.rs` を `cfg(not(target_arch = "wasm32"))` で囲い、
   wasm32 用に薄い `library_db_wasm.rs` を `sqlite-wasm-rs` 上に書く。
3. `bench-out/library.db` を `include_bytes!` で焼き、起動時 memdb に
   `sqlite3_deserialize` 相当で投入 (sqlite-wasm-rs に対応 API あり)。
4. jukebox UI の `query_all() ORDER BY composer` 経路だけが web で必要 →
   adapter trait は最小限 (`list_songs` / `list_voices` / `query_by_*`
   程度) で良い。
5. テスト: `cargo test --target wasm32-unknown-unknown --features web` で
   "焼き込んだ DB から 14 曲取れる" を確認 (wasm-bindgen-test)。

## 残スコープ外 (= さらに別 issue)

* **play_log の OPFS 永続化** — sqlite-wasm-rs `opfs-sahpool` VFS で対応可能。
  ただし COOP/COEP / SharedArrayBuffer 制約が GitHub Pages では辛いので、
  まず IndexedDB バックエンド (`relaxed-idb`) を先に評価したほうが堅い。
* **Steel REPL から DB 叩く** (#56 Phase 2/3 で扱う既知の OOS)。
* **MTV 同期** (libray.db を更新したら wasm bundle を rebuild する CI 経路) —
  reproducible build artifact なので `cargo xtask rebuild-library` を
  Pages デプロイ前に走らせる手順だけ。

---

## Sources

* [sqlite-wasm-rs (crates.io)][swr-crates]
* [sqlite-wasm-rs (docs.rs)][swr-docs]
* [sqlite-wasm-rs (GitHub: Spxg/sqlite-wasm-rs)][swr-gh]
* [rusqlite issue #603 — Support wasm32 target][rusqlite-603]
* [rusqlite issue #827 — Can't build for wasm target][rusqlite-827]
* [rusqlite PR #1010 — Add wasm32-unknown-unknown support (closed)][pr-1010]
* [rusqlite PR #935 — wasm32 + memvfs feature][pr-935]
* [sqlite.org WASM/JS index][official-wasm]
* [sqlite.org WASM persistence guide][official-persistence]
* [Chrome for Developers — SQLite Wasm + OPFS][chrome-blog]
* [PowerSync — State of SQLite Persistence on the Web (2025-11)][powersync]

[swr-crates]: https://crates.io/crates/sqlite-wasm-rs
[swr-docs]: https://docs.rs/sqlite-wasm-rs
[swr-gh]: https://github.com/Spxg/sqlite-wasm-rs
[rusqlite-603]: https://github.com/rusqlite/rusqlite/issues/603
[rusqlite-827]: https://github.com/rusqlite/rusqlite/issues/827
[pr-1010]: https://github.com/rusqlite/rusqlite/pull/1010
[pr-935]: https://github.com/rusqlite/rusqlite/pull/935
[official-wasm]: https://sqlite.org/wasm
[official-persistence]: https://sqlite.org/wasm/doc/trunk/persistence.md
[chrome-blog]: https://developer.chrome.com/blog/sqlite-wasm-in-the-browser-backed-by-the-origin-private-file-system/
[powersync]: https://www.powersync.com/blog/sqlite-persistence-on-the-web
