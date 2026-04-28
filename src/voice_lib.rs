//! Voice browser library: catalog of selectable instruments for the
//! egui side-panel browser in `ui.rs`.
//!
//! Each entry is a `VoiceSlot` carrying everything needed to make a
//! sound: which `Engine` to use, optional `ModalParams` for the
//! piano-modal engine, and an optional asset path for SFZ / SF2.
//! Selecting a slot in the GUI triggers (1) engine swap, (2) preset
//! apply, (3) asset hot-load — in one operation, picked up by the
//! next `note_on`.
//!
//! ## Catalog sources
//!
//! The visible catalog is the merge of two sources:
//!
//! 1. **Static catalog** (`builtins`) — synth engines (square / KS /
//!    sub / FM / koto), the SFZ + SF2 sample slots, and the
//!    "Live (hot edit)" rebuild slot. These have no plugin dirs to
//!    discover, so they stay hard-coded here.
//!
//! 2. **Plugin discovery** (`discover_plugin_voices`) — scans
//!    `voices_live/<name>/Cargo.toml` for a
//!    `[package.metadata.keysynth-voice]` table and emits one
//!    `VoiceSlot` per match. New voice plugins (e.g. a future
//!    `voices_live/clarinet/`) appear in the GUI by landing the
//!    directory + Cargo.toml metadata only — no edits to this file,
//!    `ui.rs`, the `Engine` enum, or Cargo deps. This is the contract
//!    the substrate (PR #40 / #41 / #46) is built to enable; manual
//!    enumeration here would defeat the point.
//!
//! User-loaded SFZ files and saved Custom presets persist to
//! `~/.keysynth/voices.json` and are appended on top of the merged
//! built-in catalog.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::synth::{Engine, ModalParams, ModalPreset};

/// How a voice reacts to MIDI / GUI note-off. Two physical models:
///
/// * `Damper` — note-off lowers the damper (or runs the release-stage
///   envelope) and the voice fades quickly. This is the piano model:
///   the action takes the felt off the string, the string is muted on
///   purpose. Hosts call `VoiceImpl::trigger_release()`.
///
/// * `Natural` — note-off has no physical analogue. A plucked string
///   (guitar, koto) has nothing the player can "release" — the hammer
///   was never holding the string in the first place. The voice fades
///   on its own loop-filter decay and is retired by the host once
///   `is_done()` flips. Hosts MUST NOT call `trigger_release()`; doing
///   so cuts the natural ring short and produces the "key-up kills the
///   note" bug the chord-pad regression caught.
///
/// `Default` returns `Damper` so legacy plugins (and every entry in the
/// static catalog) keep their pre-existing release semantics until they
/// opt into `decay_model = "natural"` in their Cargo.toml metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecayModel {
    Damper,
    Natural,
}

impl Default for DecayModel {
    fn default() -> Self {
        DecayModel::Damper
    }
}

/// Top-level grouping in the side panel. Order in `ALL` is the
/// rendering order: physically-modeled voices first (piano then
/// guitar), the lightweight synth set, sample-based pianos, and
/// finally user/custom slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Category {
    Piano,
    Guitar,
    Synth,
    Samples,
    Custom,
}

impl Category {
    pub fn label(self) -> &'static str {
        match self {
            Category::Piano => "Piano (modeled)",
            Category::Guitar => "Guitar (modeled)",
            Category::Synth => "Synth",
            Category::Samples => "Samples",
            Category::Custom => "Custom",
        }
    }

    pub const ALL: &'static [Category] = &[
        Category::Piano,
        Category::Guitar,
        Category::Synth,
        Category::Samples,
        Category::Custom,
    ];
}

/// Editorial recommendation tier shown next to the slot label so the
/// browser is self-documenting: which voices are the "use this first"
/// picks, which are stable secondary options, and which are kept around
/// for A/B oracle / regression purposes but should not be the default.
///
/// Sort key: `Best < Stable < Experimental`. The browser sorts entries
/// within each category by this rank so the recommended pick floats to
/// the top.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Recommend {
    Best,
    Stable,
    Experimental,
}

impl Recommend {
    pub fn label(self) -> &'static str {
        match self {
            Recommend::Best => "★Best",
            Recommend::Stable => "Stable",
            Recommend::Experimental => "⚠Experimental",
        }
    }

    pub fn rank(self) -> u8 {
        match self {
            Recommend::Best => 0,
            Recommend::Stable => 1,
            Recommend::Experimental => 2,
        }
    }
}

fn default_recommend() -> Recommend {
    Recommend::Stable
}

/// One selectable voice in the browser.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoiceSlot {
    pub label: String,
    pub category: Category,
    pub engine: Engine,
    /// One-line description shown under the label in the side panel.
    /// Empty string for legacy / user-added slots.
    #[serde(default)]
    pub description: String,
    /// Editorial tier (Best / Stable / Experimental). User-added slots
    /// default to `Stable` via `default_recommend()`.
    #[serde(default = "default_recommend")]
    pub recommend: Recommend,
    /// Whether MIDI / GUI note-off triggers an immediate damper release
    /// (piano-style) or lets the string ring out on its own (plucked-
    /// string voices like guitar / koto). Persisted with the slot so a
    /// user-saved Custom built on top of a guitar plugin keeps its
    /// "natural" behaviour even if the source manifest later flips.
    #[serde(default)]
    pub decay_model: DecayModel,
    /// Only meaningful for `Engine::PianoModal`. When `Some`, the
    /// global `ModalParams` cell is overwritten on selection.
    #[serde(default)]
    pub params: Option<ModalParams>,
    /// `.sfz` for `Engine::SfzPiano`, `.sf2` for `Engine::SfPiano`,
    /// `None` for everything else.
    #[serde(default)]
    pub asset_path: Option<PathBuf>,
    /// Optional GM program for `Engine::SfPiano` (defaults to 0).
    #[serde(default)]
    pub sf_program: Option<u8>,
    #[serde(default)]
    pub sf_bank: Option<u8>,
    /// For `Engine::PianoModal` only. When true, selecting this slot
    /// flips the global `MODAL_PHYSICS` flag so `make_voice` constructs
    /// voices via `ModalPianoVoice::from_physics`. When false, the LUT
    /// path is used. Mirrors `ModalPreset::Physics`'s `apply()`
    /// behaviour for non-preset (Custom) slots.
    #[serde(default)]
    pub modal_physics: bool,
    /// For `Engine::Live` slots that point at a specific
    /// `voices_live/<name>/` cdylib. When set, selecting the slot asks
    /// the `Reloader` to make `<name>` the active live slot (building
    /// it from `live_crate_subdir` if no factory has been loaded yet).
    /// `None` means "use whatever the reloader is currently watching"
    /// (i.e. the `Live (hot edit)` slot's behaviour).
    #[serde(default)]
    pub live_slot_name: Option<String>,
    /// Path of the `voices_live/<subdir>/` crate, relative to the
    /// reloader's parent directory. Used when `live_slot_name` is set
    /// but the slot has not been built yet.
    #[serde(default)]
    pub live_crate_subdir: Option<String>,
    /// Built-ins are non-removable / non-renamable. User-added entries
    /// (Load SFZ..., Save current...) set this to false.
    #[serde(default = "default_true")]
    pub builtin: bool,
}

fn default_true() -> bool {
    true
}

impl VoiceSlot {
    /// Bare-bones builder used by the typed helpers below and by the
    /// `builtins()` table to fill in `description` / `recommend` /
    /// `category` overrides without re-spelling every other field.
    fn base(label: &str, engine: Engine) -> Self {
        Self {
            label: label.to_string(),
            category: Category::Synth,
            engine,
            description: String::new(),
            recommend: Recommend::Stable,
            decay_model: DecayModel::Damper,
            params: None,
            asset_path: None,
            sf_program: None,
            sf_bank: None,
            modal_physics: false,
            live_slot_name: None,
            live_crate_subdir: None,
            builtin: true,
        }
    }

    pub fn synth(label: &str, engine: Engine) -> Self {
        Self::base(label, engine)
    }

    pub fn modal(label: &str, preset: ModalPreset) -> Self {
        Self {
            category: Category::Piano,
            params: Some(preset.params()),
            modal_physics: matches!(preset, ModalPreset::Physics),
            ..Self::base(label, Engine::PianoModal)
        }
    }

    pub fn sfz(label: &str, path: PathBuf, builtin: bool) -> Self {
        Self {
            category: Category::Samples,
            asset_path: Some(path),
            builtin,
            ..Self::base(label, Engine::SfzPiano)
        }
    }

    pub fn sf2(label: &str, path: PathBuf, program: u8, bank: u8, builtin: bool) -> Self {
        Self {
            category: Category::Samples,
            asset_path: Some(path),
            sf_program: Some(program),
            sf_bank: Some(bank),
            builtin,
            ..Self::base(label, Engine::SfPiano)
        }
    }

    /// Hot-edit slot: routes through `Engine::Live`, which calls into
    /// the registered `live_reload::Reloader`. `voice_lib` doesn't depend
    /// on the reloader directly — it just emits a slot whose engine is
    /// `Engine::Live`. Status (current dll path, last reload, errors)
    /// is rendered by the side panel from the `Reloader` handle stashed
    /// on `AppContext`.
    pub fn live(label: &str) -> Self {
        Self {
            // Custom category so it sits next to user-saved presets and
            // doesn't clutter the Piano family with a synth-style entry.
            category: Category::Custom,
            ..Self::base(label, Engine::Live)
        }
    }

    /// Named live slot pointing at a specific `voices_live/<name>/`
    /// cdylib. Used by the modeled-guitar entries — clicking the slot
    /// asks the reloader to load (and if necessary build) that crate
    /// under `slot_name`. The reloader is not consulted by `voice_lib`
    /// itself; the wiring lives in `ui::App::apply_slot`.
    pub fn live_named(label: &str, slot_name: &str, crate_subdir: &str) -> Self {
        Self {
            live_slot_name: Some(slot_name.to_string()),
            live_crate_subdir: Some(crate_subdir.to_string()),
            ..Self::base(label, Engine::Live)
        }
    }

    pub fn custom(label: &str, params: ModalParams, physics: bool) -> Self {
        Self {
            category: Category::Custom,
            params: Some(params),
            modal_physics: physics,
            builtin: false,
            ..Self::base(label, Engine::PianoModal)
        }
    }

    /// Decorate any slot with editorial metadata in one chained call.
    /// Lets `builtins()` stay declarative — each row reads as
    /// "this engine, in this category, with this tier and blurb".
    pub fn with_meta(mut self, recommend: Recommend, description: &str) -> Self {
        self.recommend = recommend;
        self.description = description.to_string();
        self
    }

    pub fn in_category(mut self, category: Category) -> Self {
        self.category = category;
        self
    }

    /// Stable identifier surfaced to the E2E layout test as
    /// "engine slot reference". Static engine variants stringify via
    /// `Debug`; live slots use their `live:<name>` form so the two
    /// `Engine::Live` builtins still come out distinct.
    pub fn engine_slot_ref(&self) -> String {
        match (&self.engine, self.live_slot_name.as_ref()) {
            (Engine::Live, Some(name)) => format!("live:{name}"),
            (engine, _) => format!("{engine:?}"),
        }
    }
}

/// Persisted slice of `VoiceLibrary` — only user-added entries are
/// written to disk. Built-ins are re-seeded from code each launch so
/// they stay in sync with the binary.
#[derive(Default, Serialize, Deserialize)]
struct PersistedLibrary {
    #[serde(default)]
    entries: Vec<VoiceSlot>,
}

pub struct VoiceLibrary {
    pub slots: Vec<VoiceSlot>,
    /// Index into `slots` for the currently-selected voice.
    pub active: usize,
}

impl VoiceLibrary {
    /// Build a fresh library: static catalog + plugin discovery from
    /// `voices_live/` (when reachable) + any persisted user entries
    /// from `~/.keysynth/voices.json`.
    ///
    /// The discovery root is resolved from `voices_live_root`; pass
    /// `None` to skip plugin discovery entirely (used by tests that
    /// don't want the real `voices_live/` tree leaking in).
    pub fn load(
        startup_sfz: Option<&Path>,
        startup_sf2: Option<&Path>,
        startup_engine: Engine,
        voices_live_root: Option<&Path>,
    ) -> Self {
        let mut slots = Self::builtins(startup_sfz, startup_sf2);
        if let Some(root) = voices_live_root {
            slots.extend(discover_plugin_voices(root));
        }
        if let Some(extra) = Self::read_persisted() {
            for slot in extra.entries {
                slots.push(slot);
            }
        }
        let active = slots
            .iter()
            .position(|s| s.engine == startup_engine)
            .unwrap_or(0);
        Self { slots, active }
    }

    /// Static catalog: synth engines + SFZ/SF2 sample slots + the
    /// hot-edit Live slot. These exist independently of the
    /// `voices_live/` plugin tree — they have no Cargo manifest to
    /// discover from, so they're enumerated here. Modeled-instrument
    /// voices (every piano variant + both guitars) are NOT in this
    /// list; they come from `discover_plugin_voices()`.
    pub fn builtins(startup_sfz: Option<&Path>, startup_sf2: Option<&Path>) -> Vec<VoiceSlot> {
        let mut v = Vec::new();

        // ─── Synth ────────────────────────────────────────────────
        // Cheap reference voices used for diagnostics, latency
        // testing, and the lower-stakes synth presets. All routed
        // through their static `Engine` variant.
        v.push(
            VoiceSlot::synth("Square", Engine::Square)
                .with_meta(Recommend::Stable, "NES-style pulse"),
        );
        v.push(
            VoiceSlot::synth("KS", Engine::Ks).with_meta(Recommend::Stable, "Karplus-Strong basic"),
        );
        v.push(
            VoiceSlot::synth("KS Rich", Engine::KsRich)
                .with_meta(Recommend::Stable, "KS with allpass dispersion"),
        );
        v.push(
            VoiceSlot::synth("Sub", Engine::Sub).with_meta(Recommend::Stable, "analog subtractive"),
        );
        v.push(
            VoiceSlot::synth("FM", Engine::Fm).with_meta(Recommend::Stable, "2-op DX7-ish bell"),
        );
        v.push(
            VoiceSlot::synth("Koto", Engine::Koto)
                .with_meta(Recommend::Stable, "pluck-position digital waveguide"),
        );

        // ─── Samples ──────────────────────────────────────────────
        // Always emitted so the side panel always shows a sample
        // section; if `main.rs` discovered an asset on disk we burn
        // it into the slot, otherwise the user can repoint via Load…
        let sfz_path = startup_sfz.map(|p| p.to_path_buf());
        let sfz_label = sfz_path
            .as_ref()
            .and_then(|p| p.file_stem().and_then(|s| s.to_str()))
            .map(|stem| format!("Salamander SFZ ({stem})"))
            .unwrap_or_else(|| "Salamander SFZ (Grand)".to_string());
        let mut sfz_slot = VoiceSlot::base(&sfz_label, Engine::SfzPiano);
        sfz_slot.category = Category::Samples;
        sfz_slot.asset_path = sfz_path;
        v.push(sfz_slot.with_meta(Recommend::Best, "high-quality Salamander Grand reference"));

        let sf2_path = startup_sf2.map(|p| p.to_path_buf());
        let sf2_label = sf2_path
            .as_ref()
            .and_then(|p| p.file_stem().and_then(|s| s.to_str()))
            .map(|stem| format!("{stem} (SF2)"))
            .unwrap_or_else(|| "GeneralUser SF2 (GM)".to_string());
        let mut sf2_slot = VoiceSlot::base(&sf2_label, Engine::SfPiano);
        sf2_slot.category = Category::Samples;
        sf2_slot.asset_path = sf2_path;
        sf2_slot.sf_program = Some(0);
        sf2_slot.sf_bank = Some(0);
        v.push(sf2_slot.with_meta(Recommend::Stable, "GM-set SoundFont, multi-instrument"));

        // ─── Custom ───────────────────────────────────────────────
        // Hot-reload edit slot. Sits in Custom so user-saved presets
        // stay visually grouped together. Watches the parent
        // `voices_live/` crate via the `Reloader`'s file watcher.
        v.push(VoiceSlot::live("Live (hot edit)").with_meta(
            Recommend::Stable,
            "watch & rebuild voices_live/src on file save",
        ));

        v
    }

    fn config_path() -> Option<PathBuf> {
        // `std::env::home_dir` is back as of recent stable, but the
        // safe path is to read `HOME` (Unix) / `USERPROFILE` (Windows).
        let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
        Some(PathBuf::from(home).join(".keysynth").join("voices.json"))
    }

    fn read_persisted() -> Option<PersistedLibrary> {
        let path = Self::config_path()?;
        let bytes = std::fs::read(&path).ok()?;
        match serde_json::from_slice::<PersistedLibrary>(&bytes) {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("voice_lib: ignoring corrupt {}: {e}", path.display());
                None
            }
        }
    }

    /// Persist user-added entries (everything with `builtin = false`).
    pub fn save(&self) -> std::io::Result<()> {
        let Some(path) = Self::config_path() else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let entries: Vec<VoiceSlot> = self.slots.iter().filter(|s| !s.builtin).cloned().collect();
        let body = serde_json::to_vec_pretty(&PersistedLibrary { entries })
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(&path, body)?;
        Ok(())
    }

    /// Append a new (user-added) slot and select it.
    pub fn add_and_select(&mut self, slot: VoiceSlot) {
        self.slots.push(slot);
        self.active = self.slots.len() - 1;
        let _ = self.save();
    }

    /// Remove the slot at `idx` if it's user-added. Returns true if
    /// removed. Built-ins are protected.
    pub fn remove(&mut self, idx: usize) -> bool {
        if self.slots.get(idx).map(|s| s.builtin).unwrap_or(true) {
            return false;
        }
        self.slots.remove(idx);
        if self.active >= self.slots.len() {
            self.active = self.slots.len().saturating_sub(1);
        }
        let _ = self.save();
        true
    }

    /// Rename a user-added slot. Built-ins are protected.
    pub fn rename(&mut self, idx: usize, new_label: String) -> bool {
        match self.slots.get_mut(idx) {
            Some(s) if !s.builtin => {
                s.label = new_label;
                let _ = self.save();
                true
            }
            _ => false,
        }
    }

    pub fn active_slot(&self) -> Option<&VoiceSlot> {
        self.slots.get(self.active)
    }
}

/// Manifest schema for the `[package.metadata.keysynth-voice]` block
/// each `voices_live/<name>/Cargo.toml` opts into. Only this block is
/// inspected — the rest of the Cargo.toml is irrelevant to the GUI
/// catalog and is left untouched. All four fields are required so the
/// side panel always has a label, a category column to file the entry
/// under, an editorial tier, and one-line caption text.
#[cfg(feature = "native")]
#[derive(Debug, Deserialize)]
struct PluginVoiceManifest {
    display_name: String,
    category: String,
    recommend: String,
    description: String,
    /// Optional override for the live slot identifier the reloader
    /// sees. Defaults to the directory name so most plugins can omit
    /// it; needed only when two plugins want distinct slot names that
    /// don't match their crate dir (e.g. legacy `voices_live/guitar/`
    /// surfaced under `live:guitar_ks` to disambiguate from the STK
    /// port). Treated as a free-form string — not validated against
    /// existing slot names.
    #[serde(default)]
    slot_name: Option<String>,
    /// Note-off semantics: `"damper"` (piano-style trigger_release) or
    /// `"natural"` (plucked-string voices, host must NOT call
    /// trigger_release). Optional; missing or unknown values map to
    /// `DecayModel::default() == Damper` so existing plugins keep
    /// their pre-PR behaviour without manifest churn.
    #[serde(default)]
    decay_model: Option<String>,
}

/// Top-level shape of a `voices_live/<name>/Cargo.toml`. We don't care
/// about most of it — the only nested table we deserialize is
/// `package.metadata.keysynth-voice`. Anything missing or
/// non-conforming results in `None` and a skipped directory.
#[cfg(feature = "native")]
#[derive(Debug, Deserialize)]
struct PluginManifestRoot {
    package: Option<PluginManifestPackage>,
}

#[cfg(feature = "native")]
#[derive(Debug, Deserialize)]
struct PluginManifestPackage {
    metadata: Option<PluginManifestMetadata>,
}

#[cfg(feature = "native")]
#[derive(Debug, Deserialize)]
struct PluginManifestMetadata {
    #[serde(rename = "keysynth-voice")]
    keysynth_voice: Option<PluginVoiceManifest>,
}

/// Scan a `voices_live/` root, parse `<name>/Cargo.toml` for each
/// subdirectory, and emit a `VoiceSlot` for every manifest that
/// declares a `[package.metadata.keysynth-voice]` block. Entries are
/// returned sorted by directory name so the result is stable across
/// platforms (filesystem read_dir order isn't).
///
/// This is the single substrate hook that lets new voice plugins land
/// in the GUI by directory addition alone — no edits to `voice_lib.rs`,
/// `ui.rs`, the `Engine` enum, or top-level Cargo deps. A directory
/// without the metadata block, or with a malformed Cargo.toml, is
/// silently skipped (logged at warn level via stderr) so a broken
/// plugin can't take down the side panel.
#[cfg(feature = "native")]
pub fn discover_plugin_voices(voices_live_root: &Path) -> Vec<VoiceSlot> {
    let read = match std::fs::read_dir(voices_live_root) {
        Ok(r) => r,
        Err(e) => {
            eprintln!(
                "voice_lib: plugin discovery skipped — cannot read {}: {e}",
                voices_live_root.display()
            );
            return Vec::new();
        }
    };

    let mut entries: Vec<(String, VoiceSlot)> = Vec::new();
    for dirent in read.flatten() {
        let path = dirent.path();
        if !path.is_dir() {
            continue;
        }
        let manifest_path = path.join("Cargo.toml");
        if !manifest_path.is_file() {
            continue;
        }
        let dir_name = match path.file_name().and_then(|s| s.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        let bytes = match std::fs::read_to_string(&manifest_path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "voice_lib: skip plugin '{dir_name}' — read {}: {e}",
                    manifest_path.display()
                );
                continue;
            }
        };

        let root: PluginManifestRoot = match toml::from_str(&bytes) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("voice_lib: skip plugin '{dir_name}' — toml parse: {e}");
                continue;
            }
        };
        let Some(meta) = root
            .package
            .and_then(|p| p.metadata)
            .and_then(|m| m.keysynth_voice)
        else {
            // Manifest exists but doesn't opt into the GUI catalog —
            // not an error, just not a voice plugin (could be the
            // parent `voices_live/Cargo.toml` itself, or a future
            // crate that's not surfaced in the side panel).
            continue;
        };

        let category = match parse_category(&meta.category) {
            Some(c) => c,
            None => {
                eprintln!(
                    "voice_lib: skip plugin '{dir_name}' — unknown category '{}'",
                    meta.category
                );
                continue;
            }
        };
        let recommend = match parse_recommend(&meta.recommend) {
            Some(r) => r,
            None => {
                eprintln!(
                    "voice_lib: skip plugin '{dir_name}' — unknown recommend '{}'",
                    meta.recommend
                );
                continue;
            }
        };

        let decay_model = meta
            .decay_model
            .as_deref()
            .map(|s| {
                parse_decay_model(s).unwrap_or_else(|| {
                    eprintln!(
                        "voice_lib: plugin '{dir_name}' decay_model='{s}' unknown; defaulting to Damper"
                    );
                    DecayModel::default()
                })
            })
            .unwrap_or_default();
        let slot_name = meta.slot_name.unwrap_or_else(|| dir_name.clone());
        let mut slot = VoiceSlot::live_named(&meta.display_name, &slot_name, &dir_name)
            .in_category(category)
            .with_meta(recommend, &meta.description);
        slot.decay_model = decay_model;
        entries.push((dir_name, slot));
    }

    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries.into_iter().map(|(_, s)| s).collect()
}

#[cfg(feature = "native")]
fn parse_category(s: &str) -> Option<Category> {
    match s {
        "Piano" => Some(Category::Piano),
        "Guitar" => Some(Category::Guitar),
        "Synth" => Some(Category::Synth),
        "Samples" => Some(Category::Samples),
        "Custom" => Some(Category::Custom),
        _ => None,
    }
}

#[cfg(feature = "native")]
fn parse_recommend(s: &str) -> Option<Recommend> {
    match s {
        "Best" => Some(Recommend::Best),
        "Stable" => Some(Recommend::Stable),
        "Experimental" => Some(Recommend::Experimental),
        _ => None,
    }
}

/// Map a manifest `decay_model = "..."` string to the enum. Lower-case
/// is the canonical form to match the rest of the manifest schema (which
/// uses snake_case keys). Returns `None` for unknown values; callers
/// fall back to `DecayModel::default()` (= `Damper`) and log a warning.
#[cfg(feature = "native")]
pub(crate) fn parse_decay_model(s: &str) -> Option<DecayModel> {
    match s {
        "damper" => Some(DecayModel::Damper),
        "natural" => Some(DecayModel::Natural),
        _ => None,
    }
}

/// No-op stub for the wasm/web build: plugin discovery is filesystem-
/// scoped and the wasm bundle has no `voices_live/` tree to read from
/// (the cdylib substrate is native-only). Keeps `VoiceLibrary::load`
/// callable from the same call sites under both feature flags without
/// `cfg` noise at every call.
#[cfg(not(feature = "native"))]
pub fn discover_plugin_voices(_voices_live_root: &Path) -> Vec<VoiceSlot> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_builtins_cover_synth_and_samples() {
        // Static catalog (without plugin discovery) must always contain
        // at least one Synth and one Samples entry — those are the two
        // categories the side panel populates from `builtins()`. Piano
        // / Guitar entries are owned by `discover_plugin_voices()` and
        // intentionally absent from this list.
        let lib = VoiceLibrary::load(None, None, Engine::Square, None);
        for cat in [Category::Synth, Category::Samples] {
            assert!(
                lib.slots.iter().any(|s| s.category == cat),
                "no built-in slot for {cat:?}"
            );
        }
    }

    #[test]
    fn cannot_remove_builtin() {
        let mut lib = VoiceLibrary::load(None, None, Engine::Square, None);
        let n = lib.slots.len();
        assert!(!lib.remove(0));
        assert_eq!(lib.slots.len(), n);
    }

    #[test]
    fn add_then_remove_user_slot() {
        let mut lib = VoiceLibrary::load(None, None, Engine::Square, None);
        let n = lib.slots.len();
        lib.add_and_select(VoiceSlot::custom("test", ModalParams::default(), false));
        assert_eq!(lib.slots.len(), n + 1);
        assert_eq!(lib.active, n);
        let user_idx = lib.slots.len() - 1;
        assert!(lib.remove(user_idx));
        assert_eq!(lib.slots.len(), n);
    }
}
