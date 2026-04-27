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
//! Built-in slots (the modal presets, Salamander SFZ, GeneralUser SF2,
//! every synth engine) are seeded at startup. User-loaded SFZ files
//! and saved Custom presets persist to `~/.keysynth/voices.json`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::synth::{Engine, ModalParams, ModalPreset};

/// Top-level grouping in the side panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Category {
    Piano,
    Synth,
    Custom,
}

impl Category {
    pub fn label(self) -> &'static str {
        match self {
            Category::Piano => "Piano",
            Category::Synth => "Other (synth)",
            Category::Custom => "Custom",
        }
    }

    pub const ALL: &'static [Category] = &[Category::Piano, Category::Synth, Category::Custom];
}

/// One selectable voice in the browser.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoiceSlot {
    pub label: String,
    pub category: Category,
    pub engine: Engine,
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
    /// Built-ins are non-removable / non-renamable. User-added entries
    /// (Load SFZ..., Save current...) set this to false.
    #[serde(default = "default_true")]
    pub builtin: bool,
}

fn default_true() -> bool {
    true
}

impl VoiceSlot {
    pub fn synth(label: &str, engine: Engine) -> Self {
        Self {
            label: label.to_string(),
            category: Category::Synth,
            engine,
            params: None,
            asset_path: None,
            sf_program: None,
            sf_bank: None,
            modal_physics: false,
            builtin: true,
        }
    }

    pub fn modal(label: &str, preset: ModalPreset) -> Self {
        Self {
            label: label.to_string(),
            category: Category::Piano,
            engine: Engine::PianoModal,
            params: Some(preset.params()),
            asset_path: None,
            sf_program: None,
            sf_bank: None,
            modal_physics: matches!(preset, ModalPreset::Physics),
            builtin: true,
        }
    }

    pub fn sfz(label: &str, path: PathBuf, builtin: bool) -> Self {
        Self {
            label: label.to_string(),
            category: Category::Piano,
            engine: Engine::SfzPiano,
            params: None,
            asset_path: Some(path),
            sf_program: None,
            sf_bank: None,
            modal_physics: false,
            builtin,
        }
    }

    pub fn sf2(label: &str, path: PathBuf, program: u8, bank: u8, builtin: bool) -> Self {
        Self {
            label: label.to_string(),
            category: Category::Piano,
            engine: Engine::SfPiano,
            params: None,
            asset_path: Some(path),
            sf_program: Some(program),
            sf_bank: Some(bank),
            modal_physics: false,
            builtin,
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
            label: label.to_string(),
            // Custom category so it sits next to user-saved presets and
            // doesn't clutter the Piano family with a synth-style entry.
            category: Category::Custom,
            engine: Engine::Live,
            params: None,
            asset_path: None,
            sf_program: None,
            sf_bank: None,
            modal_physics: false,
            builtin: true,
        }
    }

    pub fn custom(label: &str, params: ModalParams, physics: bool) -> Self {
        Self {
            label: label.to_string(),
            category: Category::Custom,
            engine: Engine::PianoModal,
            params: Some(params),
            asset_path: None,
            sf_program: None,
            sf_bank: None,
            modal_physics: physics,
            builtin: false,
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
    /// Build a fresh library: hard-coded built-ins + any persisted
    /// user entries from `~/.keysynth/voices.json`.
    pub fn load(
        startup_sfz: Option<&Path>,
        startup_sf2: Option<&Path>,
        startup_engine: Engine,
    ) -> Self {
        let mut slots = Self::builtins(startup_sfz, startup_sf2);
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

    fn builtins(startup_sfz: Option<&Path>, startup_sf2: Option<&Path>) -> Vec<VoiceSlot> {
        let mut v = Vec::new();

        // Piano family — modal presets first.
        v.push(VoiceSlot::modal("Modal (default)", ModalPreset::Default));
        v.push(VoiceSlot::modal("Modal (Round-16)", ModalPreset::Round16));
        v.push(VoiceSlot::modal("Modal (Physics)", ModalPreset::Physics));
        v.push(VoiceSlot::modal("Modal (Bright)", ModalPreset::Bright));

        // KS-string-based piano variants.
        for (label, eng) in [
            ("KS Piano", Engine::Piano),
            ("KS Piano (thick)", Engine::PianoThick),
            ("KS Piano (lite)", Engine::PianoLite),
            ("KS Piano (5AM)", Engine::Piano5AM),
        ] {
            let mut s = VoiceSlot::synth(label, eng);
            s.category = Category::Piano;
            v.push(s);
        }

        // Sample-based piano slots — use the auto-discovered paths
        // from `main.rs` if present, else stub entries that the user
        // can repoint via the right-click menu.
        if let Some(p) = startup_sfz {
            let label = match p.file_stem().and_then(|s| s.to_str()) {
                Some(stem) => format!("Salamander SFZ ({stem})"),
                None => "Salamander SFZ".to_string(),
            };
            v.push(VoiceSlot::sfz(&label, p.to_path_buf(), true));
        }
        if let Some(p) = startup_sf2 {
            let label = match p.file_stem().and_then(|s| s.to_str()) {
                Some(stem) => format!("{stem} (SF2 piano)"),
                None => "SoundFont piano".to_string(),
            };
            v.push(VoiceSlot::sf2(&label, p.to_path_buf(), 0, 0, true));
        }

        // Hot-reload edit slot. Lives in the Custom column so it's
        // visually separated from the static synth/piano entries; the
        // user can toggle in and out of it the same way as their saved
        // presets.
        v.push(VoiceSlot::live("Live (hot edit)"));

        // Synth family.
        v.push(VoiceSlot::synth("Square", Engine::Square));
        v.push(VoiceSlot::synth("KS pluck", Engine::Ks));
        v.push(VoiceSlot::synth("KS rich", Engine::KsRich));
        v.push(VoiceSlot::synth("Sub (subtractive)", Engine::Sub));
        v.push(VoiceSlot::synth("FM bell", Engine::Fm));
        v.push(VoiceSlot::synth("Koto", Engine::Koto));

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_cover_every_category() {
        let lib = VoiceLibrary::load(None, None, Engine::Square);
        for cat in Category::ALL.iter().filter(|c| **c != Category::Custom) {
            assert!(
                lib.slots.iter().any(|s| s.category == *cat),
                "no built-in slot for {cat:?}"
            );
        }
    }

    #[test]
    fn cannot_remove_builtin() {
        let mut lib = VoiceLibrary::load(None, None, Engine::Square);
        let n = lib.slots.len();
        assert!(!lib.remove(0));
        assert_eq!(lib.slots.len(), n);
    }

    #[test]
    fn add_then_remove_user_slot() {
        let mut lib = VoiceLibrary::load(None, None, Engine::Square);
        let n = lib.slots.len();
        lib.add_and_select(VoiceSlot::custom("test", ModalParams::default(), false));
        assert_eq!(lib.slots.len(), n + 1);
        assert_eq!(lib.active, n);
        let user_idx = lib.slots.len() - 1;
        assert!(lib.remove(user_idx));
        assert_eq!(lib.slots.len(), n);
    }
}
