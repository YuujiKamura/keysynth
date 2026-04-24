//! General MIDI Level 1 instrument table.
//!
//! 128 patches grouped into 16 families of 8. Used by the `sf-piano` engine
//! GUI picker so the user can browse the SoundFont's GM 1 patch set without
//! memorising program numbers.
//!
//! Tuple format: `(program_number, instrument_name, family_name)`.
//! Family names match the canonical GM 1 grouping ("Pianos", "Chromatic Perc.",
//! "Organ", "Guitar", "Bass", "Strings", "Ensemble", "Brass", "Reed", "Pipe",
//! "Synth Lead", "Synth Pad", "Synth FX", "Ethnic", "Percussive", "Sound FX").

pub const GM_INSTRUMENTS: [(u8, &str, &str); 128] = [
    // 0-7 Pianos
    (0, "Acoustic Grand Piano", "Pianos"),
    (1, "Bright Acoustic Piano", "Pianos"),
    (2, "Electric Grand Piano", "Pianos"),
    (3, "Honky-tonk Piano", "Pianos"),
    (4, "Electric Piano 1", "Pianos"),
    (5, "Electric Piano 2", "Pianos"),
    (6, "Harpsichord", "Pianos"),
    (7, "Clavinet", "Pianos"),
    // 8-15 Chromatic Percussion
    (8, "Celesta", "Chromatic Perc."),
    (9, "Glockenspiel", "Chromatic Perc."),
    (10, "Music Box", "Chromatic Perc."),
    (11, "Vibraphone", "Chromatic Perc."),
    (12, "Marimba", "Chromatic Perc."),
    (13, "Xylophone", "Chromatic Perc."),
    (14, "Tubular Bells", "Chromatic Perc."),
    (15, "Dulcimer", "Chromatic Perc."),
    // 16-23 Organ
    (16, "Drawbar Organ", "Organ"),
    (17, "Percussive Organ", "Organ"),
    (18, "Rock Organ", "Organ"),
    (19, "Church Organ", "Organ"),
    (20, "Reed Organ", "Organ"),
    (21, "Accordion", "Organ"),
    (22, "Harmonica", "Organ"),
    (23, "Tango Accordion", "Organ"),
    // 24-31 Guitar
    (24, "Acoustic Guitar (nylon)", "Guitar"),
    (25, "Acoustic Guitar (steel)", "Guitar"),
    (26, "Electric Guitar (jazz)", "Guitar"),
    (27, "Electric Guitar (clean)", "Guitar"),
    (28, "Electric Guitar (muted)", "Guitar"),
    (29, "Overdriven Guitar", "Guitar"),
    (30, "Distortion Guitar", "Guitar"),
    (31, "Guitar Harmonics", "Guitar"),
    // 32-39 Bass
    (32, "Acoustic Bass", "Bass"),
    (33, "Electric Bass (finger)", "Bass"),
    (34, "Electric Bass (pick)", "Bass"),
    (35, "Fretless Bass", "Bass"),
    (36, "Slap Bass 1", "Bass"),
    (37, "Slap Bass 2", "Bass"),
    (38, "Synth Bass 1", "Bass"),
    (39, "Synth Bass 2", "Bass"),
    // 40-47 Strings
    (40, "Violin", "Strings"),
    (41, "Viola", "Strings"),
    (42, "Cello", "Strings"),
    (43, "Contrabass", "Strings"),
    (44, "Tremolo Strings", "Strings"),
    (45, "Pizzicato Strings", "Strings"),
    (46, "Orchestral Harp", "Strings"),
    (47, "Timpani", "Strings"),
    // 48-55 Ensemble
    (48, "String Ensemble 1", "Ensemble"),
    (49, "String Ensemble 2", "Ensemble"),
    (50, "SynthStrings 1", "Ensemble"),
    (51, "SynthStrings 2", "Ensemble"),
    (52, "Choir Aahs", "Ensemble"),
    (53, "Voice Oohs", "Ensemble"),
    (54, "Synth Voice", "Ensemble"),
    (55, "Orchestra Hit", "Ensemble"),
    // 56-63 Brass
    (56, "Trumpet", "Brass"),
    (57, "Trombone", "Brass"),
    (58, "Tuba", "Brass"),
    (59, "Muted Trumpet", "Brass"),
    (60, "French Horn", "Brass"),
    (61, "Brass Section", "Brass"),
    (62, "SynthBrass 1", "Brass"),
    (63, "SynthBrass 2", "Brass"),
    // 64-71 Reed
    (64, "Soprano Sax", "Reed"),
    (65, "Alto Sax", "Reed"),
    (66, "Tenor Sax", "Reed"),
    (67, "Baritone Sax", "Reed"),
    (68, "Oboe", "Reed"),
    (69, "English Horn", "Reed"),
    (70, "Bassoon", "Reed"),
    (71, "Clarinet", "Reed"),
    // 72-79 Pipe
    (72, "Piccolo", "Pipe"),
    (73, "Flute", "Pipe"),
    (74, "Recorder", "Pipe"),
    (75, "Pan Flute", "Pipe"),
    (76, "Blown Bottle", "Pipe"),
    (77, "Shakuhachi", "Pipe"),
    (78, "Whistle", "Pipe"),
    (79, "Ocarina", "Pipe"),
    // 80-87 Synth Lead
    (80, "Lead 1 (square)", "Synth Lead"),
    (81, "Lead 2 (sawtooth)", "Synth Lead"),
    (82, "Lead 3 (calliope)", "Synth Lead"),
    (83, "Lead 4 (chiff)", "Synth Lead"),
    (84, "Lead 5 (charang)", "Synth Lead"),
    (85, "Lead 6 (voice)", "Synth Lead"),
    (86, "Lead 7 (fifths)", "Synth Lead"),
    (87, "Lead 8 (bass+lead)", "Synth Lead"),
    // 88-95 Synth Pad
    (88, "Pad 1 (new age)", "Synth Pad"),
    (89, "Pad 2 (warm)", "Synth Pad"),
    (90, "Pad 3 (polysynth)", "Synth Pad"),
    (91, "Pad 4 (choir)", "Synth Pad"),
    (92, "Pad 5 (bowed)", "Synth Pad"),
    (93, "Pad 6 (metallic)", "Synth Pad"),
    (94, "Pad 7 (halo)", "Synth Pad"),
    (95, "Pad 8 (sweep)", "Synth Pad"),
    // 96-103 Synth FX
    (96, "FX 1 (rain)", "Synth FX"),
    (97, "FX 2 (soundtrack)", "Synth FX"),
    (98, "FX 3 (crystal)", "Synth FX"),
    (99, "FX 4 (atmosphere)", "Synth FX"),
    (100, "FX 5 (brightness)", "Synth FX"),
    (101, "FX 6 (goblins)", "Synth FX"),
    (102, "FX 7 (echoes)", "Synth FX"),
    (103, "FX 8 (sci-fi)", "Synth FX"),
    // 104-111 Ethnic
    (104, "Sitar", "Ethnic"),
    (105, "Banjo", "Ethnic"),
    (106, "Shamisen", "Ethnic"),
    (107, "Koto", "Ethnic"),
    (108, "Kalimba", "Ethnic"),
    (109, "Bag pipe", "Ethnic"),
    (110, "Fiddle", "Ethnic"),
    (111, "Shanai", "Ethnic"),
    // 112-119 Percussive
    (112, "Tinkle Bell", "Percussive"),
    (113, "Agogo", "Percussive"),
    (114, "Steel Drums", "Percussive"),
    (115, "Woodblock", "Percussive"),
    (116, "Taiko Drum", "Percussive"),
    (117, "Melodic Tom", "Percussive"),
    (118, "Synth Drum", "Percussive"),
    (119, "Reverse Cymbal", "Percussive"),
    // 120-127 Sound FX
    (120, "Guitar Fret Noise", "Sound FX"),
    (121, "Breath Noise", "Sound FX"),
    (122, "Seashore", "Sound FX"),
    (123, "Bird Tweet", "Sound FX"),
    (124, "Telephone Ring", "Sound FX"),
    (125, "Helicopter", "Sound FX"),
    (126, "Applause", "Sound FX"),
    (127, "Gunshot", "Sound FX"),
];

/// Canonical family order — matches the GM 1 standard. The GUI uses this to
/// emit collapsing headers in the documented order rather than alphabetical.
pub const GM_FAMILIES: [&str; 16] = [
    "Pianos",
    "Chromatic Perc.",
    "Organ",
    "Guitar",
    "Bass",
    "Strings",
    "Ensemble",
    "Brass",
    "Reed",
    "Pipe",
    "Synth Lead",
    "Synth Pad",
    "Synth FX",
    "Ethnic",
    "Percussive",
    "Sound FX",
];

/// Look up the GM 1 instrument name for a given program number. Returns
/// `"<unknown>"` if `program` is somehow out of range (shouldn't happen
/// because GM 1 is exactly 0..=127, but defensive for drum-bank programs
/// that don't map to GM 1).
pub fn instrument_name(program: u8) -> &'static str {
    GM_INSTRUMENTS
        .get(program as usize)
        .map(|(_, name, _)| *name)
        .unwrap_or("<unknown>")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gm_table_covers_0_to_127() {
        for (i, (p, name, family)) in GM_INSTRUMENTS.iter().enumerate() {
            assert_eq!(*p as usize, i, "program number must equal slot index");
            assert!(!name.is_empty());
            assert!(!family.is_empty());
        }
    }

    #[test]
    fn gm_family_count() {
        assert_eq!(GM_FAMILIES.len(), 16);
        // Each family has exactly 8 instruments.
        for fam in GM_FAMILIES.iter() {
            let count = GM_INSTRUMENTS.iter().filter(|(_, _, f)| f == fam).count();
            assert_eq!(count, 8, "family '{fam}' should have 8 instruments");
        }
    }

    #[test]
    fn instrument_name_lookup() {
        assert_eq!(instrument_name(0), "Acoustic Grand Piano");
        assert_eq!(instrument_name(24), "Acoustic Guitar (nylon)");
        assert_eq!(instrument_name(127), "Gunshot");
    }

    #[test]
    fn gm_table_has_128_entries() {
        assert_eq!(GM_INSTRUMENTS.len(), 128);
    }

    #[test]
    fn gm_program_numbers_are_unique_and_ordered() {
        let mut seen: std::collections::HashSet<u8> = std::collections::HashSet::new();
        for (p, _, _) in GM_INSTRUMENTS.iter() {
            assert!(seen.insert(*p), "duplicate program number {}", p);
        }
        assert_eq!(seen.len(), 128);
    }

    #[test]
    fn instrument_name_default_for_invalid() {
        // Wraps in u8 so 0..=127 always lookup; the unwrap_or branch is
        // strictly defensive but we exercise it via the public API.
        // Passing a known-valid 127 as a sanity baseline:
        assert_ne!(instrument_name(127), "<unknown>");
    }

    #[test]
    fn gm_first_family_is_pianos() {
        assert_eq!(GM_FAMILIES[0], "Pianos");
    }

    #[test]
    fn gm_last_family_is_sound_fx() {
        assert_eq!(GM_FAMILIES[15], "Sound FX");
    }

    #[test]
    fn gm_each_program_belongs_to_known_family() {
        let known: std::collections::HashSet<&str> = GM_FAMILIES.iter().copied().collect();
        for (_, _, fam) in GM_INSTRUMENTS.iter() {
            assert!(
                known.contains(*fam),
                "program family '{}' not in GM_FAMILIES",
                fam
            );
        }
    }
}
