use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PitchClass(u8);

impl PitchClass {
    pub const C: Self = Self(0);
    pub const C_SHARP: Self = Self(1);
    pub const D: Self = Self(2);
    pub const D_SHARP: Self = Self(3);
    pub const E: Self = Self(4);
    pub const F: Self = Self(5);
    pub const F_SHARP: Self = Self(6);
    pub const G: Self = Self(7);
    pub const G_SHARP: Self = Self(8);
    pub const A: Self = Self(9);
    pub const A_SHARP: Self = Self(10);
    pub const B: Self = Self(11);

    pub fn new(value: u8) -> Self {
        Self(value % 12)
    }

    pub fn as_u8(self) -> u8 {
        self.0
    }

    pub fn canonical_name(self) -> &'static str {
        match self.0 {
            0 => "C",
            1 => "C#",
            2 => "D",
            3 => "D#",
            4 => "E",
            5 => "F",
            6 => "F#",
            7 => "G",
            8 => "G#",
            9 => "A",
            10 => "A#",
            11 => "B",
            _ => unreachable!("pitch class must be 0..12"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Key(PitchClass);

impl Key {
    pub const C: Self = Self(PitchClass::C);

    pub fn new(tonic: PitchClass) -> Self {
        Self(tonic)
    }

    pub fn tonic(self) -> PitchClass {
        self.0
    }

    pub fn parse(input: &str) -> Result<Self, ParseError> {
        let (pitch_class, rest) = parse_note_root(input.trim())?;
        if !rest.is_empty() {
            return Err(ParseError::new(format!(
                "unexpected trailing characters in key '{}'",
                input
            )));
        }
        Ok(Self::new(pitch_class))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Quality {
    Major,
    Minor,
    Major7,
    Minor7,
    Dominant7,
    Diminished,
    Augmented,
    Sus2,
    Sus4,
}

impl Quality {
    pub fn suffix(self) -> &'static str {
        match self {
            Self::Major => "maj",
            Self::Minor => "min",
            Self::Major7 => "maj7",
            Self::Minor7 => "min7",
            Self::Dominant7 => "7",
            Self::Diminished => "dim",
            Self::Augmented => "aug",
            Self::Sus2 => "sus2",
            Self::Sus4 => "sus4",
        }
    }

    pub fn intervals(self) -> &'static [u8] {
        match self {
            Self::Major => &[0, 4, 7, 12],
            Self::Minor => &[0, 3, 7, 12],
            Self::Major7 => &[0, 4, 7, 11, 12],
            Self::Minor7 => &[0, 3, 7, 10, 12],
            Self::Dominant7 => &[0, 4, 7, 10, 12],
            Self::Diminished => &[0, 3, 6, 12],
            Self::Augmented => &[0, 4, 8, 12],
            Self::Sus2 => &[0, 2, 7, 12],
            Self::Sus4 => &[0, 5, 7, 12],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Voicing {
    Close,
    Piano,
    Guitar,
    Open,
}

impl Voicing {
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        match input {
            "close" => Ok(Self::Close),
            "piano" => Ok(Self::Piano),
            "guitar" => Ok(Self::Guitar),
            "open" => Ok(Self::Open),
            other => Err(ParseError::new(format!("unsupported voicing '{}'", other))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Close => "close",
            Self::Piano => "piano",
            Self::Guitar => "guitar",
            Self::Open => "open",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Chord {
    pub root: PitchClass,
    pub quality: Quality,
}

impl Chord {
    pub fn voice(&self, voicing: Voicing) -> Vec<u8> {
        match voicing {
            Voicing::Close => self.close_voice(),
            Voicing::Piano => self.piano_voice(),
            Voicing::Guitar => self.guitar_voice(),
            Voicing::Open => self.open_voice(),
        }
    }

    fn close_voice(&self) -> Vec<u8> {
        let root_midi = centered_root_midi(self.root);
        self.quality
            .intervals()
            .iter()
            .map(|interval| root_midi + interval)
            .collect()
    }

    fn piano_voice(&self) -> Vec<u8> {
        let mut notes = Vec::with_capacity(self.quality.intervals().len() + 1);
        notes.push(bass_root_midi(self.root));
        notes.extend(self.close_voice());
        notes
    }

    fn guitar_voice(&self) -> Vec<u8> {
        const OPEN_STRINGS: [u8; 6] = [40, 45, 50, 55, 59, 64];

        OPEN_STRINGS
            .iter()
            .map(|open_string| self.lowest_chord_tone_at_or_above(*open_string))
            .collect()
    }

    fn open_voice(&self) -> Vec<u8> {
        let bass_root = bass_root_midi(self.root);
        let (third, fifth, seventh) = self.structure_intervals();
        let mut notes = vec![
            bass_root,
            bass_root + fifth,
            bass_root + third + 12,
            bass_root + fifth + 12,
        ];
        if let Some(seventh) = seventh {
            notes.push(bass_root + seventh + 12);
        }
        notes.push(bass_root + 24);
        notes.push(bass_root + third + 24);
        notes
    }

    fn structure_intervals(&self) -> (u8, u8, Option<u8>) {
        let intervals = self.quality.intervals();
        let third = intervals.get(1).copied().unwrap_or(4);
        let fifth = intervals.get(2).copied().unwrap_or(7);
        let seventh = if intervals.len() >= 5 {
            intervals.get(3).copied()
        } else {
            None
        };
        (third, fifth, seventh)
    }

    fn chord_tone_intervals(&self) -> Vec<u8> {
        let mut intervals = Vec::new();
        for interval in self.quality.intervals() {
            let normalized = interval % 12;
            if !intervals.contains(&normalized) {
                intervals.push(normalized);
            }
        }
        intervals
    }

    fn lowest_chord_tone_at_or_above(&self, floor: u8) -> u8 {
        let chord_tones = self.chord_tone_intervals();
        for note in floor..=127 {
            let relative_pc = (12 + note % 12 - self.root.as_u8()) % 12;
            if chord_tones.contains(&relative_pc) {
                return note;
            }
        }
        unreachable!("a chord tone must exist within one octave above any guitar string");
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseError {
    message: String,
}

impl ParseError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for ParseError {}

pub fn parse_chord(input: &str) -> Result<Chord, ParseError> {
    let token = input.trim();
    if token.is_empty() {
        return Err(ParseError::new("chord token is empty"));
    }
    if looks_like_roman_token(token) {
        return Err(ParseError::new(format!(
            "roman numeral '{}' requires parse_roman() or parse_progression_with_key()",
            token
        )));
    }
    let (root, rest) = parse_note_root(token)?;
    let quality = parse_named_quality(rest)?;
    Ok(Chord { root, quality })
}

pub fn parse_roman(input: &str, key: Key) -> Result<Chord, ParseError> {
    let token = input.trim();
    if token.is_empty() {
        return Err(ParseError::new("roman numeral token is empty"));
    }

    let roman_len = token
        .char_indices()
        .take_while(|(_, ch)| matches!(*ch, 'I' | 'V' | 'i' | 'v'))
        .last()
        .map(|(idx, ch)| idx + ch.len_utf8())
        .ok_or_else(|| ParseError::new(format!("invalid roman numeral '{}'", token)))?;

    let roman_part = &token[..roman_len];
    let degree = match roman_part.to_ascii_lowercase().as_str() {
        "i" => 0,
        "ii" => 2,
        "iii" => 4,
        "iv" => 5,
        "v" => 7,
        "vi" => 9,
        "vii" => 11,
        _ => {
            return Err(ParseError::new(format!(
                "unsupported roman numeral '{}'",
                roman_part
            )));
        }
    };

    let mut rest = &token[roman_len..];
    let mut default_quality = if roman_part.chars().all(|ch| ch.is_ascii_uppercase()) {
        Quality::Major
    } else if roman_part.chars().all(|ch| ch.is_ascii_lowercase()) {
        Quality::Minor
    } else {
        return Err(ParseError::new(format!(
            "roman numeral '{}' must use consistent case",
            roman_part
        )));
    };

    if let Some(stripped) = rest.strip_prefix('\u{00B0}') {
        default_quality = Quality::Diminished;
        rest = stripped;
    }

    let quality = parse_roman_quality(rest, default_quality)?;
    Ok(Chord {
        root: PitchClass::new(key.tonic().as_u8() + degree),
        quality,
    })
}

pub fn parse_progression(input: &str) -> Result<Vec<Chord>, ParseError> {
    parse_progression_with_key(input, None)
}

pub fn parse_progression_with_key(input: &str, key: Option<Key>) -> Result<Vec<Chord>, ParseError> {
    let tokens = split_progression_tokens(input);
    if tokens.is_empty() {
        return Err(ParseError::new("progression is empty"));
    }

    let mut chords = Vec::with_capacity(tokens.len());
    for token in tokens {
        if looks_like_roman_token(token) {
            let Some(key) = key else {
                return Err(ParseError::new(format!(
                    "roman numeral '{}' requires --key",
                    token
                )));
            };
            chords.push(parse_roman(token, key)?);
        } else {
            chords.push(parse_chord(token)?);
        }
    }
    Ok(chords)
}

fn split_progression_tokens(input: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut start: Option<usize> = None;

    for (idx, ch) in input.char_indices() {
        let is_separator = ch == '-' || ch == '|' || ch.is_whitespace();
        if is_separator {
            if let Some(token_start) = start.take() {
                if token_start < idx {
                    let token = input[token_start..idx].trim();
                    if !token.is_empty() {
                        tokens.push(token);
                    }
                }
            }
        } else if start.is_none() {
            start = Some(idx);
        }
    }

    if let Some(token_start) = start {
        let token = input[token_start..].trim();
        if !token.is_empty() {
            tokens.push(token);
        }
    }

    tokens
}

fn parse_note_root(input: &str) -> Result<(PitchClass, &str), ParseError> {
    let mut chars = input.char_indices();
    let (_, first) = chars
        .next()
        .ok_or_else(|| ParseError::new("note token is empty"))?;

    let base = match first.to_ascii_uppercase() {
        'A' => 9_i16,
        'B' => 11,
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        other => {
            return Err(ParseError::new(format!("invalid note root '{}'", other)));
        }
    };

    let mut semitone = base;
    let mut consumed = first.len_utf8();
    if let Some((idx, accidental)) = chars.next() {
        match accidental {
            '#' => {
                semitone += 1;
                consumed = idx + accidental.len_utf8();
            }
            'b' => {
                semitone -= 1;
                consumed = idx + accidental.len_utf8();
            }
            _ => {}
        }
    }

    let remainder = &input[consumed..];
    let normalized = semitone.rem_euclid(12) as u8;
    Ok((PitchClass::new(normalized), remainder))
}

fn parse_named_quality(input: &str) -> Result<Quality, ParseError> {
    match input {
        "" | "maj" => Ok(Quality::Major),
        "m" | "min" => Ok(Quality::Minor),
        "maj7" => Ok(Quality::Major7),
        "m7" | "min7" => Ok(Quality::Minor7),
        "7" => Ok(Quality::Dominant7),
        "dim" => Ok(Quality::Diminished),
        "aug" => Ok(Quality::Augmented),
        "sus2" => Ok(Quality::Sus2),
        "sus4" => Ok(Quality::Sus4),
        other => Err(ParseError::new(format!(
            "unsupported chord quality '{}'",
            other
        ))),
    }
}

fn parse_roman_quality(input: &str, default_quality: Quality) -> Result<Quality, ParseError> {
    if input.is_empty() {
        return Ok(default_quality);
    }

    match input {
        "7" => match default_quality {
            Quality::Major => Ok(Quality::Dominant7),
            Quality::Minor => Ok(Quality::Minor7),
            Quality::Diminished => Err(ParseError::new(
                "v1 roman numerals do not support diminished seventh shorthand",
            )),
            _ => parse_named_quality(input),
        },
        other => parse_named_quality(other),
    }
}

fn looks_like_roman_token(token: &str) -> bool {
    token
        .chars()
        .next()
        .map(|ch| matches!(ch, 'I' | 'V' | 'i' | 'v'))
        .unwrap_or(false)
}

fn centered_root_midi(root: PitchClass) -> u8 {
    let pc = root.as_u8();
    if pc <= 5 {
        60 + pc
    } else {
        48 + pc
    }
}

fn bass_root_midi(root: PitchClass) -> u8 {
    36 + root.as_u8()
}
