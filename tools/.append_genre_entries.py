"""Stage E genre-broadening — appends 40 cross-genre PD/CC0 entries to
bench-out/songs/manifest.json.

Standalone helper, intentionally not wired into ``run-collector.sh``:
the cross-genre URLs were curated by hand against
``Mutopia (Joplin / Traditional / SousaJP / FosterSC / Anonymous)`` and
``github.com/m-malandro/CC0-midis``, then verified with curl HEAD before
committing. Re-runs are idempotent — entries whose ``file`` already
appears in the manifest are skipped.

Run once::

    python tools/.append_genre_entries.py
    python tools/collector_validate.py   # confirm all entries pass

The leading dot keeps the file out of ``run-collector.sh``'s glob and
out of the daily cron, since this is a one-shot expansion not an
ongoing collector.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "bench-out" / "songs" / "manifest.json"


# Each tuple: (file, title, composer, era_override, license, source,
#              source_url, suggested_voice, context, tags)
# era_override == None means "let derive_era classify from composer
# year-range"; an explicit string overrides (Renaissance / Game).

JOPLIN_SOURCE = "Mutopia Project"

RAGTIME = [
    (
        "joplin_entertainer.mid",
        "S. Joplin — The Entertainer (1902)",
        "Scott Joplin (1868-1917)",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/JoplinS/entertainer/entertainer.mid",
        "piano-modal",
        "Canonical American ragtime — syncopated right-hand melody over steady stride bass. Stage E genre breadth: pulls jukebox out of pure-classical bias.",
        ["ragtime", "piano", "american", "license:public-domain"],
    ),
    (
        "joplin_elite_syncopations.mid",
        "S. Joplin — Elite Syncopations (1902)",
        "Scott Joplin (1868-1917)",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/JoplinS/EliteSyncopations/EliteSyncopations.mid",
        "piano-modal",
        "Mid-tempo Joplin ragtime — alternating bass with sixteenth-note cross-rhythms in the right hand.",
        ["ragtime", "piano", "license:public-domain"],
    ),
    (
        "joplin_pineapple_rag.mid",
        "S. Joplin — Pineapple Rag (1908)",
        "Scott Joplin (1868-1917)",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/JoplinS/PineappleRag/PineappleRag.mid",
        "piano-modal",
        "Late-period Joplin — chromatic inner-voice motion stresses voice-leading clarity at piano-roll tempi.",
        ["ragtime", "piano", "license:public-domain"],
    ),
    (
        "joplin_something_doing.mid",
        "S. Joplin — Something Doing (1903)",
        "Scott Joplin (1868-1917)",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/JoplinS/SomethingDoing/SomethingDoing.mid",
        "piano-modal",
        "Co-written with Scott Hayden. Bouncier ragtime, lots of breaks — useful counterpoint to the more uniform Entertainer.",
        ["ragtime", "piano", "license:public-domain"],
    ),
    (
        "joplin_strenuous_life.mid",
        "S. Joplin — The Strenuous Life (1902)",
        "Scott Joplin (1868-1917)",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/JoplinS/TheStrenuousLife/TheStrenuousLife.mid",
        "piano-modal",
        "Roosevelt-era Joplin rag, named after T. Roosevelt's speech. Driving stride bass.",
        ["ragtime", "piano", "license:public-domain"],
    ),
    (
        "joplin_wall_street_rag.mid",
        "S. Joplin — Wall Street Rag (1909)",
        "Scott Joplin (1868-1917)",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/JoplinS/WallStreetRag/WallStreetRag.mid",
        "piano-modal",
        "Programmatic Joplin: each section has a written caption (panic, good times, etc.). Tests dynamic-range range-mapping.",
        ["ragtime", "piano", "programmatic", "license:public-domain"],
    ),
    (
        "joplin_breeze_alabama.mid",
        "S. Joplin — A Breeze From Alabama (1902)",
        "Scott Joplin (1868-1917)",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/JoplinS/a-breeze-from-alabama/a-breeze-from-alabama.mid",
        "piano-modal",
        "Brisk march-rag hybrid. Demonstrates Joplin's debt to Sousa-era march form before settling into rag syncopation.",
        ["ragtime", "march", "piano", "license:public-domain"],
    ),
    (
        "joplin_bethena.mid",
        "S. Joplin — Bethena: A Concert Waltz (1905)",
        "Scott Joplin (1868-1917)",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/JoplinS/bethena/bethena.mid",
        "piano-modal",
        "Joplin in 3/4 — slow concert waltz, not a rag. Shows the ragtime voice can sustain non-syncopated material too.",
        ["ragtime", "waltz", "piano", "license:public-domain"],
    ),
    (
        "joplin_eugenia.mid",
        "S. Joplin — Eugenia (1905)",
        "Scott Joplin (1868-1917)",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/JoplinS/eugenia/eugenia.mid",
        "piano-modal",
        "Mid-period Joplin rag, classic AABBACCDD form.",
        ["ragtime", "piano", "license:public-domain"],
    ),
    (
        "joplin_magnetic_rag.mid",
        "S. Joplin — Magnetic Rag (1914)",
        "Scott Joplin (1868-1917)",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/JoplinS/magnetic/magnetic.mid",
        "piano-modal",
        "Joplin's last published rag. Minor-key middle section is unusual for the genre — useful tonal stress test.",
        ["ragtime", "piano", "license:public-domain"],
    ),
]

FOLK = [
    (
        "trad_aamulla_varhain.mid",
        "Trad. (Finnish) — Aamulla varhain",
        "Traditional (Finnish) — arr. Tanja Kivi",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/Traditional/AAMULLAVARHAIN/AAMULLAVARHAIN.mid",
        "piano-modal",
        "Finnish folk song with piano accompaniment. Adds a Nordic timbre into the catalog's geographic mix.",
        ["folk", "finnish", "nordic", "license:public-domain"],
    ),
    (
        "trad_banza_bushi.mid",
        "Trad. (Japanese) — Banza-Bushi",
        "Traditional (Japanese) — arr. Y. Nagai, K. Obata",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/Traditional/J95000-Banzai-Bushi/J95000-Banzai-Bushi.mid",
        "guitar-stk",
        "Edo-period min'yō melody transcribed for shamisen. Pentatonic, sparse texture — exposes voice attack envelope on isolated plucks.",
        ["folk", "japanese", "minyo", "shamisen", "license:public-domain"],
    ),
    (
        "trad_yanreso_hoi.mid",
        "Trad. (Japanese) — Yanreso Hoi",
        "Traditional (Japanese) — arr. Y. Nagai, K. Obata",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/Traditional/J95000-YanresoHoi/J95000-YanresoHoi.mid",
        "guitar-stk",
        "Japanese work-song min'yō for shamisen. Modal — useful counterpoint to the diatonic-major bias of the European catalog.",
        ["folk", "japanese", "minyo", "shamisen", "license:public-domain"],
    ),
    (
        "trad_noue_bushi.mid",
        "Trad. (Japanese) — Noue Bushi",
        "Traditional (Japanese) — arr. Y. Nagai, K. Obata",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/Traditional/J95002-Noue-Bushi/J95002-Noue-Bushi.mid",
        "guitar-stk",
        "Short Japanese folk piece for shamisen, ~30 s. Compact regression candidate.",
        ["folk", "japanese", "minyo", "shamisen", "license:public-domain"],
    ),
    (
        "trad_umewa_saitaka.mid",
        "Trad. (Japanese) — Umewa Saitaka",
        "Traditional (Japanese) — arr. Y. Nagai, K. Obata",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/Traditional/J95002-UmewaSaitaka/J95002-UmewaSaitaka.mid",
        "guitar-stk",
        "Japanese plum-blossom theme, shamisen transcription. Cherry-blossom-season programming hook for jukebox.",
        ["folk", "japanese", "minyo", "shamisen", "license:public-domain"],
    ),
    (
        "trad_doujan.mid",
        "Trad. (Japanese) — Doujan",
        "Traditional (Japanese) — arr. Y. Nagai, K. Obata",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/Traditional/J95004-Doujan/J95004-Doujan.mid",
        "guitar-stk",
        "Short shamisen min'yō, in & yo modal mix. Geographically diverse from the Mutopia classical mainline.",
        ["folk", "japanese", "minyo", "shamisen", "license:public-domain"],
    ),
    (
        "trad_saakorasa.mid",
        "Trad. (Japanese) — Saakorasa",
        "Traditional (Japanese) — arr. Y. Nagai, K. Obata",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/Traditional/J95004-Saakorasa/J95004-Saakorasa.mid",
        "guitar-stk",
        "Japanese folk shamisen piece. Tiny SMF (<1 KiB) but musically intact — handy stress test for short-piece UI rendering.",
        ["folk", "japanese", "minyo", "shamisen", "license:public-domain"],
    ),
    (
        "trad_greensleeves_accordion.mid",
        "Trad. (English) — Greensleeves (accordion arr.)",
        "Traditional (English) — arr. Mutopia Project",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/Traditional/GreensleevesAcc/GreensleevesAcc.mid",
        "piano-modal",
        "Accordion arrangement of the canonical English Renaissance folk song. Companion to the existing lute Greensleaves.mid.",
        ["folk", "english", "renaissance", "license:public-domain"],
    ),
]

MARCH = [
    (
        "sousa_el_capitan.mid",
        "J. P. Sousa — El Capitán March (1896)",
        "John Philip Sousa (1854-1932)",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/SousaJP/ElCapitanMarch/ElCapitanMarch.mid",
        "piano-modal",
        "Sousa march drawn from his comic opera. Bright, brass-band character — adds American military-march era to the catalog.",
        ["march", "american", "sousa", "license:public-domain"],
    ),
    (
        "sousa_hail_liberty.mid",
        "J. P. Sousa — Hail to the Spirit of Liberty (1900)",
        "John Philip Sousa (1854-1932)",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/SousaJP/HailtotheSpiritofLiberty/HailtotheSpiritofLiberty.mid",
        "piano-modal",
        "Sousa march. Uniform marching-band tempo — useful steady-state regression sample.",
        ["march", "american", "sousa", "license:public-domain"],
    ),
    (
        "sousa_hands_across_sea.mid",
        "J. P. Sousa — Hands Across the Sea (1899)",
        "John Philip Sousa (1854-1932)",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/SousaJP/HandsacrosstheSea/HandsacrosstheSea.mid",
        "piano-modal",
        "Sousa march dedicated to international friendship. Strong dynamic contrasts in the trio section.",
        ["march", "american", "sousa", "license:public-domain"],
    ),
    (
        "sousa_king_cotton.mid",
        "J. P. Sousa — King Cotton March (1895)",
        "John Philip Sousa (1854-1932)",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/SousaJP/KingCotton/KingCotton.mid",
        "piano-modal",
        "Atlanta Cotton States Exposition march. Long form ABAC + trio — exercises voice across many sectional repeats.",
        ["march", "american", "sousa", "license:public-domain"],
    ),
    (
        "sousa_liberty_bell.mid",
        "J. P. Sousa — The Liberty Bell March (1893)",
        "John Philip Sousa (1854-1932)",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/SousaJP/TheLibertyBell/TheLibertyBell.mid",
        "piano-modal",
        "Famous internationally as the Monty Python theme. Bell-imitation chimes in the trio — useful percussion-adjacent texture.",
        ["march", "american", "sousa", "license:public-domain"],
    ),
    (
        "sousa_stars_stripes.mid",
        "J. P. Sousa — The Stars and Stripes Forever (1896)",
        "John Philip Sousa (1854-1932)",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/SousaJP/TheStarsAndStripesForever/TheStarsAndStripesForever.mid",
        "piano-modal",
        "Official US national march. Iconic trio piccolo countermelody — multi-voice density stress.",
        ["march", "american", "sousa", "patriotic", "license:public-domain"],
    ),
]

# Foster died 1864 → derive_era → Romantic. Stays as a single Foster
# entry — the genre breadth is "American parlor song" which neither
# Mutopia trad nor Sousa marches cover.
AMERICAN_SONG = [
    (
        "foster_slumber.mid",
        "S. C. Foster — Slumber, My Darling (1862)",
        "Stephen Collins Foster (1826-1864)",
        None,
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/FosterSC/slumber/slumber.mid",
        "piano-modal",
        "American 19th-century parlor lullaby — the genre that dominated US sheet-music sales pre-ragtime. Foster is the canonical name; predates Joplin by a generation.",
        ["american-song", "lullaby", "parlor", "license:public-domain"],
    ),
]

# Anonymous Renaissance/Baroque hymns + dances. derive_era can't read
# years out of "Anonymous", so each carries an explicit era override.
ANONYMOUS = [
    (
        "anon_polish_xmas_carol.mid",
        "Anon. — Nuż my dziś krześcijani (Polish Christmas Carol)",
        "Anonymous (Polish, 16th c.)",
        "Renaissance",
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/Anonymous/Nuz_my_dzis_krzescijani/Nuz_my_dzis_krzescijani.mid",
        "piano-modal",
        "Polish Renaissance Christmas carol. Era override = Renaissance because composer string lacks a year range derive_era can parse.",
        ["renaissance", "polish", "christmas", "carol", "license:public-domain"],
    ),
    (
        "anon_old100th_orig.mid",
        "Anon. — Old 100th (original 1561 version)",
        "Anonymous (Genevan Psalter, 1551)",
        "Renaissance",
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/Anonymous/Old100-orig/Old100-orig.mid",
        "piano-modal",
        "Genevan Psalter hymn tune that became 'All People That on Earth Do Dwell' / 'Doxology'. The original syncopated rhythm before later regularization.",
        ["renaissance", "hymn", "psalter", "license:public-domain"],
    ),
    (
        "anon_bonne_heure.mid",
        "Anon. — Il Est de Bonne Heure Né",
        "Anonymous (French, Renaissance)",
        "Renaissance",
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/Anonymous/bonne_heure-song/bonne_heure-song.mid",
        "piano-modal",
        "Anonymous French Renaissance song. Modal melody — exercises voice in non-major-key repertoire.",
        ["renaissance", "french", "song", "license:public-domain"],
    ),
    (
        "anon_ich_ruf.mid",
        "Anon. — Ich ruf zu dir, Herr Jesus Christ",
        "Anonymous (German Lutheran, 16th c.)",
        "Renaissance",
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/Anonymous/Ich_ruf/Ich_ruf.mid",
        "piano-modal",
        "Lutheran chorale melody (later harmonized by Bach as BWV 639). Plain unharmonized version — useful as the source-tune A/B against Bach's chorale settings.",
        ["renaissance", "chorale", "lutheran", "license:public-domain"],
    ),
    (
        "anon_lobt_gott.mid",
        "Anon. — Lobt Gott, ihr Christen, allzugleich",
        "Anonymous (German Lutheran, 16th c.)",
        "Renaissance",
        "Public Domain",
        "https://www.mutopiaproject.org/ftp/Anonymous/Lobt_Gott/Lobt_Gott.mid",
        "piano-modal",
        "Lutheran chorale melody. Era override = Renaissance to keep it out of the Modern bucket.",
        ["renaissance", "chorale", "lutheran", "license:public-domain"],
    ),
]

# m-malandro/CC0-midis on GitHub. CC0 1.0 Universal — strictest possible
# permissive license. derive_era can't classify a GitHub handle, so
# each carries an explicit era="Game". This is the catalog's first
# explicitly-electronic / chiptune-adjacent material.
GAME_CC0_SOURCE = "github.com/m-malandro/CC0-midis"

GAME = [
    (
        "cc0_frantic_boss_battle.mid",
        "m-malandro — Frantic Boss Battle",
        "m-malandro (CC0)",
        "Game",
        "CC0",
        "https://raw.githubusercontent.com/m-malandro/CC0-midis/main/midis/frantic-boss-battle.mid",
        "synth-square",
        "CC0 indie-game-style boss battle theme — fast 16th-note runs across multiple instruments. Exercises voice polyphony at high event density.",
        ["game", "boss-battle", "electronic", "cc0", "license:cc0"],
    ),
    (
        "cc0_lighthearted_battle_theme.mid",
        "m-malandro — Lighthearted Battle Theme",
        "m-malandro (CC0)",
        "Game",
        "CC0",
        "https://raw.githubusercontent.com/m-malandro/CC0-midis/main/midis/lighthearted-battle-theme.mid",
        "synth-square",
        "Upbeat JRPG-style battle theme. Major-key, melodic — counterpoint to the frantic boss track.",
        ["game", "battle", "jrpg", "cc0", "license:cc0"],
    ),
    (
        "cc0_arena_rock.mid",
        "m-malandro — Arena Rock",
        "m-malandro (CC0)",
        "Game",
        "CC0",
        "https://raw.githubusercontent.com/m-malandro/CC0-midis/main/midis/arena-rock.mid",
        "synth-square",
        "CC0 driving rock-arena instrumental. Power-chord harmonic vocabulary — extends genre coverage beyond keyboard literature.",
        ["game", "rock", "electronic", "cc0", "license:cc0"],
    ),
    (
        "cc0_game_over.mid",
        "m-malandro — Game Over",
        "m-malandro (CC0)",
        "Game",
        "CC0",
        "https://raw.githubusercontent.com/m-malandro/CC0-midis/main/midis/game-over.mid",
        "synth-square",
        "Short 'game over' sting (<1 KiB SMF). Useful as a UX-style cue and as a smoke-test for very short pieces.",
        ["game", "sting", "sfx", "cc0", "license:cc0"],
    ),
    (
        "cc0_gather_your_party.mid",
        "m-malandro — Gather Your Party",
        "m-malandro (CC0)",
        "Game",
        "CC0",
        "https://raw.githubusercontent.com/m-malandro/CC0-midis/main/midis/gather-your-party.mid",
        "synth-square",
        "Adventure-game tavern / party-formation cue. Mid-tempo, layered — a different timbral target than the battle tracks.",
        ["game", "ambient", "adventure", "cc0", "license:cc0"],
    ),
    (
        "cc0_overture_2021.mid",
        "m-malandro — Overture 2021",
        "m-malandro (CC0)",
        "Game",
        "CC0",
        "https://raw.githubusercontent.com/m-malandro/CC0-midis/main/midis/overture-2021.mid",
        "synth-square",
        "Long-form (~44 KiB SMF) game overture. Tests sustained voice behavior over multi-minute playback.",
        ["game", "overture", "long-form", "cc0", "license:cc0"],
    ),
    (
        "cc0_sitar_jam.mid",
        "m-malandro — Sitar Jam",
        "m-malandro (CC0)",
        "Game",
        "CC0",
        "https://raw.githubusercontent.com/m-malandro/CC0-midis/main/midis/sitar-jam.mid",
        "synth-square",
        "CC0 indie game cue with sitar-flavored melodic line. Modal, pentatonic-leaning — non-Western tonal stress test.",
        ["game", "world", "modal", "cc0", "license:cc0"],
    ),
    (
        "cc0_math_metal.mid",
        "m-malandro — Math Metal",
        "m-malandro (CC0)",
        "Game",
        "CC0",
        "https://raw.githubusercontent.com/m-malandro/CC0-midis/main/midis/math-metal.mid",
        "synth-square",
        "Odd-meter math-metal-style instrumental. Irregular phrase lengths exercise the voice's transient response under unsteady accent patterns.",
        ["game", "metal", "odd-meter", "cc0", "license:cc0"],
    ),
    (
        "cc0_movin_on.mid",
        "m-malandro — Movin' On",
        "m-malandro (CC0)",
        "Game",
        "CC0",
        "https://raw.githubusercontent.com/m-malandro/CC0-midis/main/midis/movin-on.mid",
        "synth-square",
        "Mid-tempo CC0 instrumental — overworld-traversal vibe. Typical SNES-RPG-adjacent style.",
        ["game", "overworld", "rpg", "cc0", "license:cc0"],
    ),
    (
        "cc0_android_observation_room.mid",
        "m-malandro — Android Observation Room",
        "m-malandro (CC0)",
        "Game",
        "CC0",
        "https://raw.githubusercontent.com/m-malandro/CC0-midis/main/midis/android-observation-room.mid",
        "synth-square",
        "Sci-fi ambient cue. Sparse texture — exposes voice noise floor / sustain quality.",
        ["game", "ambient", "scifi", "cc0", "license:cc0"],
    ),
]


def to_entry(t: tuple) -> dict:
    file, title, composer, era, license_, url, voice, context, tags = t
    if file.startswith("cc0_"):
        source = GAME_CC0_SOURCE
        instrument = "synth"
    elif file.startswith("sousa_"):
        source = JOPLIN_SOURCE  # Mutopia
        instrument = "piano"
    elif file.startswith("foster_"):
        source = JOPLIN_SOURCE
        instrument = "piano"
    elif file.startswith("anon_"):
        source = JOPLIN_SOURCE
        instrument = "piano"
    elif file.startswith("trad_"):
        source = JOPLIN_SOURCE
        instrument = "guitar" if "shamisen" in tags or "shamisen" in title.lower() else "piano"
    elif file.startswith("joplin_"):
        source = JOPLIN_SOURCE
        instrument = "piano"
    else:
        source = JOPLIN_SOURCE
        instrument = "piano"

    entry: dict = {
        "file": file,
        "title": title,
        "composer": composer,
        "instrument": instrument,
        "license": license_,
        "source": source,
        "source_url": url,
        "suggested_voice": voice,
        "context": context,
        "tags": tags,
    }
    if era is not None:
        entry["era"] = era
    return entry


def main() -> int:
    with MANIFEST.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    existing_files = {e.get("file") for e in manifest.get("entries", [])}

    all_new = RAGTIME + FOLK + MARCH + AMERICAN_SONG + ANONYMOUS + GAME
    appended = 0
    skipped = 0
    for t in all_new:
        entry = to_entry(t)
        if entry["file"] in existing_files:
            skipped += 1
            continue
        manifest["entries"].append(entry)
        existing_files.add(entry["file"])
        appended += 1

    # Refresh the license_summary description to acknowledge cc0 source.
    if "license_summary" in manifest:
        ls = manifest["license_summary"]
        ls["all_entries"] = "Public Domain / CC-BY 4.0 / CC0 1.0 Universal"
        ls["verification"] = (
            "Mutopia Project entries verified via mutopiaproject.org catalog (Public Domain / CC-BY only — "
            "ShareAlike and NonCommercial rejected). CC0 entries from github.com/m-malandro/CC0-midis "
            "(LICENSE = CC0 1.0 Universal, verified 2026-04-29). All URLs HEAD-checked before download."
        )

    with MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"appended={appended} skipped={skipped} total_entries={len(manifest['entries'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
