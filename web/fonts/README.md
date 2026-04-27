# Web fonts

`BIZUDPGothic-Regular-subset.ttf` is a glyph-subset of [BIZ UDPGothic
Regular](https://github.com/googlefonts/morisawa-biz-ud-gothic) (SIL
Open Font License) that contains only the Japanese characters used by
the wasm32 demo's UI strings (see `src/bin/web.rs::imp` — splash gate
hint, MIDI retry hint, panic-button caption).

Bundling this subset (~35 KB) instead of the full 4.5 MB face keeps
the wasm payload tight while still rendering Japanese without
missing-glyph boxes. Trade-off: any new Japanese characters added to
the UI need the subset regenerated.

## Regenerating after adding new Japanese strings

```sh
# 1. Grab the upstream Regular face.
curl -fL -o /tmp/BIZUDPGothic-Regular.ttf \
  https://github.com/googlefonts/morisawa-biz-ud-gothic/raw/main/fonts/ttf/BIZUDPGothic-Regular.ttf

# 2. Collect every UI string into a text file. Paste / extract by
# hand, or `grep -oP` the `web.rs` source.
cat <<'TXT' > /tmp/ui-chars.txt
…paste every JP string used in src/bin/web.rs here…
TXT

# 3. Subset (fonttools): strip everything except the chars we use.
pyftsubset /tmp/BIZUDPGothic-Regular.ttf \
    --output-file=web/fonts/BIZUDPGothic-Regular-subset.ttf \
    --text-file=/tmp/ui-chars.txt \
    --layout-features='*' \
    --glyph-names \
    --no-recommended-glyphs \
    --no-hinting

# 4. Verify size — should land in the 30-100 KB range.
ls -la web/fonts/BIZUDPGothic-Regular-subset.ttf
```

`pyftsubset` ships with `fonttools` (`pip install fonttools brotli`).
