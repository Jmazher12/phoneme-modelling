# phoneme-modelling

Generates synthetic speech sounds, distorts them into graded pronunciation errors, and evaluates the results.

```bash
make install-dev   # install everything, including test tools
make generate      # generate clean clips, then distort them
make eval          # score the clips
make run           # both
```

Settings live in `config/default.json`.

## Distortions

A distortion edits a clean recording of a correct sound until it sounds like a common mistake, such as "r" said as
"w". Each sound pair is one class in `src/phoneme_modelling/distortion/distorters/`, because each pair needs its own
edit. `PairDistorter` does what they all share: read the clip, find the sound, apply the pair's edit at the requested
strength, and save a 16 kHz clip of at least 2.5 seconds.

Every pair has two versions:

- **Plain** (`r_to_w`): the distortion alone.
- **Matched** (`r_to_w_matched`): the same distortion, with the length, pitch and loudness of a correct clip of the
  same sound, so a model can't use those to tell errors from correct clips.

Each distortion is made at two strengths, set by `distortion_application_config.distortion_intensitites`:

- **0.6** goes about 60% of the way from the correct sound to the mistake.
- **1** goes all the way.

Any other strength is skipped. There is no strength 0: an unedited copy would only duplicate the clean clip, and the
clean clips are already the "correct" examples.

The numbers each pair uses are saved in `config/distortion/`, one file per pair. They were measured on the 2026-10-25
clean clips by `scripts/export_pair_parameters.py` in `phoneme-endpoint`.

### Pairs

There are 57 pairs, so 114 distorters. They are grouped below by the edit, which is a function in
`distortion/recipes.py`.

| Edit | Pairs |
|---|---|
| Move formants (`shift_formants`) | r→w, r→l, r→y, distorted r (`r_distortion`, part way to w), l→w, l→y, y→w, w→r |
| Reshape the hiss and add voicing (`voice_fricative`) | s→z, s→voiced th, f→v |
| Remove voicing and reshape the hiss (`devoice_fricative`) | z→voiceless th, z→s, voiced th→voiceless th, v→f |
| Reshape the hiss only (`reshape_fricative`) | s→voiceless th, z→voiced th, voiced th→v, voiceless th→f, voiceless th→s, sh→s |
| Turn a hiss into a stop or ch (`fricative_to_stop`) | s→d, s→t, z→d, z→t, voiced th→d, voiceless th→t, sh→t, sh→ch, v→b, f→p, f→b |
| Turn ch or j into a hiss (`affricate_to_fricative`) | ch→sh, j→z, j→sh |
| Turn ch or j into a stop (`affricate_to_stop`) | ch→t, ch→k, j→d |
| Change a stop's voicing or place (`change_stop`) | k→t, k→g, g→d, g→k, t→k, t→d, t→g, d→t, d→k, d→g, p→b, b→p |
| Reshape the nasal hum (`change_nasal_place`) | ng→n, n→m, m→n |
| Turn a nasal into a stop (`nasal_to_stop`) | ng→g, n→t, n→d, m→b |

Each pair's docstring says what its edit does. It also flags weak pairs: those the 10-25 scorecard did not rate
"works", and those measured on fewer than 3 voices.

The endpoint lists 8 more mistakes that have no distorter: r→h, r→d, r→n, r→z, g→w, g→y, t→voiceless th and m→w. The
two sounds in each differ too much for an edit to move one toward the other. On 2026-10-25 they were covered only by
real recordings of the other sound, and the clean clips here already provide those.

### Adding a pair

1. Make sure `scripts/export_pair_parameters.py` exports the pair (it exports every pair with a graded edit), and run
   it to create the pair's settings file.
2. Add `distortion/distorters/<from>_to_<to>.py` with a plain and a matched class. `_apply` is the pair's edit, usually
   one call to a function in `distortion/recipes.py`.
3. Add both classes to `PAIR_DISTORTERS` in `distortion/pair_registry.py`, and add their names to `distorter_configs`
   in the config.

## Tests

```bash
cd src && uv run pytest ../tests/distortion
```
