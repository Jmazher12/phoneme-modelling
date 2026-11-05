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
- **Matched** (`r_to_w_matched`): the same distortion, with the length, pitch and loudness of a clip of the other
  label, so a model can't use those to guess the answer. It only runs at strengths 0, 0.6 and 1.

Strengths come from `distortion_application_config.distortion_intensitites`. Strength 0 leaves the sound unedited but
still puts it in the standard clip format, which gives a fair "correct" clip to compare against. Strength 0.2 reaches
about a fifth of the natural difference between the two sounds, and 1 reaches all of it.

The numbers each pair uses are saved in `config/distortion/`. They were measured on the 2026-10-25 clean clips by
`scripts/export_pair_parameters.py` in `phoneme-endpoint`.

### Adding a pair

1. Add the pair to `PAIRS_TO_EXPORT` in `scripts/export_pair_parameters.py` and run it to create its settings file.
2. Add `distortion/distorters/<from>_to_<to>.py` with a plain and a matched class. `_apply` is the pair's edit, usually
   one call to a function in `distortion/recipes.py`.
3. Register both classes in `distortion/pipeline.py`, and add their names to `distorter_configs` in the config.

## Tests

```bash
cd src && uv run pytest ../tests/distortion
```
