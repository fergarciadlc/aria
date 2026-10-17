# Dataset creation scripts

Tools that build an ARIA dataset from a multitrack corpus. They are thin CLIs over the `aria` library, which defines the layout, naming, stem order and audio spec (`aria/constants.py`).

Only step 1 is corpus-specific: `prepare_moisesdb.py` is the adapter for [MoisesDB](https://github.com/moises-ai/moises-db), the reference corpus. For another corpus, write each track as `{id}/metadata.json` (with a `stems` list) and `{id}/stems/real/{stem}.wav`; steps 2–4 then work unchanged.

```bash
pip install -r requirements.txt   # library with all extras + moises-db
export ARIA_ROOT=/path/to/aria    # every script also accepts --aria-path
```

MP3 output needs `ffmpeg` on `PATH`.

## Pipeline

| Step | Script | Writes |
|------|--------|--------|
| 1 | `prepare_moisesdb.py` | `{id}/metadata.json`, `{id}/stems/real/{stem}.wav` |
| 2 | `generate_ai_stems.py` | `{id}/stems/ai_generated/encodec_{3,6,24}/{stem}.mp3` |
| 3 | `generate_mixes.py` | `{id}/mixes/{codec}/U{k}/mix_S-*.mp3` + `.json` sidecar |
| 4 | `create_splits.py` | `splits/{name}.jsonl` + `{name}.meta.json` |
| – | `create_metadata_csv.py` | one-row-per-track CSV |

```bash
python scripts/prepare_moisesdb.py --moisesdb-path /path/to/moisesdb   # or $MOISESDB_ROOT
python scripts/generate_ai_stems.py
python scripts/generate_mixes.py --codec encodec_3 --workers 8
python scripts/create_splits.py --aria-path $ARIA_ROOT --output $ARIA_ROOT/splits/canonical_v1.jsonl
```

Every script takes `--limit N` (first N tracks) for quick test runs, and skips outputs that already exist unless told otherwise (`--overwrite` / `--force`). Run any script with `--help` for all options.

### 1. `prepare_moisesdb.py`

Exports each MoisesDB track's stems at 44.1 kHz (`--sample-rate`) as WAV (or `--format mp3`) and writes `metadata.json` with a new UUID.

### 2. `generate_ai_stems.py`

Runs every real stem through the 48 kHz Encodec model (`aria.generators.EncodecGenerator`) once and decodes it at each bandwidth (`--bandwidths`, default `3 6 24`). Output is 256 kbps MP3 at 44.1 kHz.

- `--mode replicate` (default) feeds the 44.1 kHz audio to the 48 kHz model **without resampling**. This is how the released stems were made; it shifts Encodec's artifact grid by 44100/48000 (see `notebooks/fakeprint_encodec_3kbps.ipynb`). `--mode corrected` resamples to 48 kHz and back.
- Uses CUDA when available, otherwise CPU (MPS is skipped: Encodec quantization misbehaves there). On CPU, `--workers N` runs N processes with one model each.
- `--min-duration` skips short stems, `--max-duration` truncates long ones.

### 3. `generate_mixes.py`

Builds all 2ⁿ real/AI combinations for one codec (`--codec`) with `aria.mix.compute_mix`: stems are loaded as stereo, resampled to 44.1 kHz, zero-padded to the longest and summed. Each mix gets a `.json` sidecar with its stem names and AI indices.

### 4. `create_splits.py`

Assigns whole tracks to train/val/test (default 80/10/10, stratified by stem count, `--seed`) and writes one record per pre-computed mix. If `{ARIA_ROOT}/metrics/mix_metrics.csv` exists (or `--metrics-csv` is given), records also carry `ai_energy_ratio`. The record schema is documented in `aria/splits.py`.

- Tracks are fed to the splitter in sorted track-id order, so the same corpus and seed give the same split on any filesystem.
- `--split-from PATH` copies the track to split mapping of an existing manifest (per-mix or window) instead of splitting. It fails if a track is in two splits, or if a track on disk is absent from the reference (`--allow-unassigned` leaves such tracks out instead). The output meta file records `split_from` and sets `seed` to null.

#### Window manifests

When `--metrics-csv` points to a windowed `mix_metrics.csv` (a `window_start_sec` column and several rows per mix), `create_splits.py` writes a **window manifest**: one record per (mix, window) instead of one per mix. This is the format the window-level detector trained on (5 s and 1 s windows). The CSV is the output of a windowed metrics run; any other CSV gives the usual per-mix manifest.

Do not confuse it with the per-mix record schema "v3" in `aria/splits.py` (`ai_stem_ratio`, `duration`) or with a file named `canonical_v3.jsonl`. A file is a window manifest when its first row has `window_start_sec`.

```json
{"track_id": "...", "artist": "...", "track_name": "...", "genre": "...", "codec": "encodec_3",
 "num_stems": 8, "num_ai_stems": 1, "ai_indices": [3], "split": "train", "mix_path": "...",
 "window_index": 0, "window_start_sec": 0.0, "window_end_sec": 5.0,
 "ai_energy_ratio": 0.272252, "is_partial": false}
```

- `window_index` counts a mix's windows in start order. Times are rounded to 3 decimals, `ai_energy_ratio` to 6.
- `ai_energy_ratio` is never null. An empty cell in the CSV (a silent window, where the ratio is NaN) is labelled 0.0, and the count is logged.
- `is_partial` comes from the CSV column if present (otherwise false). Trailing windows shorter than the window size are the partial ones.
- The meta file has `"version": "3.0"`, `is_windowed: true`, `window_sec` (the most common non-partial window length) and `num_partial_windows` per split.
- Window records keep the field set of the original training manifests, so they have no `ai_stem_ratio` or `duration`.

```bash
python scripts/create_splits.py --aria-path $ARIA_ROOT \
    --metrics-csv $ARIA_ROOT/metrics/5s/mix_metrics.csv \
    --split-from splits/canonical_v2.jsonl \
    --output $ARIA_ROOT/splits/windows_5s.jsonl
```

#### The paper split

The split used in the paper cannot be regenerated from the audio alone. It was drawn from an unsorted directory listing on the training machine, and the stratified shuffle depends on the order of its input. Reuse it with `--split-from canonical_v2.jsonl` (the per-mix manifest the paper used). On the 240 paper tracks this reproduces `canonical_v2.jsonl` record for record, apart from the `ai_stem_ratio` and `duration` fields added by the current schema.

## Conventions

- **Stem order.** Stem index *i* is the *i*-th stem name in alphabetical order. `mix_S-0-2` means stems 0 and 2 are AI-generated; `mix_S-empty` is all real.
- **U-folders.** `U{k}` holds the C(n, k) mixes with exactly k AI stems.
- **Audio.** 44.1 kHz stereo; AI stems and mixes are 256 kbps CBR MP3.

## Packaging a release

`aria package` copies a subset of the dataset, optionally converting audio and including mixes, and writes `package_info.json`:

```bash
aria tiers                                                       # tiny 10, small 50, medium 100, full
aria package -s $ARIA_ROOT -o releases/aria-tiny --tier tiny     # FLAC stems
aria package -s $ARIA_ROOT -o releases/aria-small -n 50 -f mp3 -c encodec_3 --include-mixes
aria validate releases/aria-tiny
```

- Tracks are the first N in `Artist - Song` order, so tiers are nested and reproducible.
- `--include-mixes` copies pre-computed mixes and computes any that are missing.
- `-f wav|flac|mp3` converts stems and mixes; files already in that format are copied as-is.
- `--tier` overrides `-n/-f/-m/-c`.

The same is available from Python as `aria.package_dataset(source, output, tier=... | config=PackageConfig(...))`.
