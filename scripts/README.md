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
