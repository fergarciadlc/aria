# Evaluation scripts

Run existing AI-music detectors on ARIA mixes.

## `aria_deepfake_inference.py` — Deezer detector

Runs the Deezer deepfake detector on every pre-computed mix (`mixes/{codec}/U{k}/mix_S-*`, any audio format) and writes one JSON line per mix.

**Needs:** a checkout of the Deezer deepfake-detector repository (model code, config and weights), TensorFlow 2.x, `soxr`, `torchaudio`, `loguru`, and mixes from `scripts/generate_mixes.py`.

```bash
export ARIA_ROOT=/path/to/aria                         # or --aria-path
export DEEZER_DETECTOR_PATH=/path/to/deepfake-detector # or --detector-path

python -m evaluation.aria_deepfake_inference --limit 5 --output test_run.jsonl   # first 5 tracks
python -m evaluation.aria_deepfake_inference --codec encodec_3 --gpu 0           # one codec on GPU 0
```

Other options: `--config` (default `interpretable`), `--weights` (default: the config's pre-trained weights), `--output` (default `aria_inference[_codec][_limitN]_{timestamp}.jsonl`). Results are written and flushed per mix.

Each line holds the mix (`file`, `path`, `track_id`, `ai_codec`), its ground truth (`num_stems`, `stem_names`, `num_ai_stems`, `ai_stem_indices`) and the detector output over up to five slices (`real_probability` mean/std/min/max/per-slice, `deepfake_probability`, `prediction`, `confidence`, plus `encoder_distribution` and `most_likely_encoder` for the multi-task model). Failed mixes have `"prediction": "error"` and an `error` message.

> `stem_names` is in index order (alphabetical): every value in `ai_stem_indices` is a position in `stem_names`. Result files produced before the library consolidation (PR #1) listed `stem_names` in metadata order instead; don't map their indices to names.

## `spectttra_inference.py` — SONICS SpecTTTra

Generic (not ARIA-aware) inference with the [SONICS](https://github.com/awsaf49/sonics) SpecTTTra models on a file or a directory. Needs `librosa` and the `sonics` package (install it from the SONICS repository).

```bash
python evaluation/spectttra_inference.py -i /path/to/audio_or_dir -m gamma-5s -o results.csv
```

Options: `-m` model (`alpha|beta|gamma` × `5s|120s`), `-d auto|cpu|cuda`, `-s middle|first|average` chunk strategy, `-t` threshold, `-o` CSV output.

## `explore_results.ipynb`

Plots and tables over an `aria_deepfake_inference` JSONL file (update the results path where the notebook loads it). Needs `pandas`, `seaborn`, `joypy`.
