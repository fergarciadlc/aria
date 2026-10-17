"""
Energy metrics: per-stem windowed RMS and the AI energy ratio of each mix.

The AI energy ratio of a mix (alpha in the paper) is the share of its power
that comes from AI stems:

    alpha = sum_{AI stems} RMS(s)^2 / sum_{all stems} RMS(s)^2

It is computed from stem RMS values, never from the rendered mix audio, so it
needs no mix decoding. This module ports the script used to build the
regression training data and is meant to reproduce its numbers and CSV
schema exactly, so every choice below (loader, window rule, column order)
follows that script.

Stage 1 (audio): ``discover_stem_jobs`` + ``compute_stem_metrics`` produce one
row per stem file, or one row per non-overlapping window of it.
Stage 2 (no audio): ``discover_mix_entries`` + ``compute_mix_metrics`` sum the
stem powers per window for each mix.

Audio is read with ``librosa.load(path, sr=None, mono=True)`` (extra
``aria-dataset[metrics]``). Another loader would change the decoded samples
of the MP3 stems and therefore alpha, so it is not swapped for the library's
own ``load_audio``.
"""

from __future__ import annotations

import csv
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from aria.constants import AUDIO_EXTENSIONS
from aria.exceptions import MissingStemError
from aria.mix import MixSpec, get_mix_dir
from aria.track import get_stem_path, iter_track_dirs, load_track

logger = logging.getLogger(__name__)

# Column order of the output CSVs. Identical to the original script's
# windowed output; downstream readers (aria.splits.load_energy_metrics and
# the window manifests) rely on these names.
STEM_COLUMNS = (
    "uuid",
    "stem_index",
    "stem_name",
    "variant",
    "codec",
    "file_path",
    "rms",
    "rms_db",
    "duration_sec",
    "window_start_sec",
    "window_end_sec",
    "is_partial",
    "error",
)
MIX_COLUMNS = (
    "audio_path",
    "uuid",
    "codec",
    "n_ai_stems",
    "n_total_stems",
    "window_start_sec",
    "window_end_sec",
    "ai_power",
    "real_power",
    "total_power",
    "ai_energy_ratio",
    "missing_stems",
)

# Added to rms before taking the log, so silence maps to -200 dB, not -inf.
_DB_EPS = 1e-10


# ---------------------------------------------------------------------------
# Stage 1: per-stem RMS
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StemJob:
    """One stem audio file to measure."""

    uuid: str
    stem_index: int  # position in the track's alphabetically sorted stem names
    stem_name: str
    variant: str  # "real" or "ai"
    codec: str  # "" for real stems
    file_path: str


def _rms_row(
    chunk: np.ndarray,
    duration_sec: float,
    start_sec: float,
    end_sec: float,
    is_partial: bool,
) -> dict[str, Any]:
    # Same expression and dtype (float32 from the loader) as the original.
    rms = float(np.sqrt(np.mean(chunk**2))) if chunk.size else 0.0
    return {
        "rms": rms,
        "rms_db": float(20 * np.log10(rms + _DB_EPS)),
        "duration_sec": duration_sec,
        "window_start_sec": start_sec,
        "window_end_sec": end_sec,
        "is_partial": is_partial,
    }


def windowed_rms(
    y: np.ndarray, sr: int, window_sec: float | None
) -> list[dict[str, Any]]:
    """RMS of a mono signal, for the whole file or per window.

    With ``window_sec=None`` the result is one row for the whole file. With a
    window, rows cover non-overlapping windows of ``round(window_sec * sr)``
    samples starting at sample 0. The trailing partial window is dropped. A
    signal shorter than one window gives a single row spanning its full
    duration, flagged ``is_partial=True``.

    Args:
        y: Mono samples.
        sr: Sample rate.
        window_sec: Window length in seconds, or None for no windowing.

    Returns:
        Non-empty list of dicts with keys rms, rms_db, duration_sec,
        window_start_sec, window_end_sec, is_partial.
    """
    duration_sec = float(len(y) / sr)

    if window_sec is None:
        return [_rms_row(y, duration_sec, 0.0, duration_sec, False)]

    samples_per_window = int(round(window_sec * sr))
    if samples_per_window < 1:
        raise ValueError(f"window_sec={window_sec} is shorter than one sample at {sr} Hz")
    n_windows = len(y) // samples_per_window

    if n_windows == 0:
        return [_rms_row(y, duration_sec, 0.0, duration_sec, True)]

    rows = []
    for w in range(n_windows):
        start = w * samples_per_window
        end = start + samples_per_window
        rows.append(
            _rms_row(y[start:end], duration_sec, float(start / sr), float(end / sr), False)
        )
    return rows


def _import_librosa():
    try:
        import librosa
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Energy metrics need librosa. Install it with: "
            'pip install "aria-dataset[metrics]"'
        ) from e
    return librosa


def compute_audio_metrics(
    file_path: str | Path, window_sec: float | None
) -> list[dict[str, Any]]:
    """Load an audio file and measure its RMS, optionally per window.

    Always returns a non-empty list. A file that cannot be read yields one
    row with NaN metrics and the failure message in ``error``; the other rows
    have ``error`` set to NaN.
    """
    librosa = _import_librosa()
    try:
        y, sr = librosa.load(str(file_path), sr=None, mono=True)
    except Exception as e:  # noqa: BLE001 - any failure is recorded, not raised
        return [
            {
                "rms": math.nan,
                "rms_db": math.nan,
                "duration_sec": math.nan,
                "window_start_sec": math.nan,
                "window_end_sec": math.nan,
                "is_partial": False,
                "error": str(e),
            }
        ]

    rows = windowed_rms(y, sr, window_sec)
    for row in rows:
        row["error"] = math.nan
    return rows


def discover_stem_jobs(dataset_path: Path | str, codecs: Sequence[str]) -> list[StemJob]:
    """List every real and AI stem file to measure.

    Stem names and indices come from ``load_track`` (alphabetical order), real
    stems from ``stems/real`` and AI stems from ``stems/ai_generated/{codec}``.
    Stems with no file on disk are skipped.
    """
    jobs: list[StemJob] = []
    for track_dir in iter_track_dirs(dataset_path):
        track = load_track(track_dir)
        for index, stem_name in enumerate(track.stem_names):
            try:
                real_path = get_stem_path(track, stem_name, "real")
            except MissingStemError:
                pass
            else:
                jobs.append(
                    StemJob(track_dir.name, index, stem_name, "real", "", str(real_path))
                )
            for codec in codecs:
                try:
                    ai_path = get_stem_path(track, stem_name, "ai_generated", codec)
                except MissingStemError:
                    continue
                jobs.append(
                    StemJob(track_dir.name, index, stem_name, "ai", codec, str(ai_path))
                )
    return jobs


# Worker state, set once per process so the window is not pickled per job.
_worker_window_sec: float | None = None


def _init_worker(window_sec: float | None) -> None:
    global _worker_window_sec
    _worker_window_sec = window_sec


def _process_stem_job(job: StemJob) -> list[dict[str, Any]]:
    base = {
        "uuid": job.uuid,
        "stem_index": job.stem_index,
        "stem_name": job.stem_name,
        "variant": job.variant,
        "codec": job.codec,
        "file_path": job.file_path,
    }
    return [{**base, **row} for row in compute_audio_metrics(job.file_path, _worker_window_sec)]


def compute_stem_metrics(
    jobs: Sequence[StemJob], window_sec: float | None, workers: int = 1
) -> list[dict[str, Any]]:
    """Measure all stems, optionally in parallel.

    Rows come back in job order whatever the number of workers.
    """
    records: list[dict[str, Any]] = []

    if workers <= 1:
        _init_worker(window_sec)
        for job in tqdm(jobs, desc="Computing stem metrics"):
            records.extend(_process_stem_job(job))
    else:
        with Pool(workers, initializer=_init_worker, initargs=(window_sec,)) as pool:
            for rows in tqdm(
                pool.imap(_process_stem_job, jobs, chunksize=4),
                total=len(jobs),
                desc=f"Computing stem metrics ({workers} workers)",
            ):
                records.extend(rows)

    if window_sec is not None:
        partial = {r["file_path"] for r in records if r["is_partial"]}
        if partial:
            logger.warning(
                "%d stem file(s) are shorter than the %ss window; "
                "kept as one full-duration row each.",
                len(partial),
                window_sec,
            )
    return records


# ---------------------------------------------------------------------------
# Stage 2: mix-level energy ratio
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MixEntry:
    """One pre-computed mix and the stems it is made of."""

    audio_path: str
    uuid: str
    codec: str
    ai_indices: tuple[int, ...]
    n_stems: int


def discover_mix_entries(
    dataset_path: Path | str, codecs: Sequence[str]
) -> list[MixEntry]:
    """List the pre-computed mixes of every track under the given codecs.

    Walks ``{track}/mixes/{codec}/U*/`` for files with an audio extension and
    parses each name with ``MixSpec.from_filename``. Files that do not follow
    the mix naming convention are ignored. If one mix exists in several
    formats, the first in ``AUDIO_EXTENSIONS`` order is used.
    """
    entries: list[MixEntry] = []
    for track_dir in iter_track_dirs(dataset_path):
        track = load_track(track_dir)
        for codec in codecs:
            codec_dir = get_mix_dir(track, MixSpec((), codec)).parent
            if not codec_dir.is_dir():
                continue
            found: dict[tuple[str, str], tuple[int, MixSpec, Path]] = {}
            for u_dir in sorted(p for p in codec_dir.iterdir() if p.is_dir()):
                for path in sorted(u_dir.iterdir()):
                    if path.suffix not in AUDIO_EXTENSIONS:
                        continue
                    try:
                        spec = MixSpec.from_filename(path.name, codec)
                    except ValueError:
                        continue
                    rank = AUDIO_EXTENSIONS.index(path.suffix)
                    key = (u_dir.name, spec.filename)
                    if key not in found or rank < found[key][0]:
                        found[key] = (rank, spec, path)
            for key in sorted(found):
                _, spec, path = found[key]
                entries.append(
                    MixEntry(str(path), track_dir.name, codec, spec.ai_indices, track.num_stems)
                )
    return entries


def _is_blank(value: Any) -> bool:
    return value is None or value == "" or (isinstance(value, float) and math.isnan(value))


def count_stem_errors(stem_rows: Iterable[dict[str, Any]]) -> int:
    """Number of stem rows whose file could not be read."""
    return sum(1 for r in stem_rows if not _is_blank(r.get("error")))


StemLookup = dict[tuple[str, int, str, str], list[dict[str, float]]]


def build_stem_lookup(stem_rows: Iterable[dict[str, Any]]) -> StemLookup:
    """Index stem rows by (uuid, stem_index, variant, codec).

    Each value is the list of that stem's windows sorted by start time.
    Rows with an ``error`` are left out. ``codec`` is "" for real stems.
    Values may be strings (a CSV read back) or numbers.
    """
    lookup: StemLookup = {}
    for row in stem_rows:
        if not _is_blank(row.get("error")):
            continue
        codec = row["codec"]
        key = (
            row["uuid"],
            int(row["stem_index"]),
            row["variant"],
            codec if isinstance(codec, str) else "",
        )
        lookup.setdefault(key, []).append(
            {
                "rms": float(row["rms"]),
                "window_start_sec": float(row["window_start_sec"]),
                "window_end_sec": float(row["window_end_sec"]),
            }
        )
    for windows in lookup.values():
        windows.sort(key=lambda w: w["window_start_sec"])
    return lookup


def mix_rows(mix: MixEntry, lookup: StemLookup) -> list[dict[str, Any]]:
    """Energy rows of one mix: one per window shared by its stems.

    For every window, the power (rms squared) of the AI stems and of the real
    stems is summed and ``ai_energy_ratio = ai / (ai + real)``, NaN when the
    total power is 0. A stem index in ``mix.ai_indices`` is read from the AI
    rows of ``mix.codec``, any other index from the real rows. Stems with no
    usable rows are skipped and the rows get ``missing_stems=True``. If no
    stem has rows, a single row with NaN metrics is returned so the mix is
    not dropped.
    """
    ai_set = set(mix.ai_indices)
    per_window: dict[float, dict[str, float]] = {}
    missing = False

    for idx in range(mix.n_stems):
        is_ai = idx in ai_set
        key = (mix.uuid, idx, "ai" if is_ai else "real", mix.codec if is_ai else "")
        windows = lookup.get(key)
        if not windows:
            missing = True
            continue

        for window in windows:
            start = window["window_start_sec"]
            bucket = per_window.setdefault(
                start,
                {
                    "window_start_sec": start,
                    "window_end_sec": window["window_end_sec"],
                    "ai_power": 0.0,
                    "real_power": 0.0,
                },
            )
            # Stems of one mix share the window grid; if the ends disagree
            # (whole-file mode, slightly different durations) keep the longest.
            if window["window_end_sec"] > bucket["window_end_sec"]:
                bucket["window_end_sec"] = window["window_end_sec"]
            bucket["ai_power" if is_ai else "real_power"] += window["rms"] ** 2

    base = {
        "audio_path": mix.audio_path,
        "uuid": mix.uuid,
        "codec": mix.codec,
        "n_ai_stems": len(mix.ai_indices),
        "n_total_stems": mix.n_stems,
    }

    if not per_window:
        return [
            {
                **base,
                "window_start_sec": math.nan,
                "window_end_sec": math.nan,
                "ai_power": math.nan,
                "real_power": math.nan,
                "total_power": math.nan,
                "ai_energy_ratio": math.nan,
                "missing_stems": True,
            }
        ]

    rows = []
    for start in sorted(per_window):
        bucket = per_window[start]
        ai_power = bucket["ai_power"]
        real_power = bucket["real_power"]
        total_power = ai_power + real_power
        rows.append(
            {
                **base,
                "window_start_sec": bucket["window_start_sec"],
                "window_end_sec": bucket["window_end_sec"],
                "ai_power": ai_power,
                "real_power": real_power,
                "total_power": total_power,
                "ai_energy_ratio": (
                    float(ai_power / total_power) if total_power > 0 else math.nan
                ),
                "missing_stems": missing,
            }
        )
    return rows


def compute_mix_metrics(
    mixes: Iterable[MixEntry], stem_rows: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Derive the mix energy rows from stem rows (no audio is read)."""
    lookup = build_stem_lookup(stem_rows)
    records: list[dict[str, Any]] = []
    for mix in tqdm(list(mixes), desc="Computing mix metrics"):
        records.extend(mix_rows(mix, lookup))
    return records


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


def _format_cell(value: Any) -> str:
    """Format like pandas.to_csv: NaN/None as empty, floats by repr."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def write_metrics_csv(
    path: Path | str, rows: Iterable[dict[str, Any]], columns: Sequence[str]
) -> None:
    """Write rows to a CSV with the given column order."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_format_cell(row.get(c)) for c in columns])


def read_stem_csv(path: Path | str) -> list[dict[str, str]]:
    """Read a stem metrics CSV back as string rows (for ``build_stem_lookup``)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
