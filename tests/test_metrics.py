"""Tests for energy metrics (per-stem RMS and mix AI energy ratio)."""

import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from aria import metrics
from aria.dataset import ARIADataset
from aria.metrics import (
    MIX_COLUMNS,
    STEM_COLUMNS,
    MixEntry,
    build_stem_lookup,
    discover_mix_entries,
    discover_stem_jobs,
    mix_rows,
    windowed_rms,
)
from aria.mix import generate_all_mix_specs, get_mix_path
from aria.splits import load_energy_metrics
from tests.conftest import CODEC, write_audio

SR = 1000  # low rate keeps the pure-numpy tests tiny


def square(amplitude: float, n: int) -> np.ndarray:
    """Float32 +/-amplitude signal: its RMS is exactly the amplitude."""
    return np.tile(np.float32([amplitude, -amplitude]), n // 2 + 1)[:n]


def stem_row(uuid, index, variant, rms, start=0.0, end=1.0, codec="", error=""):
    return {
        "uuid": uuid,
        "stem_index": index,
        "stem_name": f"s{index}",
        "variant": variant,
        "codec": codec if variant == "ai" else "",
        "file_path": "x",
        "rms": rms,
        "rms_db": 0.0,
        "duration_sec": end,
        "window_start_sec": start,
        "window_end_sec": end,
        "is_partial": False,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


class TestWindowedRms:
    def test_splits_into_non_overlapping_windows(self):
        y = np.concatenate([square(0.1, 2 * SR), square(0.4, 2 * SR), square(0.2, 2 * SR)])
        rows = windowed_rms(y, SR, 2.0)

        assert [r["window_start_sec"] for r in rows] == [0.0, 2.0, 4.0]
        assert [r["window_end_sec"] for r in rows] == [2.0, 4.0, 6.0]
        assert [r["rms"] for r in rows] == pytest.approx([0.1, 0.4, 0.2], rel=1e-6)
        assert all(r["duration_sec"] == 6.0 for r in rows)
        assert not any(r["is_partial"] for r in rows)

    def test_rms_db(self):
        (row,) = windowed_rms(square(0.5, SR), SR, 1.0)
        assert row["rms_db"] == pytest.approx(20 * math.log10(0.5 + 1e-10))

    def test_trailing_partial_window_is_dropped(self):
        # 2.5 windows of 1 s: the last half window must not appear.
        y = np.concatenate([square(0.1, 2 * SR), square(0.9, SR // 2)])
        rows = windowed_rms(y, SR, 1.0)

        assert len(rows) == 2
        assert [r["rms"] for r in rows] == pytest.approx([0.1, 0.1], rel=1e-6)
        assert rows[0]["duration_sec"] == 2.5

    def test_exact_multiple_keeps_every_window(self):
        assert len(windowed_rms(square(0.1, 3 * SR), SR, 1.0)) == 3

    def test_short_file_yields_one_partial_row(self):
        y = square(0.3, SR // 2)
        (row,) = windowed_rms(y, SR, 1.0)

        assert row["is_partial"] is True
        assert row["window_start_sec"] == 0.0
        assert row["window_end_sec"] == 0.5
        assert row["duration_sec"] == 0.5
        assert row["rms"] == pytest.approx(0.3, rel=1e-6)

    def test_whole_file_when_no_window(self):
        y = np.concatenate([square(0.2, SR), square(0.4, SR)])
        (row,) = windowed_rms(y, SR, None)

        assert row["is_partial"] is False
        assert (row["window_start_sec"], row["window_end_sec"]) == (0.0, 2.0)
        assert row["rms"] == pytest.approx(math.sqrt((0.2**2 + 0.4**2) / 2), rel=1e-6)

    def test_window_length_is_rounded_to_samples(self):
        # 0.0105 s at 1 kHz is 10.5 samples, which rounds to 10 (banker's rounding).
        rows = windowed_rms(square(0.1, 25), SR, 0.0105)
        assert [r["window_end_sec"] for r in rows] == [0.01, 0.02]

    def test_window_shorter_than_one_sample_is_an_error(self):
        with pytest.raises(ValueError):
            windowed_rms(square(0.1, 100), SR, 0.0001)


# ---------------------------------------------------------------------------
# Mix aggregation (no audio)
# ---------------------------------------------------------------------------


def mix(ai_indices, n_stems=3, codec=CODEC, uuid="t"):
    return MixEntry(f"/data/{uuid}/mixes/{codec}/U1/mix_S-0.mp3", uuid, codec, tuple(ai_indices), n_stems)


class TestMixRows:
    def test_alpha_from_known_stem_powers(self):
        rows = (
            [stem_row("t", i, "real", r) for i, r in enumerate([0.3, 0.4, 0.5])]
            + [stem_row("t", i, "ai", r, codec=CODEC) for i, r in enumerate([0.1, 0.2, 0.6])]
        )
        lookup = build_stem_lookup(rows)

        # Stems 0 and 2 are AI, stem 1 is real.
        (row,) = mix_rows(mix([0, 2]), lookup)

        ai = 0.1**2 + 0.6**2
        real = 0.4**2
        assert row["ai_power"] == pytest.approx(ai)
        assert row["real_power"] == pytest.approx(real)
        assert row["total_power"] == pytest.approx(ai + real)
        assert row["ai_energy_ratio"] == pytest.approx(ai / (ai + real))
        assert row["missing_stems"] is False
        assert (row["n_ai_stems"], row["n_total_stems"]) == (2, 3)

    def test_all_real_is_zero_and_all_ai_is_one(self):
        rows = (
            [stem_row("t", i, "real", 0.3) for i in range(2)]
            + [stem_row("t", i, "ai", 0.2, codec=CODEC) for i in range(2)]
        )
        lookup = build_stem_lookup(rows)

        (all_real,) = mix_rows(mix([], n_stems=2), lookup)
        (all_ai,) = mix_rows(mix([0, 1], n_stems=2), lookup)

        assert all_real["ai_energy_ratio"] == 0.0
        assert all_ai["ai_energy_ratio"] == 1.0

    def test_windows_are_aggregated_independently(self):
        rows = []
        for i, (real, ai) in enumerate([(0.3, 0.1), (0.4, 0.2)]):
            for w, scale in enumerate([1.0, 2.0]):
                rows.append(stem_row("t", i, "real", real * scale, w, w + 1))
                rows.append(stem_row("t", i, "ai", ai * scale, w, w + 1, codec=CODEC))
        lookup = build_stem_lookup(rows)

        first, second = mix_rows(mix([0], n_stems=2), lookup)

        # The window scale cancels in alpha but not in the powers.
        assert first["window_start_sec"] == 0.0 and second["window_start_sec"] == 1.0
        assert first["ai_energy_ratio"] == pytest.approx(second["ai_energy_ratio"])
        assert second["total_power"] == pytest.approx(4 * first["total_power"])

    def test_silent_window_gives_nan_ratio(self):
        rows = [stem_row("t", i, v, 0.0, codec=CODEC) for i in range(2) for v in ("real", "ai")]
        (row,) = mix_rows(mix([0], n_stems=2), build_stem_lookup(rows))

        assert row["total_power"] == 0.0
        assert math.isnan(row["ai_energy_ratio"])
        assert row["missing_stems"] is False

    def test_missing_stem_is_skipped_and_flagged(self):
        rows = [
            stem_row("t", 0, "real", 0.3),
            stem_row("t", 1, "ai", 0.2, codec=CODEC),
            # Stem 2 has neither a real row nor an AI row.
        ]
        (row,) = mix_rows(mix([1]), build_stem_lookup(rows))

        assert row["missing_stems"] is True
        assert row["ai_energy_ratio"] == pytest.approx(0.2**2 / (0.2**2 + 0.3**2))

    def test_stem_with_read_error_counts_as_missing(self):
        rows = [
            stem_row("t", 0, "real", 0.3),
            stem_row("t", 1, "real", math.nan, error="boom"),
        ]
        (row,) = mix_rows(mix([], n_stems=2), build_stem_lookup(rows))

        assert row["missing_stems"] is True
        assert row["real_power"] == pytest.approx(0.09)

    def test_no_usable_stems_still_emits_one_row(self):
        (row,) = mix_rows(mix([0]), build_stem_lookup([]))

        assert row["missing_stems"] is True
        assert math.isnan(row["ai_energy_ratio"])
        assert math.isnan(row["window_start_sec"])

    def test_ai_stem_of_another_codec_is_not_used(self):
        rows = [
            stem_row("t", 0, "real", 0.3),
            stem_row("t", 0, "ai", 0.2, codec="encodec_6"),
        ]
        (row,) = mix_rows(mix([0], n_stems=1), build_stem_lookup(rows))

        assert row["missing_stems"] is True

    def test_string_rows_from_csv_are_accepted(self):
        rows = [
            {k: str(v) for k, v in stem_row("t", 1, "real", 0.3).items()},
            {k: str(v) for k, v in stem_row("t", 0, "ai", 0.1, codec=CODEC).items()},
        ]
        (row,) = mix_rows(mix([0], n_stems=2), build_stem_lookup(rows))

        assert row["ai_energy_ratio"] == pytest.approx(0.01 / (0.01 + 0.09))


# ---------------------------------------------------------------------------
# Audio and the CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def librosa():
    return pytest.importorskip("librosa")


def write_wav(path: Path, amplitude: float, num_samples: int, sr: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, square(amplitude, num_samples), sr, subtype="FLOAT")


def load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "compute_metrics.py"
    spec = importlib.util.spec_from_file_location("compute_metrics_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# Real and AI RMS per stem of the "known" track, in alphabetical stem order.
REAL_RMS = {"bass": 0.4, "drums": 0.2, "vocals": 0.3}
AI_RMS = {"bass": 0.1, "drums": 0.3, "vocals": 0.2}


@pytest.fixture
def known_dataset(tmp_path):
    """One track whose stems are +/-a square waves of known RMS, all WAV.

    Stems are written out of alphabetical order in metadata.json to check
    that indices follow the sorted names.
    """
    track = tmp_path / "ds" / "track-known"
    track.mkdir(parents=True)
    stems = ["vocals", "drums", "bass"]
    (track / "metadata.json").write_text(
        json.dumps({"id": "track-known", "artist": "A", "name": "N", "genre": "g", "stems": stems})
    )
    for name in stems:
        write_wav(track / "stems" / "real" / f"{name}.wav", REAL_RMS[name], 44100 * 3)
        write_wav(track / "stems" / "ai_generated" / CODEC / f"{name}.wav", AI_RMS[name], 44100 * 3)
    dataset = ARIADataset(track.parent)
    for spec in generate_all_mix_specs(3, CODEC):
        # Mix audio is never read; empty files only mark which mixes exist.
        path = get_mix_path(dataset[0], spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return track.parent


def expected_alpha(ai_indices):
    names = sorted(REAL_RMS)  # alphabetical stem order
    ai = sum(AI_RMS[names[i]] ** 2 for i in ai_indices)
    real = sum(REAL_RMS[n] ** 2 for i, n in enumerate(names) if i not in ai_indices)
    return ai / (ai + real)


class TestDiscovery:
    def test_stem_jobs_use_alphabetical_indices(self, known_dataset):
        jobs = discover_stem_jobs(known_dataset, [CODEC])

        real = {j.stem_name: j.stem_index for j in jobs if j.variant == "real"}
        assert real == {"bass": 0, "drums": 1, "vocals": 2}
        assert len(jobs) == 6
        assert all(j.codec == CODEC for j in jobs if j.variant == "ai")
        assert all(j.codec == "" for j in jobs if j.variant == "real")

    def test_stem_jobs_skip_unknown_codec(self, known_dataset):
        jobs = discover_stem_jobs(known_dataset, ["encodec_24"])
        assert {j.variant for j in jobs} == {"real"}

    def test_mix_entries(self, known_dataset):
        entries = discover_mix_entries(known_dataset, [CODEC])

        assert len(entries) == 8
        assert all(Path(e.audio_path).is_absolute() for e in entries)
        assert {e.ai_indices for e in entries} == {
            (), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)
        }
        assert {e.n_stems for e in entries} == {3}

    def test_mix_entries_ignore_other_files(self, known_dataset):
        mix_dir = known_dataset / "track-known" / "mixes" / CODEC / "U1"
        (mix_dir / "mix_S-0.json").write_text("{}")
        (mix_dir / "notes.mp3").touch()
        (mix_dir / "mix_S-1.wav").touch()  # same mix as mix_S-1.mp3: one entry only

        entries = discover_mix_entries(known_dataset, [CODEC])
        assert len(entries) == 8
        u1 = [e for e in entries if e.ai_indices == (1,)]
        assert Path(u1[0].audio_path).suffix == ".wav"  # .wav outranks .mp3


class TestComputeMetricsScript:
    def run(self, dataset, out, *extra):
        script = load_script()
        return script.main(["--aria-path", str(dataset), "--output-dir", str(out), *extra])

    def test_known_alpha_end_to_end(self, librosa, known_dataset, tmp_path):
        out = tmp_path / "out"
        assert self.run(known_dataset, out, "--window", "1") == 0

        stems = read_csv(out / "stem_metrics.csv")
        assert tuple(stems[0]) == STEM_COLUMNS
        assert len(stems) == 6 * 3  # 3 stems x (real + ai) x 3 windows
        bass_real = [r for r in stems if r["stem_name"] == "bass" and r["variant"] == "real"]
        assert [float(r["window_start_sec"]) for r in bass_real] == [0.0, 1.0, 2.0]
        assert float(bass_real[0]["rms"]) == pytest.approx(0.4, rel=1e-6)

        mixes = read_csv(out / "mix_metrics.csv")
        assert tuple(mixes[0]) == MIX_COLUMNS
        assert "is_partial" not in mixes[0]
        assert len(mixes) == 8 * 3
        for row in mixes:
            indices = tuple(
                int(i) for i in Path(row["audio_path"]).stem[len("mix_S-"):].split("-") if i.isdigit()
            )
            assert row["codec"] == CODEC
            assert row["missing_stems"] == "False"
            assert float(row["ai_energy_ratio"]) == pytest.approx(expected_alpha(indices), rel=1e-5)

    def test_non_windowed_run_gives_one_row_per_file_and_mix(self, librosa, known_dataset, tmp_path):
        out = tmp_path / "out"
        assert self.run(known_dataset, out) == 0

        stems = read_csv(out / "stem_metrics.csv")
        mixes = read_csv(out / "mix_metrics.csv")
        assert len(stems) == 6
        assert len(mixes) == 8
        assert {r["window_start_sec"] for r in mixes} == {"0.0"}
        assert all(float(r["window_end_sec"]) == pytest.approx(3.0) for r in mixes)

    def test_output_is_readable_by_load_energy_metrics(self, librosa, known_dataset, tmp_path):
        out = tmp_path / "out"
        self.run(known_dataset, out)

        lookup = load_energy_metrics(out / "mix_metrics.csv")
        assert lookup["track-known/mixes/encodec_3/U0/mix_S-empty.mp3"] == 0.0
        assert lookup["track-known/mixes/encodec_3/U3/mix_S-0-1-2.mp3"] == 1.0
        assert lookup["track-known/mixes/encodec_3/U1/mix_S-0.mp3"] == pytest.approx(
            expected_alpha((0,)), rel=1e-5
        )

    def test_existing_stem_csv_is_reused_unless_overwrite(self, librosa, known_dataset, tmp_path, monkeypatch):
        out = tmp_path / "out"
        self.run(known_dataset, out)

        def boom(*args, **kwargs):
            raise AssertionError("stem stage must not run")

        monkeypatch.setattr(metrics, "compute_stem_metrics", boom)
        script = load_script()
        monkeypatch.setattr(script, "compute_stem_metrics", boom)
        argv = ["--aria-path", str(known_dataset), "--output-dir", str(out)]
        assert script.main(argv) == 0
        with pytest.raises(AssertionError):
            script.main([*argv, "--overwrite"])

    def test_workers_give_same_rows_as_serial(self, librosa, known_dataset, tmp_path):
        self.run(known_dataset, tmp_path / "serial", "--window", "1")
        self.run(known_dataset, tmp_path / "pool", "--window", "1", "--workers", "2")

        for name in ("stem_metrics.csv", "mix_metrics.csv"):
            assert (tmp_path / "serial" / name).read_text() == (tmp_path / "pool" / name).read_text()

    def test_silent_window_and_short_file_in_csv(self, librosa, tmp_path):
        track = tmp_path / "ds" / "t1"
        track.mkdir(parents=True)
        (track / "metadata.json").write_text(
            json.dumps({"id": "t1", "artist": "A", "name": "N", "genre": "g", "stems": ["a"]})
        )
        # 1.5 s: one full 1 s window (silent) and a dropped 0.5 s tail.
        write_wav(track / "stems" / "real" / "a.wav", 0.0, 66150)
        write_wav(track / "stems" / "ai_generated" / CODEC / "a.wav", 0.0, 66150)
        dataset = ARIADataset(track.parent)
        for spec in generate_all_mix_specs(1, CODEC):
            path = get_mix_path(dataset[0], spec)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        out = tmp_path / "out"
        assert self.run(track.parent, out, "--window", "1") == 0
        assert [r["is_partial"] for r in read_csv(out / "stem_metrics.csv")] == ["False"] * 2
        mixes = read_csv(out / "mix_metrics.csv")
        assert len(mixes) == 2
        assert {r["ai_energy_ratio"] for r in mixes} == {""}  # NaN is written empty

        # Same data with a 2 s window: the stems are shorter than a window.
        out2 = tmp_path / "out2"
        assert self.run(track.parent, out2, "--window", "2") == 0
        stems = read_csv(out2 / "stem_metrics.csv")
        assert [r["is_partial"] for r in stems] == ["True", "True"]
        assert float(stems[0]["window_end_sec"]) == pytest.approx(1.5)

    def test_unreadable_stem_is_recorded_and_mix_flagged(
        self, librosa, known_dataset, tmp_path, monkeypatch
    ):
        real_load = librosa.load

        def flaky_load(path, **kwargs):
            if path.endswith("real/drums.wav"):
                raise RuntimeError("cannot decode")
            return real_load(path, **kwargs)

        monkeypatch.setattr(librosa, "load", flaky_load)

        out = tmp_path / "out"
        assert self.run(known_dataset, out) == 0

        stems = read_csv(out / "stem_metrics.csv")
        errors = [r for r in stems if r["error"]]
        assert len(errors) == 1 and errors[0]["stem_name"] == "drums"
        assert errors[0]["rms"] == ""
        assert errors[0]["error"] == "cannot decode"
        mixes = {Path(r["audio_path"]).stem: r for r in read_csv(out / "mix_metrics.csv")}
        assert mixes["mix_S-empty"]["missing_stems"] == "True"
        assert mixes["mix_S-1"]["missing_stems"] == "False"  # drums is AI there

    def test_default_output_dir_is_under_dataset(self, librosa, known_dataset):
        script = load_script()
        assert script.main(["--aria-path", str(known_dataset)]) == 0
        assert (known_dataset / "metrics" / "mix_metrics.csv").exists()

    def test_bad_arguments(self, known_dataset, tmp_path):
        script = load_script()
        base = ["--aria-path", str(known_dataset), "--output-dir", str(tmp_path / "o")]
        assert script.main([*base, "--window", "0"]) == 2
        assert script.main([*base, "--workers", "0"]) == 2
        assert script.main(["--aria-path", str(tmp_path / "missing")]) == 2


def test_mp3_stems_from_conftest_dataset(librosa, mock_dataset_path, tmp_path):
    """The conftest dataset has MP3 AI stems; run the CLI over it."""
    for track in ARIADataset(mock_dataset_path):
        for spec in generate_all_mix_specs(track.num_stems, CODEC):
            write_audio(get_mix_path(track, spec), 44100 // 2)

    out = tmp_path / "out"
    script = load_script()
    argv = ["--aria-path", str(mock_dataset_path), "--output-dir", str(out), "--window", "0.2"]
    assert script.main(argv) == 0

    mixes = read_csv(out / "mix_metrics.csv")
    assert tuple(mixes[0]) == MIX_COLUMNS
    assert {r["uuid"] for r in mixes} == {"track-001", "track-002", "track-003"}
    assert all(r["missing_stems"] == "False" for r in mixes)
    assert all(0.0 <= float(r["ai_energy_ratio"]) <= 1.0 for r in mixes)
