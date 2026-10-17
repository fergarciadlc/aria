"""Tests for window manifests: windowed metrics CSV to one record per window."""

import csv
import json
import logging
from pathlib import Path

import pytest

from aria.dataset import ARIADataset
from aria.mix import generate_all_mix_specs, get_mix_path
from aria.splits import (
    WindowRecord,
    build_window_manifest,
    is_window_manifest,
    load_window_manifest,
    load_window_metrics,
    window_manifest_summary,
    write_manifest,
)
from tests.conftest import CODEC, write_audio

HEADER = ["audio_path", "ai_energy_ratio", "window_start_sec", "window_end_sec", "is_partial"]


def write_csv(path: Path, rows: list[dict], fields: list[str] = HEADER) -> Path:
    """Write rows as a CSV the way pandas does: NaN becomes an empty cell."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return path


def row(rel: str, ratio, start, end, partial=False) -> dict:
    return {
        "audio_path": f"/media/other-machine/aria/{rel}",
        "ai_energy_ratio": ratio,
        "window_start_sec": start,
        "window_end_sec": end,
        "is_partial": partial,
    }


REL = "t1/mixes/encodec_3/U1/mix_S-0.mp3"


class TestLoadWindowMetrics:
    def test_sorts_windows_by_start(self, tmp_path):
        csv_path = write_csv(tmp_path / "m.csv", [
            row(REL, 0.3, 10.0, 15.0),
            row(REL, 0.1, 0.0, 5.0),
            row(REL, 0.2, 5.0, 10.0),
        ])
        metrics = load_window_metrics(csv_path)

        assert metrics.is_windowed
        assert [w.start_sec for w in metrics.windows[REL]] == [0.0, 5.0, 10.0]
        assert [w.ai_energy_ratio for w in metrics.windows[REL]] == [0.1, 0.2, 0.3]

    def test_key_is_last_five_path_components(self, tmp_path):
        csv_path = write_csv(tmp_path / "m.csv", [row(REL, 0.1, 0, 5), row(REL, 0.1, 5, 10)])
        assert list(load_window_metrics(csv_path).windows) == [REL]

    def test_empty_ratio_becomes_zero_and_is_counted(self, tmp_path, caplog):
        csv_path = write_csv(tmp_path / "m.csv", [
            row(REL, None, 0.0, 5.0),
            row(REL, 0.5, 5.0, 10.0),
            row(REL, "nan", 10.0, 15.0),
        ])
        with caplog.at_level(logging.INFO, logger="aria.splits"):
            metrics = load_window_metrics(csv_path)

        assert [w.ai_energy_ratio for w in metrics.windows[REL]] == [0.0, 0.5, 0.0]
        assert metrics.num_silent == 2
        assert "2 window(s) with empty ai_energy_ratio" in caplog.text

    def test_nan_bounds_fall_back_to_zero(self, tmp_path):
        csv_path = write_csv(tmp_path / "m.csv", [
            row(REL, 0.1, None, None),
            row(REL, 0.1, 0.0, 5.0),
        ])
        windows = load_window_metrics(csv_path).windows[REL]
        assert [(w.start_sec, w.end_sec) for w in windows] == [(0.0, 0.0), (0.0, 5.0)]

    def test_window_size_is_most_common_non_partial_length(self, tmp_path):
        csv_path = write_csv(tmp_path / "m.csv", [
            row(REL, 0.1, 0.0, 5.0),
            row(REL, 0.1, 5.0, 10.0),
            row(REL, 0.1, 10.0, 12.5, partial=True),
            row("t2/mixes/encodec_3/U1/mix_S-0.mp3", 0.1, 0.0, 2.0, partial=True),
            row("t2/mixes/encodec_3/U1/mix_S-0.mp3", 0.1, 0.0, 5.0),
        ])
        metrics = load_window_metrics(csv_path)
        assert metrics.window_sec == 5.0
        assert [w.is_partial for w in metrics.windows[REL]] == [False, False, True]

    def test_window_size_ignores_float_noise(self, tmp_path):
        csv_path = write_csv(tmp_path / "m.csv", [
            row(REL, 0.1, 0.0, 1.0004),
            row(REL, 0.1, 1.0, 2.0),
        ])
        assert load_window_metrics(csv_path).window_sec == 1.0

    def test_csv_without_window_columns_is_not_windowed(self, tmp_path):
        csv_path = write_csv(
            tmp_path / "m.csv",
            [{"audio_path": f"/x/{REL}", "ai_energy_ratio": 0.2}],
            fields=["audio_path", "ai_energy_ratio"],
        )
        metrics = load_window_metrics(csv_path)
        assert not metrics.is_windowed
        assert metrics.window_sec is None

    def test_single_row_per_mix_is_not_windowed(self, tmp_path):
        """Matches the original: windowed means some mix has several rows."""
        csv_path = write_csv(tmp_path / "m.csv", [row(REL, 0.1, 0.0, 5.0)])
        assert not load_window_metrics(csv_path).is_windowed

    def test_missing_is_partial_column_means_false(self, tmp_path):
        fields = ["audio_path", "ai_energy_ratio", "window_start_sec", "window_end_sec"]
        csv_path = tmp_path / "m.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(fields)
            w.writerow([f"/x/{REL}", 0.1, 0.0, 5.0])
            w.writerow([f"/x/{REL}", 0.1, 5.0, 10.0])
        assert not any(w.is_partial for w in load_window_metrics(csv_path).windows[REL])


@pytest.fixture
def dataset_with_mixes(mock_dataset_path):
    for track in ARIADataset(mock_dataset_path):
        for spec in generate_all_mix_specs(track.num_stems, CODEC):
            write_audio(get_mix_path(track, spec), 44100 // 2)
    return mock_dataset_path


def full_window_csv(dataset_root: Path, path: Path, n_windows: int = 3) -> Path:
    """Hand-written windowed CSV covering every mix of the synthetic dataset."""
    rows = []
    for track in ARIADataset(dataset_root):
        for i, spec in enumerate(generate_all_mix_specs(track.num_stems, CODEC)):
            rel = get_mix_path(track, spec).relative_to(dataset_root).as_posix()
            for w in range(n_windows):
                last = w == n_windows - 1
                rows.append(row(
                    rel,
                    0.123456789 * (i + 1) / 100,
                    w * 5.0,
                    w * 5.0 + (2.5 if last else 5.0),
                    partial=last,
                ))
    return write_csv(path, rows)


class TestBuildWindowManifest:
    def test_expands_every_mix_to_its_windows(self, dataset_with_mixes, tmp_path):
        metrics = load_window_metrics(full_window_csv(dataset_with_mixes, tmp_path / "m.csv"))
        tracks = list(ARIADataset(dataset_with_mixes))
        assignment = {t.id: "train" for t in tracks}

        records, skipped = build_window_manifest(
            tracks, dataset_with_mixes, assignment, CODEC, metrics.windows
        )

        assert skipped == 0
        assert len(records) == (4 + 8 + 16) * 3
        first = records[0]
        assert first.window_index == 0
        assert [r.window_index for r in records[:3]] == [0, 1, 2]
        assert [r.is_partial for r in records[:3]] == [False, False, True]
        assert first.window_end_sec == 5.0
        assert records[2].window_end_sec == 12.5

    def test_rounding(self, dataset_with_mixes, tmp_path):
        track = next(iter(ARIADataset(dataset_with_mixes)))
        spec = generate_all_mix_specs(track.num_stems, CODEC)[1]
        rel = get_mix_path(track, spec).relative_to(dataset_with_mixes).as_posix()
        csv_path = write_csv(tmp_path / "m.csv", [
            row(rel, 0.12345678, 0.00049, 4.99951),
            row(rel, 0.1, 5.0, 10.0),
        ])
        metrics = load_window_metrics(csv_path)

        records, _ = build_window_manifest(
            [track], dataset_with_mixes, {track.id: "val"}, CODEC, metrics.windows
        )
        assert records[0].window_start_sec == 0.0
        assert records[0].window_end_sec == 5.0
        assert records[0].ai_energy_ratio == 0.123457

    def test_skips_mixes_without_metrics(self, dataset_with_mixes, tmp_path):
        track = next(iter(ARIADataset(dataset_with_mixes)))
        spec = generate_all_mix_specs(track.num_stems, CODEC)[0]
        rel = get_mix_path(track, spec).relative_to(dataset_with_mixes).as_posix()
        csv_path = write_csv(tmp_path / "m.csv", [row(rel, 0.0, 0, 5), row(rel, 0.0, 5, 10)])
        metrics = load_window_metrics(csv_path)

        records, skipped = build_window_manifest(
            [track], dataset_with_mixes, {track.id: "test"}, CODEC, metrics.windows
        )
        assert len(records) == 2
        assert skipped == 2 ** track.num_stems - 1

    def test_json_key_order_matches_original_manifests(self, dataset_with_mixes, tmp_path):
        metrics = load_window_metrics(full_window_csv(dataset_with_mixes, tmp_path / "m.csv"))
        tracks = list(ARIADataset(dataset_with_mixes))
        records, _ = build_window_manifest(
            tracks, dataset_with_mixes, {t.id: "train" for t in tracks}, CODEC, metrics.windows
        )
        assert list(records[0].to_dict()) == [
            "track_id", "artist", "track_name", "genre", "codec",
            "num_stems", "num_ai_stems", "ai_indices", "split", "mix_path",
            "window_index", "window_start_sec", "window_end_sec",
            "ai_energy_ratio", "is_partial",
        ]


class TestWindowManifestIO:
    @pytest.fixture
    def written(self, dataset_with_mixes, tmp_path):
        metrics = load_window_metrics(full_window_csv(dataset_with_mixes, tmp_path / "m.csv"))
        tracks = list(ARIADataset(dataset_with_mixes))
        splits = dict(zip((t.id for t in tracks), ("train", "val", "test"), strict=True))
        records, _ = build_window_manifest(
            tracks, dataset_with_mixes, splits, CODEC, metrics.windows
        )
        path = tmp_path / "out" / "w.jsonl"
        write_manifest(records, path)
        return records, path

    def test_roundtrip(self, written):
        records, path = written
        assert load_window_manifest(path) == records

    def test_is_window_manifest(self, written, tmp_path):
        _, path = written
        assert is_window_manifest(path)
        per_mix = tmp_path / "per_mix.jsonl"
        per_mix.write_text(json.dumps({"track_id": "t", "split": "train"}) + "\n")
        assert not is_window_manifest(per_mix)
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        assert not is_window_manifest(empty)

    def test_detector_reader_semantics(self, written):
        """Re-implements the field reads of detector-models' load_aria_w5s_windows."""
        _, path = written
        rows: dict[str, list[dict]] = {s: [] for s in ("train", "val", "test")}
        schema_checked = False
        with open(path) as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                r = json.loads(raw_line)
                if not schema_checked:
                    assert "window_start_sec" in r
                    schema_checked = True
                split_name = r.get("split")
                if split_name not in rows:
                    continue
                rows[split_name].append({
                    "mix_path": r["mix_path"],
                    "track_id": r["track_id"],
                    "split": split_name,
                    "window_index": int(r["window_index"]),
                    "window_start_sec": float(r["window_start_sec"]),
                    "window_end_sec": float(r["window_end_sec"]),
                    "ai_energy_ratio": float(r["ai_energy_ratio"]),
                    "is_partial": bool(r.get("is_partial", False)),
                })

        assert all(rows[s] for s in rows)
        assert sum(len(v) for v in rows.values()) == (4 + 8 + 16) * 3
        for entry in rows["train"]:
            assert entry["window_end_sec"] > entry["window_start_sec"]
            assert 0.0 <= entry["ai_energy_ratio"] <= 1.0

    def test_summary(self, written):
        records, _ = written
        summary = window_manifest_summary(records, CODEC, 42, 0.8, 0.1, window_sec=5.0)

        assert summary["version"] == "3.0"
        assert summary["is_windowed"] is True
        assert summary["record_unit"] == "window"
        assert summary["window_sec"] == 5.0
        assert summary["total_records"] == len(records)
        train = summary["splits"]["train"]
        assert train["num_windows"] == len([r for r in records if r.split == "train"])
        assert train["num_partial_windows"] == train["num_windows"] // 3

    def test_missing_field_is_clear_error(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"track_id": "t"}) + "\n")
        with pytest.raises(ValueError, match="Missing field"):
            load_window_manifest(path)

    def test_windowrecord_from_dict_defaults_is_partial(self, written):
        records, _ = written
        data = records[0].to_dict()
        del data["is_partial"]
        assert WindowRecord.from_dict(data).is_partial is False
