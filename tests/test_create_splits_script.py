"""End-to-end tests of scripts/create_splits.py on the synthetic dataset."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aria.dataset import ARIADataset
from aria.mix import generate_all_mix_specs, get_mix_path
from aria.splits import (
    SplitManifest,
    is_window_manifest,
    load_window_manifest,
    stratified_split_by_num_stems,
)
from tests.conftest import CODEC, make_track, write_audio
from tests.test_window_manifest import full_window_csv

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "create_splits.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("create_splits_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dataset(mock_dataset_path):
    for track in ARIADataset(mock_dataset_path):
        for spec in generate_all_mix_specs(track.num_stems, CODEC):
            write_audio(get_mix_path(track, spec), 44100 // 2)
    return mock_dataset_path


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_windowed_csv_gives_window_manifest(script, dataset, tmp_path):
    csv_path = full_window_csv(dataset, tmp_path / "windows.csv", n_windows=3)
    out = tmp_path / "splits" / "windows.jsonl"

    script.create_splits(str(dataset), str(out), metrics_csv=str(csv_path), seed=1)

    assert is_window_manifest(out)
    records = load_window_manifest(out)
    assert len(records) == (4 + 8 + 16) * 3
    meta = json.loads((out.parent / "windows.meta.json").read_text())
    assert meta["version"] == "3.0"
    assert meta["is_windowed"] is True
    assert meta["window_sec"] == 5.0
    assert meta["total_records"] == len(records)
    assert sum(s["num_partial_windows"] for s in meta["splits"].values()) == 28


def test_non_windowed_csv_keeps_per_mix_manifest(script, dataset, tmp_path):
    csv_path = tmp_path / "mix.csv"
    lines = ["audio_path,ai_energy_ratio"]
    for track in ARIADataset(dataset):
        for spec in generate_all_mix_specs(track.num_stems, CODEC):
            lines.append(f"/other/{get_mix_path(track, spec).relative_to(dataset)},0.5")
    csv_path.write_text("\n".join(lines) + "\n")
    out = tmp_path / "mix.jsonl"

    script.create_splits(str(dataset), str(out), metrics_csv=str(csv_path))

    assert not is_window_manifest(out)
    manifest = SplitManifest(out)
    assert len(manifest) == 28
    assert {r.ai_energy_ratio for r in manifest} == {0.5}


def test_split_from_reuses_the_reference_split(script, dataset, tmp_path):
    """A per-mix manifest and a window manifest can share one track split."""
    per_mix = tmp_path / "per_mix.jsonl"
    script.create_splits(str(dataset), str(per_mix), seed=3)
    reference = {r["track_id"]: r["split"] for r in read_lines(per_mix)}
    flipped = {t: "test" if s == "train" else "train" for t, s in reference.items()}
    ref_path = tmp_path / "ref.jsonl"
    ref_path.write_text("".join(
        json.dumps({"track_id": t, "split": s}) + "\n" for t, s in flipped.items()
    ))

    out = tmp_path / "windows.jsonl"
    csv_path = full_window_csv(dataset, tmp_path / "w.csv")
    script.create_splits(
        str(dataset), str(out), metrics_csv=str(csv_path), seed=99, split_from=str(ref_path)
    )

    assert {r.track_id: r.split for r in load_window_manifest(out)} == flipped
    meta = json.loads((tmp_path / "windows.meta.json").read_text())
    assert meta["split_from"] == "ref.jsonl"
    assert meta["seed"] is None


def test_split_from_missing_track_is_an_error(script, dataset, tmp_path):
    tracks = list(ARIADataset(dataset))
    ref_path = tmp_path / "ref.jsonl"
    ref_path.write_text(json.dumps({"track_id": tracks[0].id, "split": "train"}) + "\n")

    with pytest.raises(ValueError, match="not in the reference split"):
        script.create_splits(str(dataset), str(tmp_path / "o.jsonl"), split_from=str(ref_path))
    assert not (tmp_path / "o.jsonl").exists()


def test_split_from_allow_unassigned_drops_other_tracks(script, dataset, tmp_path):
    tracks = list(ARIADataset(dataset))
    ref_path = tmp_path / "ref.jsonl"
    ref_path.write_text(json.dumps({"track_id": tracks[0].id, "split": "val"}) + "\n")
    out = tmp_path / "o.jsonl"

    script.create_splits(
        str(dataset), str(out), split_from=str(ref_path), allow_unassigned=True
    )
    assert {r["track_id"] for r in read_lines(out)} == {tracks[0].id}


def test_split_from_conflict_is_an_error(script, dataset, tmp_path):
    track = next(iter(ARIADataset(dataset)))
    ref_path = tmp_path / "ref.jsonl"
    ref_path.write_text("".join(
        json.dumps({"track_id": track.id, "split": s}) + "\n" for s in ("train", "val")
    ))
    with pytest.raises(ValueError, match="both"):
        script.create_splits(str(dataset), str(tmp_path / "o.jsonl"), split_from=str(ref_path))


def test_default_split_does_not_depend_on_track_order(script, tmp_path, monkeypatch):
    root = tmp_path / "aria"
    root.mkdir()
    for i in range(12):
        make_track(root, f"track-{i:02d}", ["bass", "drums"], 2000, name=f"Song {i}")
    for track in ARIADataset(root):
        for spec in generate_all_mix_specs(track.num_stems, CODEC):
            write_audio(get_mix_path(track, spec), 2000)

    out_a = tmp_path / "a.jsonl"
    script.create_splits(str(root), str(out_a), seed=5)

    original = script.ARIADataset

    class Reversed:
        def __init__(self, path):
            self.tracks = list(reversed(list(original(path))))

        def __iter__(self):
            return iter(self.tracks)

    monkeypatch.setattr(script, "ARIADataset", Reversed)
    out_b = tmp_path / "b.jsonl"
    script.create_splits(str(root), str(out_b), seed=5)

    def assignment(path):
        return {r["track_id"]: r["split"] for r in read_lines(path)}

    assert assignment(out_a) == assignment(out_b)

    # Guard: the plain stratified split would have differed for these orders.
    tracks = list(original(root))
    plain = stratified_split_by_num_stems(tracks, 0.8, 0.1, 5)
    assert plain != stratified_split_by_num_stems(list(reversed(tracks)), 0.8, 0.1, 5)


def test_cli_exits_nonzero_on_split_conflict(dataset, tmp_path):
    track = next(iter(ARIADataset(dataset)))
    ref_path = tmp_path / "ref.jsonl"
    ref_path.write_text("".join(
        json.dumps({"track_id": track.id, "split": s}) + "\n" for s in ("train", "val")
    ))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--aria-path", str(dataset),
         "--output", str(tmp_path / "o.jsonl"), "--split-from", str(ref_path)],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    assert result.returncode == 1
    assert "both" in result.stderr
