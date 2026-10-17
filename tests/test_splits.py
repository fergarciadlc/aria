"""Tests for split manifest creation and loading."""

import json

import pytest

from aria.dataset import ARIADataset
from aria.mix import generate_all_mix_specs, get_mix_path
from aria.splits import (
    SplitManifest,
    build_manifest,
    manifest_summary,
    stratified_split_by_num_stems,
    write_manifest,
)
from tests.conftest import CODEC, write_audio


@pytest.fixture
def dataset_with_mixes(mock_dataset_path):
    """Mock dataset with every encodec_3 mix pre-computed (as MP3)."""
    for track in ARIADataset(mock_dataset_path):
        for spec in generate_all_mix_specs(track.num_stems, CODEC):
            write_audio(get_mix_path(track, spec), 44100 // 2)
    return mock_dataset_path


def test_split_is_by_track_and_deterministic(mock_dataset_path):
    tracks = list(ARIADataset(mock_dataset_path))
    a = stratified_split_by_num_stems(tracks, 0.5, 0.25, seed=1)
    b = stratified_split_by_num_stems(tracks, 0.5, 0.25, seed=1)
    assert a == b
    assert set(a) == {t.id for t in tracks}
    assert set(a.values()) <= {"train", "val", "test"}


def test_split_rejects_bad_ratios(mock_dataset_path):
    with pytest.raises(ValueError):
        stratified_split_by_num_stems(list(ARIADataset(mock_dataset_path)), 0.9, 0.1, seed=0)


def test_manifest_roundtrip(dataset_with_mixes, temp_dir):
    tracks = list(ARIADataset(dataset_with_mixes))
    assignment = {t.id: "train" for t in tracks}
    records, skipped = build_manifest(tracks, dataset_with_mixes, assignment, CODEC)

    assert skipped == 0
    assert len(records) == 4 + 8 + 16

    path = temp_dir / "splits" / "m.jsonl"
    write_manifest(records, path)
    manifest = SplitManifest(path, dataset_root=dataset_with_mixes)

    assert list(manifest) == records
    record = manifest.get_split("train")[0]
    assert manifest.resolve_mix_path(record).exists()
    assert record.duration == pytest.approx(0.5, abs=0.01)
    assert record.ai_energy_ratio is None


def test_manifest_with_energy_ratio(dataset_with_mixes):
    tracks = list(ARIADataset(dataset_with_mixes))
    assignment = {t.id: "test" for t in tracks}
    first = tracks[0]
    rel = get_mix_path(first, generate_all_mix_specs(first.num_stems, CODEC)[1]).relative_to(dataset_with_mixes)

    records, skipped = build_manifest(
        tracks, dataset_with_mixes, assignment, CODEC, energy_lookup={rel.as_posix(): 0.25}
    )
    assert len(records) == 1
    assert skipped == 4 + 8 + 16 - 1
    assert records[0].ai_energy_ratio == 0.25

    summary = manifest_summary(records, CODEC, seed=0, train_ratio=0.8, val_ratio=0.1)
    assert summary["splits"]["test"]["num_mixes"] == 1
    assert summary["has_energy_ratio"] is True


def test_loads_legacy_manifest(temp_dir):
    """v2 manifests (ai_energy_ratio, no duration/ai_stem_ratio) still load."""
    line = {
        "track_id": "t", "artist": "a", "track_name": "n", "genre": "g", "codec": CODEC,
        "num_stems": 4, "num_ai_stems": 1, "ai_indices": [2], "ai_energy_ratio": 0.01,
        "split": "val", "mix_path": "t/mixes/encodec_3/U1/mix_S-2.mp3",
    }
    path = temp_dir / "legacy.jsonl"
    path.write_text(json.dumps(line) + "\n")

    record = next(iter(SplitManifest(path)))
    assert record.ai_stem_ratio == 0.25
    assert record.ai_energy_ratio == 0.01
    assert record.duration is None


def test_missing_field_is_clear_error(temp_dir):
    path = temp_dir / "bad.jsonl"
    path.write_text(json.dumps({"track_id": "t"}) + "\n")
    with pytest.raises(ValueError, match="Missing field"):
        SplitManifest(path)
