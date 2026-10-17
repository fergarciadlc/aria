"""Tests for the PyTorch dataset (skipped without torch)."""

import pytest

torch = pytest.importorskip("torch")

from aria.dataset import ARIADataset  # noqa: E402
from aria.mix import generate_all_mix_specs, get_mix_path  # noqa: E402
from aria.splits import SplitManifest, build_manifest, write_manifest  # noqa: E402
from aria.torch import ARIATorchDataset, aria_collate_fn  # noqa: E402
from tests.conftest import CODEC, SR, write_audio  # noqa: E402


def test_on_the_fly_all_mixes(mock_dataset_path):
    ds = ARIATorchDataset(mock_dataset_path, segment_duration=0.25, label_mode="count")
    assert len(ds) == 4 + 8 + 16

    item = ds[0]
    assert item["audio"].shape == (2, int(0.25 * SR))
    batch = aria_collate_fn([ds[i] for i in range(4)])
    assert batch["audio"].shape == (4, 2, int(0.25 * SR))


def test_balanced_one_mix_per_level(mock_dataset_path):
    ds = ARIATorchDataset(mock_dataset_path, mix_sampling="balanced", seed=0)
    # n + 1 levels per track for 2, 3 and 4 stems
    assert len(ds) == 3 + 4 + 5
    assert ARIATorchDataset(mock_dataset_path, mix_sampling="balanced", seed=0).get_labels().tolist() == ds.get_labels().tolist()


def test_unknown_mix_sampling(mock_dataset_path):
    with pytest.raises(ValueError):
        ARIATorchDataset(mock_dataset_path, mix_sampling="nope")


def test_manifest_mode_labels_and_segments(mock_dataset_path, temp_dir):
    tracks = list(ARIADataset(mock_dataset_path))
    for track in tracks:
        for spec in generate_all_mix_specs(track.num_stems, CODEC):
            write_audio(get_mix_path(track, spec), SR)
    records, _ = build_manifest(tracks, mock_dataset_path, {t.id: "train" for t in tracks}, CODEC)
    write_manifest(records, temp_dir / "m.jsonl")
    manifest = SplitManifest(temp_dir / "m.jsonl", dataset_root=mock_dataset_path)

    ds = ARIATorchDataset(
        mock_dataset_path, manifest=manifest, split="train", label_mode="count",
        segment_duration=0.5, sample_rate=48000,
    )
    assert len(ds) == len(records)
    assert sorted(ds.get_labels().tolist()) == sorted(r.num_ai_stems for r in records)
    assert ds[5]["audio"].shape == (2, 24000)

    with pytest.raises(ValueError, match="ai_energy_ratio"):
        ARIATorchDataset(mock_dataset_path, manifest=manifest, split="train", label_mode="energy_ratio")


def test_eval_mode_is_deterministic(mock_dataset_path):
    ds = ARIATorchDataset(mock_dataset_path, segment_duration=0.1, eval_mode=True)
    assert torch.equal(ds[3]["audio"], ds[3]["audio"])
