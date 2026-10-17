"""Tests for reusing an existing track split and for order-independent splitting."""

import json
import random
from types import SimpleNamespace

import pytest

from aria.splits import (
    assign_reference_splits,
    read_track_splits,
    split_tracks,
    stratified_split_by_num_stems,
)


def fake_tracks(n: int = 60) -> list[SimpleNamespace]:
    """Tracks with 3 stem counts; ids sort differently from creation order."""
    return [
        SimpleNamespace(id=f"{(i * 7) % n:03d}-track", num_stems=3 + i % 3)
        for i in range(n)
    ]


def write_lines(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


class TestSplitTracks:
    def test_same_split_for_any_input_order(self):
        tracks = fake_tracks()
        shuffled = list(tracks)
        random.Random(0).shuffle(shuffled)

        assert split_tracks(tracks, 0.8, 0.1, 42) == split_tracks(shuffled, 0.8, 0.1, 42)

    def test_is_stratified_split_over_sorted_ids(self):
        tracks = fake_tracks()
        ordered = sorted(tracks, key=lambda t: t.id)
        assert split_tracks(tracks, 0.8, 0.1, 7) == stratified_split_by_num_stems(
            ordered, 0.8, 0.1, 7
        )

    def test_plain_stratified_split_does_depend_on_order(self):
        """Why split_tracks exists, and why the paper split must be reused."""
        tracks = fake_tracks()
        reverse = list(reversed(tracks))
        assert stratified_split_by_num_stems(tracks, 0.8, 0.1, 42) != (
            stratified_split_by_num_stems(reverse, 0.8, 0.1, 42)
        )


class TestReadTrackSplits:
    def test_reads_per_mix_and_window_manifests(self, tmp_path):
        per_mix = write_lines(tmp_path / "a.jsonl", [
            {"track_id": "t1", "split": "train", "mix_path": "x"},
            {"track_id": "t1", "split": "train", "mix_path": "y"},
            {"track_id": "t2", "split": "val", "mix_path": "z"},
        ])
        windowed = write_lines(tmp_path / "b.jsonl", [
            {"track_id": "t3", "split": "test", "window_start_sec": 0.0},
            {"track_id": "t3", "split": "test", "window_start_sec": 5.0},
        ])
        assert read_track_splits(per_mix) == {"t1": "train", "t2": "val"}
        assert read_track_splits(windowed) == {"t3": "test"}

    def test_track_in_two_splits_is_an_error(self, tmp_path):
        path = write_lines(tmp_path / "c.jsonl", [
            {"track_id": "t1", "split": "train"},
            {"track_id": "t1", "split": "test"},
        ])
        with pytest.raises(ValueError, match="both 'train' and 'test'"):
            read_track_splits(path)

    def test_unknown_split_name_is_an_error(self, tmp_path):
        path = write_lines(tmp_path / "d.jsonl", [{"track_id": "t1", "split": "dev"}])
        with pytest.raises(ValueError, match="Unknown split"):
            read_track_splits(path)

    def test_missing_field_and_missing_file(self, tmp_path):
        path = write_lines(tmp_path / "e.jsonl", [{"track_id": "t1"}])
        with pytest.raises(ValueError, match="Missing field"):
            read_track_splits(path)
        with pytest.raises(FileNotFoundError):
            read_track_splits(tmp_path / "nope.jsonl")


class TestAssignReferenceSplits:
    def test_uses_reference_instead_of_resplitting(self):
        tracks = fake_tracks(6)
        reference = {t.id: "test" for t in tracks}
        assert assign_reference_splits(tracks, reference) == reference

    def test_track_missing_from_reference_is_an_error(self):
        tracks = fake_tracks(6)
        reference = {t.id: "train" for t in tracks[1:]}
        with pytest.raises(ValueError, match=r"1 track\(s\) are not in the reference"):
            assign_reference_splits(tracks, reference)

    def test_allow_missing_leaves_the_track_out(self):
        tracks = fake_tracks(6)
        reference = {t.id: "train" for t in tracks[1:]}
        assignment = assign_reference_splits(tracks, reference, allow_missing=True)
        assert tracks[0].id not in assignment
        assert len(assignment) == 5

    def test_extra_reference_tracks_are_ignored(self):
        tracks = fake_tracks(6)
        reference = {t.id: "val" for t in tracks} | {"ghost": "train"}
        assert set(assign_reference_splits(tracks, reference)) == {t.id for t in tracks}
