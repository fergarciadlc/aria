"""Tests for the dataset contract shared by the library and the scripts:
stem order, mix naming/lookup, sample rates, MP3 output."""

import json

import numpy as np
import pytest
import soundfile as sf

from aria.audio import ffmpeg_available, load_audio, save_audio, to_stereo
from aria.mix import (
    MixSpec,
    compute_mix,
    find_mix_path,
    get_or_compute_mix,
    mix_metadata,
)
from aria.track import iter_track_dirs, load_track
from tests.conftest import CODEC, write_audio


class TestStemOrder:
    def test_indices_follow_sorted_stem_names(self, mock_track_with_ai):
        """mix_S-0 must make the alphabetically first stem AI (as generate_mixes did)."""
        track = load_track(mock_track_with_ai)
        assert track.stem_names == ["bass", "drums", "vocals"]

        spec = MixSpec(ai_indices=(0,), codec=CODEC)
        mix = compute_mix(track, spec)

        real = {n: load_audio(mock_track_with_ai / "stems" / "real" / f"{n}.wav")[0] for n in track.stem_names}
        ai_bass, _ = load_audio(mock_track_with_ai / "stems" / "ai_generated" / CODEC / "bass.mp3")
        expected = ai_bass + real["drums"] + real["vocals"]
        np.testing.assert_allclose(mix, expected, atol=1e-6)

    def test_mix_metadata_matches_track(self, mock_track_with_ai):
        track = load_track(mock_track_with_ai)
        meta = mix_metadata(track, MixSpec(ai_indices=(2, 0), codec=CODEC))
        assert meta == {
            "mix_name": "mix_S-0-2",
            "num_stems": 3,
            "num_ai_stems": 2,
            "ai_stem_indices": [0, 2],
            "ai_codec": CODEC,
            "stem_names": ["bass", "drums", "vocals"],
        }
        json.dumps(meta)


class TestMixSpecFromFilename:
    @pytest.mark.parametrize(
        "name,indices",
        [("mix_S-empty.mp3", ()), ("mix_S-0-2.wav", (0, 2)), ("mix_S-3", (3,))],
    )
    def test_roundtrip(self, name, indices):
        spec = MixSpec.from_filename(name, CODEC)
        assert spec.ai_indices == indices
        assert spec.filename == name.split(".")[0]

    def test_rejects_other_files(self):
        with pytest.raises(ValueError):
            MixSpec.from_filename("vocals.mp3", CODEC)


class TestPrecomputedMixes:
    def test_find_mix_path_any_format(self, mock_track_with_ai):
        track = load_track(mock_track_with_ai)
        spec = MixSpec(ai_indices=(1,), codec=CODEC)
        assert find_mix_path(track, spec) is None

        path = track.path / "mixes" / CODEC / "U1" / "mix_S-1.mp3"
        write_audio(path, 1000)
        assert find_mix_path(track, spec) == path

    def test_get_or_compute_prefers_precomputed(self, mock_track_with_ai):
        track = load_track(mock_track_with_ai)
        spec = MixSpec(ai_indices=(1,), codec=CODEC)
        write_audio(track.path / "mixes" / CODEC / "U1" / "mix_S-1.wav", 1234)

        audio = get_or_compute_mix(track, spec)
        assert audio.shape[-1] == 1234


class TestSampleRates:
    def test_compute_mix_resamples_to_common_rate(self, mock_track_with_ai):
        """A 48 kHz real stem mixed with 44.1 kHz AI stems must be resampled."""
        track = load_track(mock_track_with_ai)
        write_audio(track.path / "stems" / "real" / "drums.wav", 48000, sr=48000)

        mix = compute_mix(track, MixSpec(ai_indices=(0, 2), codec=CODEC), target_sr=44100)
        assert mix.shape == (2, 44100)

    def test_compute_mix_mono_stem(self, mock_track_with_ai):
        track = load_track(mock_track_with_ai)
        sf.write(track.path / "stems" / "real" / "drums.wav", np.zeros(44100, dtype=np.float32), 44100)
        mix = compute_mix(track, MixSpec(ai_indices=(), codec=CODEC))
        assert mix.shape[0] == 2

    def test_to_stereo(self):
        assert to_stereo(np.zeros((1, 10))).shape == (2, 10)
        assert to_stereo(np.zeros((4, 10))).shape == (2, 10)


class TestSaveAudioFormats:
    def test_format_must_match_extension(self, temp_dir):
        with pytest.raises(ValueError):
            save_audio(temp_dir / "a.wav", np.zeros((2, 10)), 44100, format="flac")

    @pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
    def test_save_mp3(self, temp_dir):
        audio = (np.random.default_rng(0).standard_normal((2, 44100)) * 0.1).astype(np.float32)
        path = temp_dir / "out.mp3"
        save_audio(path, audio, 44100, mp3_bitrate=256)

        loaded, sr = load_audio(path)
        assert sr == 44100
        assert loaded.shape == (2, 44100)


def test_iter_track_dirs(mock_dataset_path):
    (mock_dataset_path / "not-a-track").mkdir()
    dirs = iter_track_dirs(mock_dataset_path)
    assert [d.name for d in dirs] == ["track-001", "track-002", "track-003"]
