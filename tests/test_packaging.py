"""Tests for dataset packaging."""

import pytest

from aria.audio import ffmpeg_available
from aria.mix import generate_all_mix_specs, get_mix_path
from aria.packaging import PackageConfig, package_dataset
from aria.track import load_track
from aria.validate import validate_dataset
from tests.conftest import CODEC, write_audio


def test_package_flac_then_validate(mock_dataset_path, temp_dir):
    out = temp_dir / "pkg"
    stats = package_dataset(mock_dataset_path, out, config=PackageConfig(audio_format="flac"), verbose=False)

    assert stats.errors == []
    assert stats.tracks_processed == 3
    assert stats.stems_copied == 2 * (2 + 3 + 4)
    assert validate_dataset(out).valid


def test_package_copies_precomputed_mixes(mock_dataset_path, temp_dir):
    src = load_track(mock_dataset_path / "track-001")
    for spec in generate_all_mix_specs(src.num_stems, CODEC):
        write_audio(get_mix_path(src, spec, format="wav"), 777)

    out = temp_dir / "pkg"
    config = PackageConfig(num_tracks=1, audio_format="wav", include_mixes=True)
    stats = package_dataset(mock_dataset_path, out, config=config, verbose=False)

    assert stats.errors == []
    assert stats.mixes_generated == 4
    packaged = load_track(out / "track-001")
    import soundfile as sf
    for spec in generate_all_mix_specs(packaged.num_stems, CODEC):
        assert sf.info(get_mix_path(packaged, spec, format="wav")).frames == 777
        assert get_mix_path(packaged, spec, format="wav").with_suffix(".json").exists()


@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not installed")
def test_package_mp3(mock_dataset_path, temp_dir):
    out = temp_dir / "pkg"
    config = PackageConfig(num_tracks=1, audio_format="mp3")
    stats = package_dataset(mock_dataset_path, out, config=config, verbose=False)

    assert stats.errors == []
    assert sorted(p.name for p in (out / "track-001" / "stems" / "real").iterdir()) == ["drums.mp3", "vocals.mp3"]
    assert validate_dataset(out).valid
