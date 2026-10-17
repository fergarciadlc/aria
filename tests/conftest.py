"""
Pytest fixtures for ARIA tests.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SR = 44100
CODEC = "encodec_3"


def write_audio(path: Path, num_samples: int, sr: int = SR, seed: int = 0) -> None:
    """Write short stereo noise; MP3 is encoded by libsndfile."""
    rng = np.random.default_rng(seed)
    audio = (rng.standard_normal((2, num_samples)) * 0.1).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.T, sr)


def make_track(
    root: Path,
    track_id: str,
    stems: list[str],
    num_samples: int,
    artist: str = "Test Artist",
    name: str = "Test Song",
    genre: str = "rock",
) -> Path:
    """Create a track with real WAV stems and MP3 AI stems."""
    track_path = root / track_id
    track_path.mkdir()

    metadata = {
        "id": track_id,
        "artist": artist,
        "name": name,
        "genre": genre,
        "provider": "test",
        "stems": stems,
        "number_of_stems": len(stems),
    }
    with open(track_path / "metadata.json", "w") as f:
        json.dump(metadata, f)

    for i, stem_name in enumerate(stems):
        write_audio(track_path / "stems" / "real" / f"{stem_name}.wav", num_samples, seed=i)
        write_audio(
            track_path / "stems" / "ai_generated" / CODEC / f"{stem_name}.mp3",
            num_samples,
            seed=100 + i,
        )

    return track_path


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory (alias of pytest's tmp_path)."""
    return tmp_path


@pytest.fixture
def mock_track_path(temp_dir):
    """Create a mock track directory with metadata and real stems."""
    track_path = make_track(temp_dir, "test-track-001", ["vocals", "drums", "bass"], SR)
    # Real stems only; mock_track_with_ai adds the AI stems back.
    for path in (track_path / "stems" / "ai_generated" / CODEC).iterdir():
        path.unlink()
    (track_path / "stems" / "ai_generated" / CODEC).rmdir()
    (track_path / "stems" / "ai_generated").rmdir()
    return track_path


@pytest.fixture
def mock_track_with_ai(mock_track_path):
    """Create a mock track with MP3 AI-generated stems."""
    for i, stem_name in enumerate(["vocals", "drums", "bass"]):
        write_audio(
            mock_track_path / "stems" / "ai_generated" / CODEC / f"{stem_name}.mp3",
            SR,
            seed=100 + i,
        )
    return mock_track_path


@pytest.fixture
def mock_dataset_path(temp_dir):
    """Create a mock dataset with multiple tracks."""
    tracks_data = [
        {"track_id": "track-001", "artist": "Artist A", "name": "Song 1", "genre": "rock", "stems": ["vocals", "drums"]},
        {"track_id": "track-002", "artist": "Artist B", "name": "Song 2", "genre": "pop", "stems": ["vocals", "drums", "bass"]},
        {"track_id": "track-003", "artist": "Artist C", "name": "Song 3", "genre": "rock", "stems": ["vocals", "drums", "bass", "guitar"]},
    ]

    for track_data in tracks_data:
        make_track(temp_dir, num_samples=SR // 2, **track_data)

    return temp_dir
