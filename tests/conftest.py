"""
Pytest fixtures for ARIA tests.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_track_path(temp_dir):
    """Create a mock track directory with metadata and stems."""
    track_id = "test-track-001"
    track_path = temp_dir / track_id
    track_path.mkdir()
    
    # Create metadata
    metadata = {
        "id": track_id,
        "artist": "Test Artist",
        "name": "Test Song",
        "genre": "rock",
        "provider": "test",
        "stems": ["vocals", "drums", "bass"],
        "number_of_stems": 3,
    }
    
    with open(track_path / "metadata.json", "w") as f:
        json.dump(metadata, f)
    
    # Create real stems directory
    real_stems_path = track_path / "stems" / "real"
    real_stems_path.mkdir(parents=True)
    
    # Create mock audio files (short, 1 second at 48kHz)
    sr = 48000
    duration = 1.0
    samples = int(sr * duration)
    
    for stem_name in metadata["stems"]:
        audio = np.random.randn(2, samples).astype(np.float32) * 0.1
        sf.write(real_stems_path / f"{stem_name}.wav", audio.T, sr)
    
    return track_path


@pytest.fixture
def mock_track_with_ai(mock_track_path):
    """Create a mock track with AI-generated stems."""
    # Create AI stems directory
    ai_stems_path = mock_track_path / "stems" / "ai_generated" / "encodec_3"
    ai_stems_path.mkdir(parents=True)
    
    # Create mock AI audio files
    sr = 48000
    samples = int(sr * 1.0)
    
    for stem_name in ["vocals", "drums", "bass"]:
        audio = np.random.randn(2, samples).astype(np.float32) * 0.1
        sf.write(ai_stems_path / f"{stem_name}.wav", audio.T, sr)
    
    return mock_track_path


@pytest.fixture
def mock_dataset_path(temp_dir):
    """Create a mock dataset with multiple tracks."""
    tracks_data = [
        {"id": "track-001", "artist": "Artist A", "name": "Song 1", "genre": "rock", "stems": ["vocals", "drums"]},
        {"id": "track-002", "artist": "Artist B", "name": "Song 2", "genre": "pop", "stems": ["vocals", "drums", "bass"]},
        {"id": "track-003", "artist": "Artist C", "name": "Song 3", "genre": "rock", "stems": ["vocals", "drums", "bass", "guitar"]},
    ]
    
    sr = 48000
    samples = int(sr * 0.5)  # Short samples for speed
    
    for track_data in tracks_data:
        track_path = temp_dir / track_data["id"]
        track_path.mkdir()
        
        # Metadata
        metadata = {
            **track_data,
            "provider": "test",
            "number_of_stems": len(track_data["stems"]),
        }
        with open(track_path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        # Real stems
        real_path = track_path / "stems" / "real"
        real_path.mkdir(parents=True)
        
        for stem_name in track_data["stems"]:
            audio = np.random.randn(2, samples).astype(np.float32) * 0.1
            sf.write(real_path / f"{stem_name}.wav", audio.T, sr)
        
        # AI stems
        ai_path = track_path / "stems" / "ai_generated" / "encodec_3"
        ai_path.mkdir(parents=True)
        
        for stem_name in track_data["stems"]:
            audio = np.random.randn(2, samples).astype(np.float32) * 0.1
            sf.write(ai_path / f"{stem_name}.wav", audio.T, sr)
    
    return temp_dir


