"""Tests for track module."""

import pytest

from aria.track import Track, load_track, get_stem_path, get_available_codecs, has_all_stems
from aria.exceptions import TrackNotFoundError, MissingStemError


class TestTrack:
    """Tests for Track dataclass."""
    
    def test_track_creation(self):
        """Test creating a Track instance."""
        from pathlib import Path
        
        track = Track(
            id="test-id",
            artist="Test Artist",
            name="Test Song",
            genre="rock",
            stem_names=["vocals", "drums"],
            path=Path("/fake/path"),
        )
        
        assert track.id == "test-id"
        assert track.num_stems == 2
        assert track.display_name == "Test Artist - Test Song"
    
    def test_track_repr(self):
        """Test Track string representation."""
        from pathlib import Path
        
        track = Track(
            id="test",
            artist="Artist",
            name="Song",
            genre="pop",
            stem_names=["a", "b", "c"],
            path=Path("/fake"),
        )
        
        assert "Artist - Song" in repr(track)
        assert "stems=3" in repr(track)


class TestLoadTrack:
    """Tests for load_track function."""
    
    def test_load_track_success(self, mock_track_path):
        """Test loading a valid track."""
        track = load_track(mock_track_path)
        
        assert track.id == "test-track-001"
        assert track.artist == "Test Artist"
        assert track.name == "Test Song"
        assert track.genre == "rock"
        assert track.stem_names == ["vocals", "drums", "bass"]
        assert track.path == mock_track_path
    
    def test_load_track_not_found(self, temp_dir):
        """Test loading from non-existent path."""
        with pytest.raises(TrackNotFoundError):
            load_track(temp_dir / "nonexistent")
    
    def test_load_track_no_metadata(self, temp_dir):
        """Test loading from directory without metadata."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        with pytest.raises(TrackNotFoundError):
            load_track(empty_dir)


class TestGetStemPath:
    """Tests for get_stem_path function."""
    
    def test_get_real_stem_path(self, mock_track_path):
        """Test getting path to real stem."""
        track = load_track(mock_track_path)
        path = get_stem_path(track, "vocals", "real")
        
        assert path.exists()
        assert path.name == "vocals.wav"

    def test_get_real_stem_path_falls_back_to_mp3(self, mock_track_path):
        """Test real stems can be loaded from MP3-backed datasets."""
        wav_path = mock_track_path / "stems" / "real" / "vocals.wav"
        wav_path.unlink()
        (mock_track_path / "stems" / "real" / "vocals.mp3").touch()

        track = load_track(mock_track_path)
        path = get_stem_path(track, "vocals", "real")

        assert path.exists()
        assert path.name == "vocals.mp3"
    
    def test_get_ai_stem_path(self, mock_track_with_ai):
        """Test getting path to AI stem."""
        track = load_track(mock_track_with_ai)
        path = get_stem_path(track, "vocals", "ai_generated", "encodec_3")
        
        assert path.exists()
        assert path.name == "vocals.mp3"
        assert "encodec_3" in str(path)

    def test_get_ai_stem_path_falls_back_to_wav(self, mock_track_path):
        """Test backward compatibility with legacy WAV AI stems."""
        legacy_ai_dir = mock_track_path / "stems" / "ai_generated" / "encodec_3"
        legacy_ai_dir.mkdir(parents=True)
        (legacy_ai_dir / "vocals.wav").touch()

        track = load_track(mock_track_path)
        path = get_stem_path(track, "vocals", "ai_generated", "encodec_3")

        assert path.exists()
        assert path.name == "vocals.wav"
    
    def test_missing_stem_raises(self, mock_track_path):
        """Test that missing stem raises error."""
        track = load_track(mock_track_path)
        
        with pytest.raises(MissingStemError) as exc_info:
            get_stem_path(track, "nonexistent", "real")
        
        assert "nonexistent" in str(exc_info.value)
    
    def test_ai_stem_requires_codec(self, mock_track_path):
        """Test that ai_generated source requires codec."""
        track = load_track(mock_track_path)
        
        with pytest.raises(ValueError, match="codec must be specified"):
            get_stem_path(track, "vocals", "ai_generated")


class TestGetAvailableCodecs:
    """Tests for get_available_codecs function."""
    
    def test_no_ai_stems(self, mock_track_path):
        """Test when no AI stems exist."""
        track = load_track(mock_track_path)
        codecs = get_available_codecs(track)
        
        assert codecs == []
    
    def test_with_ai_stems(self, mock_track_with_ai):
        """Test when AI stems exist."""
        track = load_track(mock_track_with_ai)
        codecs = get_available_codecs(track)
        
        assert codecs == ["encodec_3"]


class TestHasAllStems:
    """Tests for has_all_stems function."""
    
    def test_has_all_real_stems(self, mock_track_path):
        """Test checking all real stems exist."""
        track = load_track(mock_track_path)
        assert has_all_stems(track, "real") is True
    
    def test_has_all_ai_stems(self, mock_track_with_ai):
        """Test checking all AI stems exist."""
        track = load_track(mock_track_with_ai)
        assert has_all_stems(track, "ai_generated", "encodec_3") is True
    
    def test_missing_ai_stems(self, mock_track_path):
        """Test when AI stems don't exist."""
        track = load_track(mock_track_path)
        assert has_all_stems(track, "ai_generated", "encodec_3") is False


