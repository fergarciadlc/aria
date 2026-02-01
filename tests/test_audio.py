"""Tests for audio module."""

import numpy as np
import pytest

from aria.audio import load_audio, save_audio, pad_to_length, sum_stems, resample
from aria.exceptions import AudioLoadError


class TestLoadAudio:
    """Tests for load_audio function."""
    
    def test_load_audio_success(self, mock_track_path):
        """Test loading an audio file."""
        audio_path = mock_track_path / "stems" / "real" / "vocals.wav"
        audio, sr = load_audio(audio_path)
        
        assert audio.shape[0] == 2  # Stereo
        assert sr == 48000
    
    def test_load_audio_not_found(self, temp_dir):
        """Test loading non-existent file."""
        with pytest.raises(AudioLoadError, match="does not exist"):
            load_audio(temp_dir / "nonexistent.wav")
    
    def test_load_audio_with_resample(self, mock_track_path, temp_dir):
        """Test loading with resampling."""
        import soundfile as sf
        
        # Create a 44100 Hz file
        audio_44k = np.random.randn(2, 44100).astype(np.float32)
        path_44k = temp_dir / "audio_44k.wav"
        sf.write(path_44k, audio_44k.T, 44100)
        
        # Load with resample to 48000
        audio, sr = load_audio(path_44k, target_sr=48000)
        
        assert sr == 48000
        assert audio.shape[-1] == 48000  # Resampled length


class TestSaveAudio:
    """Tests for save_audio function."""
    
    def test_save_audio_wav(self, temp_dir):
        """Test saving as WAV."""
        audio = np.random.randn(2, 1000).astype(np.float32)
        path = temp_dir / "test.wav"
        
        save_audio(path, audio, 48000, format="wav")
        
        assert path.exists()
        
        # Verify can be loaded back
        loaded, sr = load_audio(path)
        assert sr == 48000
        assert loaded.shape == audio.shape
    
    def test_save_audio_flac(self, temp_dir):
        """Test saving as FLAC."""
        audio = np.random.randn(2, 1000).astype(np.float32)
        path = temp_dir / "test.flac"
        
        save_audio(path, audio, 48000, format="flac")
        
        assert path.exists()
    
    def test_save_creates_directory(self, temp_dir):
        """Test that save creates parent directories."""
        audio = np.random.randn(2, 1000).astype(np.float32)
        path = temp_dir / "nested" / "dir" / "test.wav"
        
        save_audio(path, audio, 48000)
        
        assert path.exists()


class TestPadToLength:
    """Tests for pad_to_length function."""
    
    def test_pad_shorter(self):
        """Test padding shorter audio."""
        audio = np.ones((2, 100))
        padded = pad_to_length(audio, 200)
        
        assert padded.shape == (2, 200)
        assert np.all(padded[:, :100] == 1)
        assert np.all(padded[:, 100:] == 0)
    
    def test_pad_same_length(self):
        """Test padding when already correct length."""
        audio = np.ones((2, 100))
        padded = pad_to_length(audio, 100)
        
        assert padded.shape == (2, 100)
        np.testing.assert_array_equal(padded, audio)
    
    def test_pad_longer(self):
        """Test padding when audio is longer (no change)."""
        audio = np.ones((2, 200))
        padded = pad_to_length(audio, 100)
        
        assert padded.shape == (2, 200)


class TestSumStems:
    """Tests for sum_stems function."""
    
    def test_sum_equal_length(self):
        """Test summing stems of equal length."""
        stem1 = np.ones((2, 100))
        stem2 = np.ones((2, 100)) * 2
        
        mixed = sum_stems([stem1, stem2])
        
        assert mixed.shape == (2, 100)
        np.testing.assert_array_equal(mixed, np.ones((2, 100)) * 3)
    
    def test_sum_different_lengths(self):
        """Test summing stems of different lengths."""
        stem1 = np.ones((2, 100))
        stem2 = np.ones((2, 150))
        
        mixed = sum_stems([stem1, stem2])
        
        assert mixed.shape == (2, 150)
        # First 100 samples should be 2, rest should be 1
        np.testing.assert_array_equal(mixed[:, :100], np.ones((2, 100)) * 2)
        np.testing.assert_array_equal(mixed[:, 100:], np.ones((2, 50)))
    
    def test_sum_empty_raises(self):
        """Test that summing empty list raises."""
        with pytest.raises(ValueError, match="empty"):
            sum_stems([])


class TestResample:
    """Tests for resample function."""
    
    def test_resample_upsample(self):
        """Test upsampling."""
        audio = np.random.randn(2, 44100).astype(np.float32)
        resampled = resample(audio, 44100, 48000)
        
        assert resampled.shape == (2, 48000)
    
    def test_resample_downsample(self):
        """Test downsampling."""
        audio = np.random.randn(2, 48000).astype(np.float32)
        resampled = resample(audio, 48000, 44100)
        
        assert resampled.shape == (2, 44100)
    
    def test_resample_same_rate(self):
        """Test no resampling when rates match."""
        audio = np.random.randn(2, 1000).astype(np.float32)
        resampled = resample(audio, 48000, 48000)
        
        np.testing.assert_array_equal(resampled, audio)


