"""Tests for mix module."""

import numpy as np
import pytest

from aria.mix import (
    MixSpec,
    generate_all_mix_specs,
    compute_mix,
    get_mix_path,
)
from aria.track import load_track


class TestMixSpec:
    """Tests for MixSpec dataclass."""
    
    def test_mixspec_creation(self):
        """Test creating a MixSpec."""
        spec = MixSpec(ai_indices=(0, 2), codec="encodec_3")
        
        assert spec.ai_indices == (0, 2)
        assert spec.codec == "encodec_3"
        assert spec.num_ai_stems == 2
    
    def test_mixspec_u_folder(self):
        """Test U folder naming."""
        assert MixSpec(ai_indices=(), codec="e").u_folder == "U0"
        assert MixSpec(ai_indices=(0,), codec="e").u_folder == "U1"
        assert MixSpec(ai_indices=(0, 1, 2), codec="e").u_folder == "U3"
    
    def test_mixspec_filename_empty(self):
        """Test filename for all-real mix."""
        spec = MixSpec(ai_indices=(), codec="encodec_3")
        assert spec.filename == "mix_S-empty"
    
    def test_mixspec_filename_single(self):
        """Test filename with single AI stem."""
        spec = MixSpec(ai_indices=(2,), codec="encodec_3")
        assert spec.filename == "mix_S-2"
    
    def test_mixspec_filename_multiple(self):
        """Test filename with multiple AI stems."""
        spec = MixSpec(ai_indices=(0, 2, 3), codec="encodec_3")
        assert spec.filename == "mix_S-0-2-3"
    
    def test_mixspec_is_all_real(self):
        """Test is_all_real check."""
        assert MixSpec(ai_indices=(), codec="e").is_all_real() is True
        assert MixSpec(ai_indices=(0,), codec="e").is_all_real() is False
    
    def test_mixspec_is_all_ai(self):
        """Test is_all_ai check."""
        assert MixSpec(ai_indices=(0, 1, 2), codec="e").is_all_ai(3) is True
        assert MixSpec(ai_indices=(0, 1), codec="e").is_all_ai(3) is False
    
    def test_mixspec_frozen(self):
        """Test that MixSpec is immutable."""
        spec = MixSpec(ai_indices=(0,), codec="e")
        
        with pytest.raises(AttributeError):
            spec.codec = "other"


class TestGenerateAllMixSpecs:
    """Tests for generate_all_mix_specs function."""
    
    def test_generate_2_stems(self):
        """Test generating specs for 2 stems."""
        specs = generate_all_mix_specs(2, "encodec_3")
        
        # 2^2 = 4 combinations
        assert len(specs) == 4
        
        ai_indices = [s.ai_indices for s in specs]
        assert () in ai_indices  # All real
        assert (0,) in ai_indices
        assert (1,) in ai_indices
        assert (0, 1) in ai_indices  # All AI
    
    def test_generate_4_stems(self):
        """Test generating specs for 4 stems."""
        specs = generate_all_mix_specs(4, "encodec_3")
        
        # 2^4 = 16 combinations
        assert len(specs) == 16
    
    def test_generate_0_stems(self):
        """Test generating specs for 0 stems."""
        specs = generate_all_mix_specs(0, "encodec_3")
        
        # Only 1 combination: empty
        assert len(specs) == 1
        assert specs[0].ai_indices == ()


class TestComputeMix:
    """Tests for compute_mix function."""
    
    def test_compute_all_real_mix(self, mock_track_with_ai):
        """Test computing all-real mix."""
        track = load_track(mock_track_with_ai)
        spec = MixSpec(ai_indices=(), codec="encodec_3")
        
        audio = compute_mix(track, spec)
        
        assert audio.shape[0] == 2  # Stereo
        assert audio.shape[1] > 0
    
    def test_compute_all_ai_mix(self, mock_track_with_ai):
        """Test computing all-AI mix."""
        track = load_track(mock_track_with_ai)
        spec = MixSpec(ai_indices=(0, 1, 2), codec="encodec_3")
        
        audio = compute_mix(track, spec)
        
        assert audio.shape[0] == 2
    
    def test_compute_mixed_mix(self, mock_track_with_ai):
        """Test computing mix with some real, some AI."""
        track = load_track(mock_track_with_ai)
        spec = MixSpec(ai_indices=(1,), codec="encodec_3")
        
        audio = compute_mix(track, spec)
        
        assert audio.shape[0] == 2


class TestGetMixPath:
    """Tests for get_mix_path function."""
    
    def test_get_mix_path_format(self, mock_track_with_ai):
        """Test mix path format."""
        track = load_track(mock_track_with_ai)
        spec = MixSpec(ai_indices=(0, 2), codec="encodec_3")
        
        path = get_mix_path(track, spec)
        
        assert "mixes" in str(path)
        assert "encodec_3" in str(path)
        assert "U2" in str(path)
        assert "mix_S-0-2.wav" in str(path)


