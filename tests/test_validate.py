"""Tests for validate module."""

import pytest

from aria.validate import (
    ValidationResult,
    validate_track,
    validate_dataset_path,
    validate_dataset,
    check_dataset_exists,
    get_dataset_summary,
)
from aria.track import load_track


class TestValidationResult:
    """Tests for ValidationResult dataclass."""
    
    def test_initial_state(self):
        """Test initial state is valid."""
        result = ValidationResult()
        
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []
    
    def test_add_error(self):
        """Test adding an error."""
        result = ValidationResult()
        result.add_error("Something went wrong")
        
        assert result.valid is False
        assert len(result.errors) == 1
    
    def test_add_warning(self):
        """Test adding a warning doesn't affect validity."""
        result = ValidationResult()
        result.add_warning("Minor issue")
        
        assert result.valid is True
        assert len(result.warnings) == 1
    
    def test_merge(self):
        """Test merging results."""
        result1 = ValidationResult()
        result1.add_error("Error 1")
        
        result2 = ValidationResult()
        result2.add_warning("Warning 1")
        
        result1.merge(result2)
        
        assert result1.valid is False
        assert len(result1.errors) == 1
        assert len(result1.warnings) == 1


class TestValidateTrack:
    """Tests for validate_track function."""
    
    def test_valid_track(self, mock_track_path):
        """Test validating a valid track."""
        track = load_track(mock_track_path)
        result = validate_track(track, check_ai_stems=False)
        
        assert result.valid is True
    
    def test_track_with_ai_stems(self, mock_track_with_ai):
        """Test validating track with AI stems."""
        track = load_track(mock_track_with_ai)
        result = validate_track(track)
        
        assert result.valid is True
    
    def test_track_missing_ai_stems(self, mock_track_path):
        """Test track without AI stems gets warning."""
        track = load_track(mock_track_path)
        result = validate_track(track, check_ai_stems=True)
        
        # Should have warning about no AI stems
        assert len(result.warnings) > 0


class TestValidateDatasetPath:
    """Tests for validate_dataset_path function."""
    
    def test_valid_path(self, mock_dataset_path):
        """Test validating a valid dataset path."""
        result = validate_dataset_path(mock_dataset_path)
        
        assert result.valid is True
    
    def test_nonexistent_path(self, temp_dir):
        """Test validating non-existent path."""
        result = validate_dataset_path(temp_dir / "nonexistent")
        
        assert result.valid is False
        assert any("does not exist" in e for e in result.errors)
    
    def test_empty_directory(self, temp_dir):
        """Test validating empty directory."""
        result = validate_dataset_path(temp_dir)
        
        assert result.valid is False
        assert any("No track directories" in e for e in result.errors)


class TestValidateDataset:
    """Tests for validate_dataset function."""
    
    def test_valid_dataset(self, mock_dataset_path):
        """Test validating a valid dataset."""
        result = validate_dataset(mock_dataset_path)
        
        assert result.valid is True
    
    def test_validate_with_limit(self, mock_dataset_path):
        """Test validating with track limit."""
        result = validate_dataset(mock_dataset_path, limit=1)
        
        # Should still be valid, just checked fewer tracks
        assert result.valid is True


class TestCheckDatasetExists:
    """Tests for check_dataset_exists function."""
    
    def test_exists(self, mock_dataset_path):
        """Test checking existing dataset."""
        assert check_dataset_exists(mock_dataset_path) is True
    
    def test_not_exists(self, temp_dir):
        """Test checking non-existent path."""
        assert check_dataset_exists(temp_dir / "nonexistent") is False
    
    def test_empty_directory(self, temp_dir):
        """Test checking empty directory."""
        assert check_dataset_exists(temp_dir) is False


class TestGetDatasetSummary:
    """Tests for get_dataset_summary function."""
    
    def test_valid_dataset(self, mock_dataset_path):
        """Test getting summary of valid dataset."""
        summary = get_dataset_summary(mock_dataset_path)
        
        assert summary["exists"] is True
        assert summary["num_tracks"] == 3
        assert "encodec_3" in summary["codecs"]
    
    def test_nonexistent_dataset(self, temp_dir):
        """Test getting summary of non-existent dataset."""
        summary = get_dataset_summary(temp_dir / "nonexistent")
        
        assert summary["exists"] is False


