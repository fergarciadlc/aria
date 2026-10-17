"""Tests for dataset module."""

import pytest

from aria.dataset import ARIADataset
from aria.exceptions import DatasetNotFoundError, TrackNotFoundError


class TestARIADataset:
    """Tests for ARIADataset class."""
    
    def test_load_dataset(self, mock_dataset_path):
        """Test loading a dataset."""
        dataset = ARIADataset(mock_dataset_path)
        
        assert len(dataset) == 3
    
    def test_load_nonexistent_raises(self, temp_dir):
        """Test loading from non-existent path raises."""
        with pytest.raises(DatasetNotFoundError):
            ARIADataset(temp_dir / "nonexistent")
    
    def test_load_empty_raises(self, temp_dir):
        """Test loading empty directory raises."""
        with pytest.raises(DatasetNotFoundError, match="no tracks found"):
            ARIADataset(temp_dir)
    
    def test_getitem_by_index(self, mock_dataset_path):
        """Test accessing track by index."""
        dataset = ARIADataset(mock_dataset_path)
        track = dataset[0]
        
        assert track is not None
        assert track.id is not None
    
    def test_getitem_by_id(self, mock_dataset_path):
        """Test accessing track by ID."""
        dataset = ARIADataset(mock_dataset_path)
        
        # Get first track's ID
        first_track = dataset[0]
        
        # Access by ID
        same_track = dataset[first_track.id]
        
        assert same_track.id == first_track.id
    
    def test_getitem_invalid_id_raises(self, mock_dataset_path):
        """Test accessing with invalid ID raises."""
        dataset = ARIADataset(mock_dataset_path)
        
        with pytest.raises(TrackNotFoundError):
            dataset["nonexistent-id"]
    
    def test_getitem_index_out_of_range(self, mock_dataset_path):
        """Test accessing with out-of-range index raises."""
        dataset = ARIADataset(mock_dataset_path)
        
        with pytest.raises(IndexError):
            dataset[100]
    
    def test_iteration(self, mock_dataset_path):
        """Test iterating over dataset."""
        dataset = ARIADataset(mock_dataset_path)
        
        tracks = list(dataset)
        
        assert len(tracks) == 3
    
    def test_filter_by_genre(self, mock_dataset_path):
        """Test filtering by genre."""
        dataset = ARIADataset(mock_dataset_path)
        
        rock_tracks = dataset.filter(genre="rock")
        
        assert len(rock_tracks) == 2
        assert all("rock" in t.genre.lower() for t in rock_tracks)
    
    def test_filter_by_min_stems(self, mock_dataset_path):
        """Test filtering by minimum stems."""
        dataset = ARIADataset(mock_dataset_path)
        
        # Track with 4 stems
        tracks = dataset.filter(min_stems=4)
        
        assert len(tracks) == 1
        assert tracks[0].num_stems == 4
    
    def test_filter_by_max_stems(self, mock_dataset_path):
        """Test filtering by maximum stems."""
        dataset = ARIADataset(mock_dataset_path)
        
        tracks = dataset.filter(max_stems=2)
        
        assert len(tracks) == 1
        assert tracks[0].num_stems == 2
    
    def test_get_by_name(self, mock_dataset_path):
        """Test finding track by name."""
        dataset = ARIADataset(mock_dataset_path)
        
        track = dataset.get_by_name("Artist A")
        
        assert track is not None
        assert "Artist A" in track.display_name
    
    def test_get_by_name_not_found(self, mock_dataset_path):
        """Test get_by_name returns None when not found."""
        dataset = ARIADataset(mock_dataset_path)
        
        track = dataset.get_by_name("Nonexistent Artist")
        
        assert track is None
    
    def test_genres_property(self, mock_dataset_path):
        """Test genres property."""
        dataset = ARIADataset(mock_dataset_path)
        
        genres = dataset.genres
        
        assert "rock" in genres
        assert "pop" in genres
    
    def test_stats(self, mock_dataset_path):
        """Test stats method."""
        dataset = ARIADataset(mock_dataset_path)
        
        stats = dataset.stats()
        
        assert stats["num_tracks"] == 3
        assert stats["total_stems"] == 9  # 2 + 3 + 4
        assert "genres" in stats
    
    def test_repr(self, mock_dataset_path):
        """Test string representation."""
        dataset = ARIADataset(mock_dataset_path)
        
        repr_str = repr(dataset)
        
        assert "ARIADataset" in repr_str
        assert "tracks=3" in repr_str


