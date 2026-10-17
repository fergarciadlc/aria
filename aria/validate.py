"""
Dataset validation utilities.

Functions to check dataset integrity and completeness.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aria.exceptions import MissingStemError
from aria.track import (
    Track,
    get_available_codecs,
    get_stem_path,
    is_track_dir,
    iter_track_dirs,
    load_track,
)


@dataclass
class ValidationResult:
    """Result of validating a track or dataset.
    
    Attributes:
        valid: True if validation passed.
        errors: List of error messages.
        warnings: List of warning messages.
    """
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def add_error(self, message: str) -> None:
        """Add an error and mark as invalid."""
        self.errors.append(message)
        self.valid = False
    
    def add_warning(self, message: str) -> None:
        """Add a warning (doesn't affect validity)."""
        self.warnings.append(message)
    
    def merge(self, other: "ValidationResult") -> None:
        """Merge another result into this one."""
        if not other.valid:
            self.valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
    
    def __repr__(self) -> str:
        status = "✓ valid" if self.valid else f"✗ {len(self.errors)} errors"
        if self.warnings:
            status += f", {len(self.warnings)} warnings"
        return f"ValidationResult({status})"


def validate_track(
    track: Track,
    check_real_stems: bool = True,
    check_ai_stems: bool = True,
    codecs: list[str] | None = None,
) -> ValidationResult:
    """Validate a single track.
    
    Args:
        track: Track to validate.
        check_real_stems: Check that all real stems exist.
        check_ai_stems: Check that AI stems exist.
        codecs: Specific codecs to check. If None, checks all available.
    
    Returns:
        ValidationResult with any errors/warnings.
    """
    result = ValidationResult()
    
    # Check metadata
    if not track.id:
        result.add_error(f"Track missing ID: {track.path}")
    
    if not track.stem_names:
        result.add_error(f"Track has no stems: {track.display_name}")
        return result  # Can't check stems if none defined
    
    # Check real stems
    if check_real_stems:
        for stem_name in track.stem_names:
            try:
                get_stem_path(track, stem_name, "real")
            except MissingStemError:
                result.add_error(f"Missing real stem '{stem_name}' for {track.display_name}")
    
    # Check AI stems
    if check_ai_stems:
        available = get_available_codecs(track)
        
        if not available and not codecs:
            result.add_warning(f"No AI stems found for {track.display_name}")
        else:
            codecs_to_check = codecs if codecs else available
            
            for codec in codecs_to_check:
                if codec not in available:
                    result.add_error(f"Codec '{codec}' not available for {track.display_name}")
                    continue
                
                for stem_name in track.stem_names:
                    try:
                        get_stem_path(track, stem_name, "ai_generated", codec)
                    except MissingStemError:
                        result.add_error(
                            f"Missing AI stem '{stem_name}' ({codec}) for {track.display_name}"
                        )
    
    return result


def validate_dataset_path(path: Path | str) -> ValidationResult:
    """Validate that a path contains a valid ARIA dataset.
    
    Checks:
    - Path exists
    - Contains at least one track directory
    - Track directories have metadata.json
    
    Args:
        path: Path to dataset root.
    
    Returns:
        ValidationResult with any errors/warnings.
    """
    result = ValidationResult()
    path = Path(path)
    
    if not path.exists():
        result.add_error(f"Dataset path does not exist: {path}")
        return result
    
    if not path.is_dir():
        result.add_error(f"Dataset path is not a directory: {path}")
        return result
    
    if not iter_track_dirs(path):
        result.add_error(f"No track directories found in {path}")
        return result
    
    # Check for common issues
    dirs_without_metadata = [d for d in path.iterdir() if d.is_dir() and not is_track_dir(d)]
    
    for d in dirs_without_metadata:
        if not d.name.startswith("."):  # Ignore hidden dirs
            result.add_warning(f"Directory without metadata.json: {d.name}")
    
    return result


def validate_dataset(
    path: Path | str,
    check_real_stems: bool = True,
    check_ai_stems: bool = True,
    codecs: list[str] | None = None,
    limit: int | None = None,
) -> ValidationResult:
    """Validate an entire ARIA dataset.
    
    Args:
        path: Path to dataset root.
        check_real_stems: Check that all real stems exist.
        check_ai_stems: Check that AI stems exist.
        codecs: Specific codecs to check.
        limit: Maximum number of tracks to check (for quick validation).
    
    Returns:
        ValidationResult with all errors/warnings.
    """
    result = ValidationResult()
    path = Path(path)
    
    # First validate the path itself
    path_result = validate_dataset_path(path)
    result.merge(path_result)
    
    if not result.valid:
        return result
    
    # Find and validate tracks
    track_dirs = iter_track_dirs(path)
    
    if limit is not None:
        track_dirs = track_dirs[:limit]
    
    for track_dir in track_dirs:
        try:
            track = load_track(track_dir)
            track_result = validate_track(
                track, 
                check_real_stems=check_real_stems,
                check_ai_stems=check_ai_stems,
                codecs=codecs,
            )
            result.merge(track_result)
        except Exception as e:
            result.add_error(f"Failed to load track {track_dir.name}: {e}")
    
    return result


def check_dataset_exists(path: Path | str) -> bool:
    """Quick check if a valid dataset exists at path.
    
    Args:
        path: Path to check.
    
    Returns:
        True if path contains at least one valid track.
    """
    path = Path(path)
    
    if not path.exists() or not path.is_dir():
        return False
    
    # Look for at least one track with metadata
    return any(is_track_dir(item) for item in path.iterdir())


def get_dataset_summary(path: Path | str) -> dict[str, Any]:
    """Get a quick summary of a dataset.
    
    Args:
        path: Path to dataset root.
    
    Returns:
        Dictionary with dataset summary info.
    """
    path = Path(path)
    
    if not check_dataset_exists(path):
        return {"exists": False, "path": str(path)}
    
    track_dirs = iter_track_dirs(path)
    
    # Codec folders present in any track
    codecs = sorted({
        d.name
        for track_dir in track_dirs
        if (track_dir / "stems" / "ai_generated").exists()
        for d in (track_dir / "stems" / "ai_generated").iterdir()
        if d.is_dir()
    })
    
    return {
        "exists": True,
        "path": str(path),
        "num_tracks": len(track_dirs),
        "codecs": codecs,
    }


