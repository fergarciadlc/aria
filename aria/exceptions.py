"""
Custom exceptions for the ARIA library.

Keep exceptions simple and descriptive.
"""


class ARIAError(Exception):
    """Base exception for all ARIA library errors."""

    pass


class DatasetNotFoundError(ARIAError):
    """Raised when the dataset path does not exist or is invalid."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Dataset not found at: {path}")


class TrackNotFoundError(ARIAError):
    """Raised when a requested track cannot be found."""

    def __init__(self, identifier: str):
        self.identifier = identifier
        super().__init__(f"Track not found: {identifier}")


class MissingStemError(ARIAError):
    """Raised when a required stem file is missing."""

    def __init__(self, stem_name: str, source: str, track_id: str):
        self.stem_name = stem_name
        self.source = source
        self.track_id = track_id
        super().__init__(
            f"Missing stem '{stem_name}' ({source}) for track {track_id}"
        )


class AudioLoadError(ARIAError):
    """Raised when an audio file cannot be loaded."""

    def __init__(self, path: str, reason: str = ""):
        self.path = path
        self.reason = reason
        msg = f"Failed to load audio: {path}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


