"""
ARIA - Authentic vs Reconstructed Isolated Audio

A Python library for loading, exploring, and training on the ARIA dataset.
"""

from aria.dataset import ARIADataset
from aria.track import Track
from aria.mix import MixSpec, compute_mix, generate_all_mix_specs
from aria.audio import load_audio, save_audio
from aria.config import Config
from aria.exceptions import (
    ARIAError,
    DatasetNotFoundError,
    TrackNotFoundError,
    MissingStemError,
    AudioLoadError,
)
from aria.validate import (
    check_dataset_exists,
    validate_dataset,
    validate_track,
    get_dataset_summary,
)
from aria.packaging import (
    package_dataset,
    PackageConfig,
    list_tiers,
)

__version__ = "0.1.0"

__all__ = [
    # Main classes
    "ARIADataset",
    "Track",
    "MixSpec",
    "Config",
    # Functions
    "compute_mix",
    "generate_all_mix_specs",
    "load_audio",
    "save_audio",
    # Validation
    "check_dataset_exists",
    "validate_dataset",
    "validate_track",
    "get_dataset_summary",
    # Packaging
    "package_dataset",
    "PackageConfig",
    "list_tiers",
    # Exceptions
    "ARIAError",
    "DatasetNotFoundError",
    "TrackNotFoundError",
    "MissingStemError",
    "AudioLoadError",
]

