"""
Generators for creating AI-reconstructed stems.

These are used for dataset creation, not for loading. Model dependencies
(torch, transformers) are imported lazily; install the ``generators`` extra.
"""

from aria.generators.encodec import EncodecGenerator
from aria.generators.protocol import AudioGenerator

__all__ = ["AudioGenerator", "EncodecGenerator"]
