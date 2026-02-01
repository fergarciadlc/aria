"""
Protocol for audio generators.

Using Protocol (structural subtyping) instead of ABC for flexibility.
"""

from typing import Protocol

import numpy as np


class AudioGenerator(Protocol):
    """Protocol for AI audio generators.

    Implementations must provide:
    - process(): Transform audio through the generator
    - name: Identifier for the generator (e.g., "encodec_3")

    Example implementation:
        class MyGenerator:
            def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
                return some_transformation(audio)

            @property
            def name(self) -> str:
                return "my_generator"
    """

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Process audio through the generator.

        Args:
            audio: Input audio array of shape (channels, samples).
            sr: Sample rate of the input audio.

        Returns:
            Processed audio array of shape (channels, samples).
        """
        ...

    @property
    def name(self) -> str:
        """Identifier for this generator (e.g., 'encodec_3')."""
        ...


