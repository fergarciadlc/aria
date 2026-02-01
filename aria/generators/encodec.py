"""
Encodec-based audio generator.

Wraps the existing Encodec implementation from ae_models.
"""

import numpy as np
import torch

# Type hint for Encodec model (avoid import at module level for flexibility)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ae_models.encodec import Encodec


class EncodecGenerator:
    """Generate AI-reconstructed audio using Encodec.

    This wraps the existing Encodec model from ae_models.

    Example:
        generator = EncodecGenerator(bandwidth=3, device="cuda")
        reconstructed = generator.process(audio, sr=48000)

    Attributes:
        bandwidth: Encodec bandwidth in kbps (3, 6, 12, or 24).
        device: PyTorch device ("cpu", "cuda", "mps").
    """

    SUPPORTED_BANDWIDTHS = (3, 6, 12, 24)
    ENCODEC_SAMPLE_RATE = 48000

    def __init__(
        self,
        bandwidth: int = 3,
        device: str | None = None,
    ):
        """Initialize the Encodec generator.

        Args:
            bandwidth: Target bandwidth in kbps. Default is 3 (most degraded).
            device: PyTorch device. If None, auto-detects (prefers CUDA, falls back to CPU).
        """
        if bandwidth not in self.SUPPORTED_BANDWIDTHS:
            raise ValueError(
                f"Unsupported bandwidth: {bandwidth}. "
                f"Choose from: {self.SUPPORTED_BANDWIDTHS}"
            )

        self.bandwidth = bandwidth
        self.device = device or self._detect_device()
        self._model: "Encodec | None" = None

    @staticmethod
    def _detect_device() -> str:
        """Detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        # Note: MPS has known issues with Encodec quantization
        return "cpu"

    @property
    def model(self) -> "Encodec":
        """Lazy-load the Encodec model."""
        if self._model is None:
            from ae_models.encodec import Encodec

            self._model = Encodec(
                bandwidth=self.bandwidth,
                sr=self.ENCODEC_SAMPLE_RATE,
                device=self.device,
            )
        return self._model

    @property
    def name(self) -> str:
        """Generator identifier."""
        return f"encodec_{self.bandwidth}"

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Process audio through Encodec.

        Args:
            audio: Input audio of shape (channels, samples).
            sr: Sample rate (must be 48000 for Encodec).

        Returns:
            Reconstructed audio of shape (channels, samples).

        Raises:
            ValueError: If sample rate is not 48000.
        """
        if sr != self.ENCODEC_SAMPLE_RATE:
            raise ValueError(
                f"Encodec requires {self.ENCODEC_SAMPLE_RATE}Hz, got {sr}Hz. "
                "Resample before calling process()."
            )

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()

        # Handle mono by duplicating to stereo
        if audio_tensor.dim() == 1:
            audio_tensor = torch.stack([audio_tensor, audio_tensor], dim=0)

        # Process through Encodec
        with torch.no_grad():
            reconstructed = self.model.autoencode(audio_tensor)
            reconstructed = reconstructed.squeeze(0).cpu().numpy()

        return reconstructed


