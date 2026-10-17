"""
Encodec-based audio generator.

Wraps the 48 kHz Encodec model from Hugging Face transformers. Requires the
``generators`` extra (torch + transformers).

Processing modes:
    "replicate": audio is fed to the 48 kHz model at its own sample rate
        (44.1 kHz for ARIA) without resampling. This is how the released
        ARIA AI stems were generated; it shifts Encodec's artifact grid by
        44100/48000 (see notebooks/fakeprint_encodec_3kbps.ipynb).
    "corrected": audio is resampled to 48 kHz before the model and back to
        the input rate afterwards.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np

from aria.audio import resample, to_stereo

EncodecMode = Literal["replicate", "corrected"]

MODEL_ID = "facebook/encodec_48khz"


def detect_device() -> str:
    """Pick a torch device for Encodec: CUDA if available, otherwise CPU.

    MPS is skipped on purpose: Encodec quantization has known issues there.
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class EncodecGenerator:
    """Generate AI-reconstructed audio using Encodec.

    One model instance serves every bandwidth: audio is encoded once at the
    highest bandwidth and decoded from a prefix of its codebooks per
    requested bandwidth.

    Example:
        generator = EncodecGenerator()
        outputs = generator.process_multi(audio, sr=44100, bandwidths=[3, 6, 24])
        outputs[3]  # (2, samples) reconstruction at 3 kbps

    Attributes:
        mode: "replicate" or "corrected" (see module docstring).
        device: PyTorch device ("cpu", "cuda").
    """

    SUPPORTED_BANDWIDTHS = (3, 6, 12, 24)
    MODEL_SAMPLE_RATE = 48000

    def __init__(
        self,
        bandwidth: int = 3,
        mode: EncodecMode = "replicate",
        device: str | None = None,
    ):
        """Initialize the Encodec generator.

        Args:
            bandwidth: Default bandwidth in kbps for process(). Default is 3
                (most degraded).
            mode: "replicate" (released ARIA data) or "corrected".
            device: PyTorch device. If None, auto-detects (CUDA, else CPU).
        """
        _check_bandwidths([bandwidth])
        if mode not in ("replicate", "corrected"):
            raise ValueError(f"Unknown mode: {mode}. Use 'replicate' or 'corrected'.")

        self.bandwidth = bandwidth
        self.mode = mode
        self.device = device or detect_device()
        self._model = None
        self._processor = None

    def _load(self) -> None:
        """Lazy-load the Encodec model and processor."""
        if self._model is not None:
            return

        from transformers import AutoProcessor, EncodecModel

        self._model = EncodecModel.from_pretrained(MODEL_ID)
        self._processor = AutoProcessor.from_pretrained(MODEL_ID)
        self._model.eval()
        self._model.to(self.device)

    @property
    def name(self) -> str:
        """Generator identifier."""
        return f"encodec_{self.bandwidth}"

    def process(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Process audio through Encodec at the default bandwidth.

        Args:
            audio: Input audio of shape (channels, samples).
            sr: Sample rate of the input.

        Returns:
            Reconstructed stereo audio of shape (2, samples) at rate sr.
        """
        return self.process_multi(audio, sr, [self.bandwidth])[self.bandwidth]

    def process_multi(
        self,
        audio: np.ndarray,
        sr: int,
        bandwidths: Sequence[int],
    ) -> dict[int, np.ndarray]:
        """Process audio through Encodec at several bandwidths in one pass.

        Args:
            audio: Input audio of shape (channels, samples).
            sr: Sample rate of the input.
            bandwidths: Bandwidths in kbps.

        Returns:
            Mapping bandwidth -> reconstructed stereo audio (2, samples) at
            rate sr, trimmed to the input length.
        """
        import torch

        _check_bandwidths(bandwidths)
        self._load()

        audio = to_stereo(audio.astype(np.float32))
        num_samples = audio.shape[-1]

        model_input = audio
        if self.mode == "corrected":
            model_input = resample(audio, sr, self.MODEL_SAMPLE_RATE)

        # In replicate mode the processor is told the audio is 48 kHz even
        # though it isn't — that is the documented dataset recipe.
        inputs = self._processor(
            raw_audio=model_input,
            sampling_rate=self._processor.sampling_rate,
            return_tensors="pt",
        ).to(self.device)

        outputs = {}
        with torch.no_grad():
            encoded = self._model.encode(
                inputs["input_values"],
                inputs["padding_mask"],
                bandwidth=max(bandwidths),
            )
            for bandwidth in bandwidths:
                # The 48 kHz model uses 2 codebooks per 3 kbps.
                num_codebooks = (bandwidth // 3) * 2
                decoded = self._model.decode(
                    encoded.audio_codes[:, :, :num_codebooks],
                    encoded.audio_scales,
                    inputs["padding_mask"],
                )[0]
                rebuilt = to_stereo(np.atleast_2d(torch.squeeze(decoded).cpu().numpy()))
                if self.mode == "corrected":
                    rebuilt = resample(rebuilt, self.MODEL_SAMPLE_RATE, sr)
                outputs[bandwidth] = rebuilt[:, :num_samples]

        return outputs


def _check_bandwidths(bandwidths: Sequence[int]) -> None:
    unsupported = [b for b in bandwidths if b not in EncodecGenerator.SUPPORTED_BANDWIDTHS]
    if unsupported or not bandwidths:
        raise ValueError(
            f"Unsupported bandwidths: {list(bandwidths)}. "
            f"Choose from: {EncodecGenerator.SUPPORTED_BANDWIDTHS}"
        )
