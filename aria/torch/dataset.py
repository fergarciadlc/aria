"""
PyTorch Dataset for ARIA.

Provides a map-style dataset for training audio deepfake detectors.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from aria.dataset import ARIADataset
from aria.mix import MixSpec, generate_all_mix_specs, get_or_compute_mix
from aria.track import Track


def aria_collate_fn(batch: list[dict]) -> dict:
    """Custom collate function for ARIATorchDataset.
    
    Handles variable-length metadata fields (ai_indices, track_id, codec)
    that can't be automatically stacked into tensors.
    
    Args:
        batch: List of sample dictionaries from __getitem__.
        
    Returns:
        Batched dictionary with:
            - audio: Tensor of shape (B, 2, T)
            - label: Tensor of shape (B,) 
            - track_id: List of strings
            - ai_indices: List of tuples
            - codec: List of strings
    """
    return {
        'audio': torch.stack([item['audio'] for item in batch]),
        'label': torch.tensor([item['label'] for item in batch]),
        'track_id': [item['track_id'] for item in batch],
        'ai_indices': [item['ai_indices'] for item in batch],
        'codec': [item['codec'] for item in batch],
    }


@dataclass
class MixSample:
    """A sample from the dataset.

    Attributes:
        track_id: Track identifier.
        mix_spec: Mix specification.
        label: Label value (interpretation depends on label_mode).
    """

    track_id: str
    mix_spec: MixSpec
    label: int | float


LabelMode = Literal["binary", "count", "ratio"]
MixSampling = Literal["all", "binary", "balanced"]


class ARIATorchDataset(Dataset):
    """PyTorch Dataset for ARIA mixes.

    Generates mixes on-the-fly or loads pre-computed mixes.
    Supports random cropping for training and fixed segments for evaluation.

    Example:
        >>> from torch.utils.data import DataLoader
        >>> dataset = ARIATorchDataset("/path/to/aria", codec="encodec_3")
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in loader:
        ...     audio = batch["audio"]  # (B, 2, T)
        ...     label = batch["label"]  # (B,)

    Attributes:
        aria_dataset: Underlying ARIADataset.
        codec: Codec for AI stems.
        sample_rate: Target sample rate.
        segment_duration: Duration of audio segments in seconds.
    """

    def __init__(
        self,
        path: str | Path,
        codec: str = "encodec_3",
        sample_rate: int = 48000,
        segment_duration: float = 5.0,
        label_mode: LabelMode = "binary",
        mix_sampling: MixSampling = "all",
        eval_mode: bool = False,
        seed: int | None = None,
    ):
        """Initialize the dataset.

        Args:
            path: Path to ARIA dataset.
            codec: Codec for AI-generated stems (e.g., "encodec_3").
            sample_rate: Target sample rate for audio.
            segment_duration: Duration of audio segments in seconds.
            label_mode: How to generate labels:
                - "binary": 0 = all real, 1 = has any AI
                - "count": Number of AI stems (0, 1, 2, ...)
                - "ratio": Fraction of AI stems (0.0 to 1.0)
            mix_sampling: Which mixes to include:
                - "all": All 2^n combinations per track
                - "binary": Only U0 (all real) and Un (all AI)
                - "balanced": Equal samples per U-level
            eval_mode: If True, use deterministic center crops.
            seed: Random seed for reproducibility.
        """
        self.aria_dataset = ARIADataset(path)
        self.codec = codec
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.segment_samples = int(segment_duration * sample_rate)
        self.label_mode = label_mode
        self.mix_sampling = mix_sampling
        self.eval_mode = eval_mode

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Build list of all (track, mix_spec) combinations
        self._samples = self._build_samples()

    def _build_samples(self) -> list[MixSample]:
        """Build list of all samples based on mix_sampling strategy."""
        samples = []

        for track in self.aria_dataset:
            mix_specs = generate_all_mix_specs(track.num_stems, self.codec)

            # Filter based on sampling strategy
            if self.mix_sampling == "binary":
                # Only all-real (U0) and all-AI (Un)
                mix_specs = [
                    s for s in mix_specs
                    if s.is_all_real() or s.is_all_ai(track.num_stems)
                ]
            elif self.mix_sampling == "balanced":
                # Group by num_ai_stems and take equal from each
                # For now, keep all - balancing done via weighted sampling
                pass

            for spec in mix_specs:
                label = self._compute_label(spec, track.num_stems)
                samples.append(MixSample(
                    track_id=track.id,
                    mix_spec=spec,
                    label=label,
                ))

        return samples

    def _compute_label(self, spec: MixSpec, num_stems: int) -> int | float:
        """Compute label based on label_mode."""
        if self.label_mode == "binary":
            return 0 if spec.is_all_real() else 1
        elif self.label_mode == "count":
            return spec.num_ai_stems
        elif self.label_mode == "ratio":
            return spec.num_ai_stems / num_stems if num_stems > 0 else 0.0
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample.

        Returns:
            Dictionary with keys:
                - "audio": Tensor of shape (2, segment_samples)
                - "label": Label value
                - "track_id": Track identifier
                - "ai_indices": Tuple of AI stem indices
                - "codec": Codec name
        """
        sample = self._samples[idx]
        track = self.aria_dataset[sample.track_id]

        # Get or compute mix audio
        audio = get_or_compute_mix(
            track,
            sample.mix_spec,
            target_sr=self.sample_rate,
        )

        # Crop to segment
        audio = self._crop_audio(audio)

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()

        return {
            "audio": audio_tensor,
            "label": sample.label,
            "track_id": sample.track_id,
            "ai_indices": sample.mix_spec.ai_indices,
            "codec": sample.mix_spec.codec,
        }

    def _crop_audio(self, audio: np.ndarray) -> np.ndarray:
        """Crop audio to segment_samples.

        Args:
            audio: Audio array of shape (channels, samples).

        Returns:
            Cropped audio of shape (channels, segment_samples).
        """
        current_samples = audio.shape[-1]

        if current_samples <= self.segment_samples:
            # Pad if too short
            pad_width = ((0, 0), (0, self.segment_samples - current_samples))
            return np.pad(audio, pad_width, mode="constant")

        if self.eval_mode:
            # Center crop for evaluation
            start = (current_samples - self.segment_samples) // 2
        else:
            # Random crop for training
            start = random.randint(0, current_samples - self.segment_samples)

        return audio[:, start : start + self.segment_samples]

    def get_labels(self) -> np.ndarray:
        """Get all labels as numpy array (useful for class weights)."""
        return np.array([s.label for s in self._samples])

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced data (binary mode only).

        Returns:
            Tensor of class weights [weight_0, weight_1].
        """
        if self.label_mode != "binary":
            raise ValueError("Class weights only available for binary label_mode")

        labels = self.get_labels()
        n_samples = len(labels)
        n_positive = labels.sum()
        n_negative = n_samples - n_positive

        # Inverse frequency weighting
        weight_positive = n_samples / (2 * n_positive) if n_positive > 0 else 1.0
        weight_negative = n_samples / (2 * n_negative) if n_negative > 0 else 1.0

        return torch.tensor([weight_negative, weight_positive])


