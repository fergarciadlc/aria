"""
PyTorch Dataset for ARIA.

Provides a map-style dataset for training audio deepfake detectors.
"""

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from aria.audio import resample, to_stereo
from aria.constants import DEFAULT_CODEC, SAMPLE_RATE
from aria.dataset import ARIADataset
from aria.mix import MixSpec, generate_all_mix_specs, get_or_compute_mix
from aria.splits import ManifestRecord, SplitManifest


def aria_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
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
        mix_path: Absolute path to a pre-computed mix file. When set,
            audio is loaded directly from this path instead of computing
            on-the-fly from stems.
    """

    track_id: str
    mix_spec: MixSpec
    label: int | float
    mix_path: Path | None = field(default=None)


LabelMode = Literal["binary", "count", "ratio", "energy_ratio"]
MixSampling = Literal["all", "binary", "balanced"]


class ARIATorchDataset(Dataset):
    """PyTorch Dataset for ARIA mixes.

    Generates mixes on-the-fly or loads pre-computed mixes.
    Supports random cropping for training and fixed segments for evaluation.

    Random crops draw from torch's RNG, which DataLoader seeds differently in
    every worker; call torch.manual_seed() for reproducible crops.

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
        codec: str = DEFAULT_CODEC,
        sample_rate: int = SAMPLE_RATE,
        segment_duration: float = 5.0,
        label_mode: LabelMode = "binary",
        mix_sampling: MixSampling = "all",
        eval_mode: bool = False,
        seed: int | None = None,
        manifest: SplitManifest | None = None,
        split: str | None = None,
    ):
        """Initialize the dataset.

        Args:
            path: Path to ARIA dataset.
            codec: Codec for AI-generated stems (e.g., "encodec_3"). In
                manifest mode, only records of this codec are used.
            sample_rate: Target sample rate for audio. Files at another
                rate are resampled on load.
            segment_duration: Duration of audio segments in seconds.
            label_mode: How to generate labels:
                - "binary": 0 = all real, 1 = has any AI
                - "count": Number of AI stems (0, 1, 2, ...)
                - "ratio": Fraction of AI stems (0.0 to 1.0)
                - "energy_ratio": Energy share of AI stems (manifest mode
                  only; the manifest must include ai_energy_ratio)
            mix_sampling: Which mixes to include (only used when manifest
                is not provided):
                - "all": All 2^n combinations per track
                - "binary": Only U0 (all real) and Un (all AI)
                - "balanced": One random mix per U-level per track
            eval_mode: If True, use deterministic center crops.
            seed: Seed for "balanced" mix selection.
            manifest: Optional SplitManifest. When provided, samples are
                loaded directly from the manifest (with pre-computed mix
                paths) rather than discovered from the dataset directory.
                mix_sampling is ignored in this mode.
            split: Split partition to use from the manifest ("train", "val",
                or "test"). Required when manifest is provided.
        """
        self.path = Path(path)
        self.codec = codec
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.segment_samples = int(segment_duration * sample_rate)
        self.label_mode = label_mode
        self.mix_sampling = mix_sampling
        self.eval_mode = eval_mode

        if manifest is not None and split is None:
            raise ValueError("split must be specified when manifest is provided (e.g. 'train').")

        if label_mode == "energy_ratio" and manifest is None:
            raise ValueError("label_mode 'energy_ratio' requires a manifest.")

        # Only needed to compute mixes on the fly (non-manifest mode).
        self.aria_dataset: ARIADataset | None = None

        if manifest is not None:
            self._samples = self._build_samples_from_manifest(manifest, split)
        else:
            self.aria_dataset = ARIADataset(self.path)
            self._samples = self._build_samples(random.Random(seed))

    def _build_samples_from_manifest(
        self, manifest: SplitManifest, split: str
    ) -> list[MixSample]:
        """Build samples from a split manifest.

        Each ManifestRecord maps 1-to-1 to a MixSample with an explicit
        mix_path, so audio is loaded directly without on-the-fly computation.
        """
        records = [r for r in manifest.get_split(split) if r.codec == self.codec]
        samples = []

        for record in records:
            mix_spec = MixSpec(ai_indices=record.ai_indices, codec=record.codec)
            label = self._compute_label_from_record(record)
            mix_path = manifest.resolve_mix_path(record)
            samples.append(MixSample(
                track_id=record.track_id,
                mix_spec=mix_spec,
                label=label,
                mix_path=mix_path,
            ))

        return samples

    def _compute_label_from_record(self, record: ManifestRecord) -> int | float:
        """Compute label from a manifest record."""
        if self.label_mode == "energy_ratio":
            if record.ai_energy_ratio is None:
                raise ValueError(
                    f"Manifest record {record.mix_path} has no ai_energy_ratio; "
                    "use a manifest created with a metrics CSV."
                )
            return record.ai_energy_ratio
        spec = MixSpec(ai_indices=record.ai_indices, codec=record.codec)
        return self._compute_label(spec, record.num_stems)

    def _build_samples(self, rng: random.Random) -> list[MixSample]:
        """Build list of all samples based on mix_sampling strategy."""
        assert self.aria_dataset is not None
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
                # One random mix per number of AI stems (U-level)
                by_level: dict[int, list[MixSpec]] = defaultdict(list)
                for s in mix_specs:
                    by_level[s.num_ai_stems].append(s)
                mix_specs = [rng.choice(by_level[k]) for k in sorted(by_level)]
            elif self.mix_sampling != "all":
                raise ValueError(f"Unknown mix_sampling: {self.mix_sampling}")

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

    def __getitem__(self, idx: int) -> dict[str, Any]:
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

        # Load audio: read just the segment from the pre-computed file
        # (manifest mode), otherwise compute the mix on-the-fly from stems.
        if sample.mix_path is not None:
            audio = self._read_segment(sample.mix_path)
        else:
            assert self.aria_dataset is not None
            track = self.aria_dataset[sample.track_id]
            audio = get_or_compute_mix(
                track,
                sample.mix_spec,
                target_sr=self.sample_rate,
            )
            audio = self._crop_audio(to_stereo(audio))

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()

        return {
            "audio": audio_tensor,
            "label": sample.label,
            "track_id": sample.track_id,
            "ai_indices": sample.mix_spec.ai_indices,
            "codec": sample.mix_spec.codec,
        }

    def _crop_start(self, num_samples: int, segment_samples: int) -> int:
        """Start of the crop window: centred in eval mode, random otherwise."""
        if num_samples <= segment_samples:
            return 0
        if self.eval_mode:
            return (num_samples - segment_samples) // 2
        return int(torch.randint(0, num_samples - segment_samples + 1, ()).item())

    def _read_segment(self, path: Path) -> np.ndarray:
        """Read one cropped segment from an audio file without decoding it all.

        Returns:
            Stereo audio of shape (2, segment_samples) at self.sample_rate.
        """
        info = sf.info(path)
        # Segment length in the file's own sample rate
        file_segment = math.ceil(self.segment_samples * info.samplerate / self.sample_rate)
        start = self._crop_start(info.frames, file_segment)

        audio, sr = sf.read(
            path, start=start, frames=file_segment, always_2d=True, dtype="float32"
        )
        audio = to_stereo(audio.T)
        if sr != self.sample_rate:
            audio = resample(audio, sr, self.sample_rate)
        return self._crop_audio(audio)

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

        start = self._crop_start(current_samples, self.segment_samples)
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


