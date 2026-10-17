"""
PyTorch integration for ARIA dataset.
"""

from aria.torch.dataset import ARIATorchDataset, aria_collate_fn

__all__ = ["ARIATorchDataset", "aria_collate_fn"]


