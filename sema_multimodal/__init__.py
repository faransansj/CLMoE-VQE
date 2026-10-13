"""Heterogeneous VQA/caption continual learning, separate from archived VQA runs.

Model imports are intentionally lazy: data/scoring and command-line help do not
load BLIP, torchvision, or pretrained assets.
"""

__all__ = ["MultimodalReadout"]


def __getattr__(name):
    if name == "MultimodalReadout":
        from .model import MultimodalReadout
        return MultimodalReadout
    raise AttributeError(name)
