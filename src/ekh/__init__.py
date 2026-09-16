"""EKH: consensus RNA secondary structure prediction with an evolutionary
KH-99 grammar and two-pass Gap-Bracket CYK parsing."""

from .model import EKH, Prediction

__all__ = ["EKH", "Prediction"]
__version__ = "1.0.0"
