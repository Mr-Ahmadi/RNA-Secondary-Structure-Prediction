"""Gap-Bracket: consensus RNA secondary structure prediction, pseudoknots
included, with the KH-99 grammar, a phylogenetic column model and two-pass CYK."""

from .model import EKH, Decoding, Prediction
from .parser import PassSettings

__all__ = ["EKH", "Decoding", "PassSettings", "Prediction"]
__version__ = "2.0.0"
