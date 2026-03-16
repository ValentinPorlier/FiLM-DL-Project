"""Sort of CLEVR — dataset 2D simplifié pour valider l'architecture FiLM."""

from .dataset import HDF5Dataset, CLASSES, NUM_CLASSES
from .model import SortOfClevrFiLMModel
from .train import train_model, evaluate, evaluate_per_class, visualize_predictions

__all__ = [
    "HDF5Dataset",
    "CLASSES",
    "NUM_CLASSES",
    "SortOfClevrFiLMModel",
    "train_model",
    "evaluate",
    "evaluate_per_class",
    "visualize_predictions",
]
