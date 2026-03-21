"""Style Transfer conditionnel via FiLM / Instance Normalisation."""

from .dataset import Dataset_ImageAndStyle
from .model import StyleTransferNetwork, VGGExtractor, get_gram_matrix, compute_loss
from .train import prepare_styletransfer_modele, train_model_styletransfer, charger_image_aleatoire, preparer_pour_plot

__all__ = [
    "Dataset_ImageAndStyle",
    "StyleTransferNetwork",
    "VGGExtractor",
    "get_gram_matrix",
    "compute_loss",
    "prepare_styletransfer_modele",
    "train_model_styletransfer",
    "charger_image_aleatoire",
    "preparer_pour_plot",
]
