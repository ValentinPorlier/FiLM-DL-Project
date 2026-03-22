"""Dataset pour le transfert de style conditionnel.

Fournit Dataset_ImageAndStyle qui associe chaque image de contenu à une
image de style choisie aléatoirement dans le dossier de styles.
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets


class Dataset_ImageAndStyle(Dataset):
    """Dataset associant images de contenu et images de style.

    Parameters
    ----------
    path_to_data : str
        Chemin vers le dossier racine contenant ``10k_img_resized/``
        et ``img_style_resized/``.
    transform : callable
        Transformation appliquée à chaque image.
    max_samples : int | None
        Nombre maximum d'images de contenu à utiliser.
    """

    def __init__(
        self,
        path_to_data: str,
        transform,
        max_samples: int | None = None,
    ) -> None:
        image_path = path_to_data + "10k_img_resized"
        style_path = path_to_data + "img_style_resized"

        dataset = datasets.ImageFolder(root=image_path, transform=transform)
        self.image = Subset(dataset, range(max_samples)) if max_samples is not None else dataset
        self.style_image = datasets.ImageFolder(root=style_path, transform=transform)

    def __len__(self) -> int:
        """Retourne le nombre d'images de contenu."""
        return len(self.image)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retourne (image_contenu, image_style) — le style est choisi aléatoirement.

        Parameters
        ----------
        idx : int
            Index de l'image de contenu.

        Returns
        -------
        tuple[Tensor, Tensor]
            (image_contenu, image_style)
        """
        img_content, _ = self.image[idx]
        random_style_idx = random.randint(0, len(self.style_image) - 1)
        image_style, _ = self.style_image[random_style_idx]
        return img_content, image_style


if __name__ == "__main__":
    pass
