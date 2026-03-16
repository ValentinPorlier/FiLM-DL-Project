"""Dataset HDF5 pour Sort of CLEVR.

Chaque échantillon contient :
- une image 2D RGB (H×W×3) stockée dans un fichier HDF5
- une question en texte brut (string)
- un encodage numérique de la question (vecteur float de taille 10)
- une réponse parmi les 11 classes prédéfinies
"""

from __future__ import annotations

import ast

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ── Vocabulaire des réponses ───────────────────────────────────────────────
CLASSES: list[str] = [
    "right", "blue", "circle", "left", "bottom",
    "yellow", "square", "green", "red", "top", "gray",
]
NUM_CLASSES: int = len(CLASSES)
_CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


class HDF5Dataset(Dataset):
    """Dataset Sort of CLEVR chargé depuis des fichiers HDF5 + CSV.

    Les encodings et labels sont pré-calculés à l'init pour éviter
    ast.literal_eval et list.index à chaque sample (gain x10 sur le chargement).

    Parameters
    ----------
    h5_path      : chemin vers le fichier HDF5 contenant les images
    dataset_name : nom du dataset dans le fichier HDF5
    csv_path     : chemin vers le CSV contenant questions, encodings, réponses
    transform    : transformation optionnelle sur les images (non utilisé par défaut)
    """

    def __init__(
        self,
        h5_path: str,
        dataset_name: str,
        csv_path: str,
        transform=None,
    ) -> None:
        self.h5_path      = h5_path
        self.dataset_name = dataset_name
        self.transform    = transform

        df = pd.read_csv(csv_path)
        self.length = len(df)

        # Pré-cache des questions (liste de strings)
        self.questions = df["question"].tolist()

        # Pré-cache des encodings → numpy array float32 (N, 10)
        encs = df["encoding"].tolist()
        parsed = [ast.literal_eval(e) if isinstance(e, str) else e for e in encs]
        self.encodings = torch.tensor(np.array(parsed, dtype=np.float32))  # (N, 10)

        # Pré-cache des labels → LongTensor (N,)
        self.labels = torch.tensor(
            [_CLASS_TO_IDX[a] for a in df["answer"].tolist()],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return self.length

    def _open_hdf5(self) -> None:
        """Ouverture paresseuse du fichier HDF5 (compatible multiprocess)."""
        self._hf     = h5py.File(self.h5_path, "r")
        self._ds_img = self._hf[self.dataset_name]

    def __getitem__(self, index: int):
        """Retourne (question_str, image, label, encoding).

        Returns
        -------
        question_str : str
        image        : FloatTensor (3, H, W) normalisé dans [0, 1]
        label        : LongTensor  scalaire (indice de classe)
        encoding     : FloatTensor (10,) encodage numérique de la question
        """
        if not hasattr(self, "_hf"):
            self._open_hdf5()

        image = torch.from_numpy(self._ds_img[index]).permute(2, 0, 1).float() / 255.0

        return self.questions[index], image, self.labels[index], self.encodings[index]
