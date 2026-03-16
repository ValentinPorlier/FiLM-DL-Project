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

        # Charge toutes les images en RAM d'un coup (beaucoup plus rapide que lire une par une)
        print(f"Chargement des images en RAM ({h5_path})...", flush=True)
        with h5py.File(h5_path, "r") as f:
            images_np = f[dataset_name][:]  # (N, H, W, 3) uint8
        self.images = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        return self.questions[index], self.images[index], self.labels[index], self.encodings[index]
