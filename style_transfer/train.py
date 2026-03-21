"""Fonctions d'entraînement et d'évaluation pour le transfert de style conditionnel."""

from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from .dataset import Dataset_ImageAndStyle
from .model import StyleTransferNetwork, VGGExtractor, compute_loss


def prepare_styletransfer_modele(
    path_to_data: str | Path,
    batch_size: int = 128,
) -> tuple:
    """Prépare le dataset, le dataloader, le modèle et le device.

    Parameters
    ----------
    path_to_data : str | Path
        Chemin vers le dossier racine du dataset (contient ``10k_img_resized/``
        et ``img_style_resized/``).
    batch_size : int
        Taille des mini-batchs.

    Returns
    -------
    tuple
        (model, dataloader, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Calculs effectués sur: {device}")

    path_to_data = str(path_to_data)
    if not path_to_data.endswith("/"):
        path_to_data += "/"

    dataset = Dataset_ImageAndStyle(path_to_data, transform=transforms.ToTensor())
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    model = StyleTransferNetwork().to(device)
    return model, dataloader, device


def train_model_styletransfer(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lambda_style: float = 1_000_000,
) -> dict:
    """Entraîne le modèle de transfert de style et retourne l'historique.

    Parameters
    ----------
    model        : réseau de transfert de style
    dataloader   : dataloader du dataset contenu/style
    device       : CPU ou CUDA
    epochs       : nombre d'epochs
    lambda_style : poids de la loss de style

    Returns
    -------
    dict
        {"train_loss": list[float]}
    """
    print("Lancement de l'entraînement...")
    history = {"train_loss": []}

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    vgg = VGGExtractor().to(device)

    for epoch in range(epochs):
        run_loss = 0.0
        model.train()

        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for images, styles in loop:
            images = images.to(device)
            styles = styles.to(device)

            optimizer.zero_grad()
            outputs = model(images, styles)
            content_loss, style_loss = compute_loss(vgg, outputs, images, styles)
            loss = content_loss + lambda_style * style_loss
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        t_loss = run_loss / len(dataloader)
        history["train_loss"].append(t_loss)
        print(f"  => Train loss {t_loss:.4f}", flush=True)

    return history


def charger_image_aleatoire(dossier: str | Path) -> tuple[torch.Tensor, str]:
    """Charge une image aléatoire depuis un dossier et la retourne en tenseur.

    Parameters
    ----------
    dossier : str | Path
        Chemin vers le dossier contenant les images.

    Returns
    -------
    tuple[Tensor, str]
        (tenseur (1, 3, H, W), nom_du_fichier)

    Raises
    ------
    FileNotFoundError
        Si aucune image n'est trouvée dans le dossier.
    """
    extensions = (".jpg", ".jpeg", ".png")
    liste_fichiers = [
        f for f in os.listdir(dossier)
        if f.lower().endswith(extensions)
    ]
    if not liste_fichiers:
        raise FileNotFoundError(f"Aucune image trouvée dans le dossier : {dossier}")

    nom_fichier = random.choice(liste_fichiers)
    chemin_complet = os.path.join(dossier, nom_fichier)

    img = Image.open(chemin_complet).convert("RGB")
    img_tensor = transforms.ToTensor()(img)
    return img_tensor.unsqueeze(0), nom_fichier


def preparer_pour_plot(tenseur: torch.Tensor):
    """Convertit un tenseur image en array numpy prêt à l'affichage.

    Parameters
    ----------
    tenseur : Tensor de forme (1, C, H, W) ou (C, H, W)

    Returns
    -------
    numpy.ndarray de forme (H, W, C), valeurs dans [0, 1]
    """
    t = tenseur.squeeze(0).cpu().clamp(0, 1)
    return t.permute(1, 2, 0).numpy()
