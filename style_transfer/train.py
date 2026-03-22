"""Fonctions d'entraînement et d'évaluation pour le transfert de style conditionnel."""


import random
from pathlib import Path

import torch
from PIL import Image
from stqdm import stqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import Dataset_ImageAndStyle
from .model import StyleTransferNetwork, VGGExtractor, compute_loss


def prepare_styletransfer_modele(
    path_to_data: str | Path,
    batch_size: int = 8,
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
    lr: float = 1e-3,
    lambda_style: float = 1_000_000,
    st_container=None,
) -> dict:
    """Entraîne le modèle de transfert de style et retourne l'historique.

    Parameters
    ----------
    model        : réseau de transfert de style
    dataloader   : dataloader du dataset contenu/style
    device       : CPU ou CUDA
    epochs       : nombre d'epochs
    lr           : learning rate pour Adam
    lambda_style : poids de la loss de style
    st_container : conteneur Streamlit pour la barre de progression (optionnel)

    Returns
    -------
    dict
        {"train_loss": list[float]}
    """
    history = {"train_loss": []}

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    vgg = VGGExtractor().to(device)

    for epoch in range(epochs):
        run_loss = 0.0
        model.train()

        loop = stqdm(dataloader, st_container=st_container,
                     backend=False, frontend=True,
                     desc=f"Epoch {epoch + 1}/{epochs}")
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

    return history


def charger_image_aleatoire(
    dossier: str | Path,
    style: str | None = None,
) -> tuple[torch.Tensor, str]:
    """Charge une image aléatoire depuis un dossier, avec filtre optionnel par style.

    Parameters
    ----------
    dossier : str | Path
        Chemin vers le dossier contenant les images.
    style : str | None
        Préfixe de style pour filtrer les images (ex. ``"baroque"``).
        Si None, toutes les images sont candidates.

    Returns
    -------
    tuple[Tensor, str]
        (tenseur (1, 3, H, W), nom_du_fichier)

    Raises
    ------
    FileNotFoundError
        Si aucune image correspondante n'est trouvée.
    """
    dossier = Path(dossier)
    extensions = {".jpg", ".jpeg", ".png"}
    pattern = f"{style}*" if style else "*"
    liste_fichiers = [
        f for f in dossier.glob(pattern)
        if f.suffix.lower() in extensions
    ]

    if not liste_fichiers:
        raise FileNotFoundError(f"Aucune image trouvée dans : {dossier}")

    chemin_image = random.choice(liste_fichiers)
    img = Image.open(chemin_image).convert("RGB")
    img_tensor = transforms.ToTensor()(img)
    return img_tensor.unsqueeze(0), chemin_image.name


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


if __name__ == "__main__":
    pass
