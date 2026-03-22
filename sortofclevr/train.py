"""Fonctions d'entraînement et d'évaluation pour Sort of CLEVR."""


import queue as _queue
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from stqdm import stqdm
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import CLASSES, HDF5Dataset, NUM_CLASSES
from .model import SortOfClevrFiLMModel


def _progress_bar(iterable, *, st_container=None, **kwargs):
    """Renvoie une barre de progression adaptée au contexte d'exécution.

    Utilise stqdm si un conteneur Streamlit est fourni, sinon tqdm classique.

    Parameters
    ----------
    iterable : iterable
        L'itérable à parcourir.
    st_container : streamlit.container ou None
        Si fourni, la barre s'affiche dans l'interface Streamlit.
        Si None, la barre s'affiche dans le terminal.
    **kwargs
        Arguments supplémentaires passés à stqdm ou tqdm (desc, leave, etc.).

    Returns
    -------
    stqdm ou tqdm
        Barre de progression encapsulant l'itérable.
    """
    if st_container is not None:
        return stqdm(
            iterable,
            st_container=st_container,
            backend=False,
            frontend=True,
            **kwargs,
        )
    return tqdm(iterable, **kwargs)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epochs: int = 10,
    progress_queue: _queue.Queue | None = None,
    pretrain: bool = False,
    st_container: Any | None = None,
) -> dict:
    """Entraîne le modèle et évalue sur la validation à chaque epoch.

    Si ``pretrain`` est True, charge les poids pré-entraînés et retourne
    l'historique figé sans relancer l'entraînement.

    Si ``progress_queue`` est fourni, envoie un dict par epoch :
    ``{"epoch", "num_epochs", "train_loss", "train_acc", "val_loss", "val_acc"}``
    et à la fin : ``{"done": True, "history": dict}``.

    Parameters
    ----------
    model : torch.nn.Module
        Le modèle à entraîner.
    train_loader : DataLoader
        Dataloader d'entraînement.
    val_loader : DataLoader
        Dataloader de validation.
    optimizer : torch.optim.Optimizer
        Optimiseur utilisé pour la descente de gradient.
    criterion : torch.nn.Module
        Fonction de perte (CrossEntropyLoss par défaut).
    device : torch.device
        Appareil de calcul (CPU ou GPU).
    epochs : int
        Nombre d'epochs d'entraînement.
    progress_queue : queue.Queue ou None
        File partagée avec le thread Streamlit pour les mises à jour.
    pretrain : bool
        Si True, charge les poids pré-entraînés et retourne l'historique sans entraîner.
    st_container : Any ou None
        Conteneur Streamlit pour afficher la barre de progression.

    Returns
    -------
    dict
        Historique avec les clés ``"train_loss"``, ``"train_acc"``,
        ``"val_loss"``, ``"val_acc"`` (listes de longueur ``epochs``).
    """
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    if pretrain:
        state_dict = torch.load(
            "sortofclevr/data/model_weights.pth",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.eval()

        history["train_loss"] = [
            1.0875507179838027, 0.5851487347679417, 0.2778746431326344,
            0.21704500456796075, 0.16917431289262144, 0.15382466041041118,
            0.13014081577314948, 0.12260489029823428, 0.11155957037950084,
            0.09872226716175567,
        ]
        history["train_acc"] = [
            0.4119, 0.6985571428571429, 0.8555285714285714, 0.8947,
            0.9220571428571429, 0.9289285714285714, 0.9404714285714286,
            0.9454, 0.9495142857142858, 0.9574571428571429,
        ]
        history["val_loss"] = [
            1.1361560940742492, 0.5641029372811317, 0.3086692146956921,
            0.2231459490954876, 0.19102550223469733, 0.1957884855568409,
            0.16398447044193745, 0.154434834420681, 0.13628026992082595,
            0.132127396017313,
        ]
        history["val_acc"] = [
            0.4549, 0.7708, 0.8555, 0.8975, 0.9092,
            0.9096, 0.9286, 0.9304, 0.9391, 0.9421,
        ]

        if progress_queue is not None:
            progress_queue.put({
                "mode": "pretrained_loaded",
                "message": "Modèle pré-entraîné chargé.",
            })

        return history

    for epoch in range(epochs):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        batch_loop = _progress_bar(
            train_loader,
            st_container=st_container,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False,
        )

        for _, images, labels, encodings in batch_loop:
            images = images.to(device)
            encodings = encodings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, encodings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            total += labels.size(0)
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()

            batch_loop.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / total:.1%}",
            )

        t_loss = run_loss / len(train_loader)
        t_acc = correct / total
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        if progress_queue is not None:
            progress_queue.put({
                "epoch": epoch + 1,
                "num_epochs": epochs,
                "train_loss": t_loss,
                "train_acc": t_acc,
                "val_loss": v_loss,
                "val_acc": v_acc,
            })

    if progress_queue is not None:
        progress_queue.put({"done": True, "history": history})

    return history


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Calcule la loss moyenne et l'accuracy sur un dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        Le modèle à évaluer (mis en mode eval).
    dataloader : DataLoader
        Dataloader sur lequel effectuer l'évaluation.
    criterion : torch.nn.Module
        Fonction de perte pour calculer la loss.
    device : torch.device
        Appareil de calcul (CPU ou GPU).

    Returns
    -------
    tuple[float, float]
        (loss_moyenne, accuracy) sur l'ensemble du dataloader.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for _, images, labels, encodings in dataloader:
            images = images.to(device)
            encodings = encodings.to(device)
            labels = labels.to(device)
            outputs = model(images, encodings)
            total_loss += criterion(outputs, labels).item()
            correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def evaluate_per_class(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    progress_queue: _queue.Queue | None = None,
    st_container: Any | None = None,
) -> dict[str, float]:
    """Calcule l'accuracy pour chaque classe de réponse.

    Parameters
    ----------
    model : torch.nn.Module
        Le modèle à évaluer (mis en mode eval).
    dataloader : DataLoader
        Dataloader sur lequel effectuer l'évaluation.
    device : torch.device
        Appareil de calcul (CPU ou GPU).
    progress_queue : queue.Queue ou None
        File partagée avec le thread Streamlit pour les mises à jour de batch.
    st_container : Any ou None
        Conteneur Streamlit pour afficher la barre de progression.

    Returns
    -------
    dict[str, float]
        Dictionnaire ``{nom_classe: accuracy}`` pour chaque classe présente
        dans le dataloader.
    """
    model.eval()
    correct_pred = {c: 0 for c in CLASSES}
    total_pred = {c: 0 for c in CLASSES}

    with torch.no_grad():
        loop = _progress_bar(
            enumerate(dataloader, start=1),
            st_container=st_container,
            total=len(dataloader),
            desc="Évaluation",
        )

        for batch_i, (_, images, labels, encodings) in loop:
            images = images.to(device)
            encodings = encodings.to(device)
            labels = labels.to(device)
            preds = torch.argmax(model(images, encodings), dim=1)

            if progress_queue is not None:
                progress_queue.put({
                    "batch": batch_i,
                    "batch_tot": len(dataloader),
                })

            for target, pred in zip(labels, preds):
                cls = CLASSES[target.item()]
                total_pred[cls] += 1
                if target == pred:
                    correct_pred[cls] += 1

    return {
        cls: correct_pred[cls] / total_pred[cls]
        for cls in CLASSES
        if total_pred[cls] > 0
    }


def prepare_objects(
    train_h5: str | Path,
    train_csv: str | Path,
    val_h5: str | Path,
    val_csv: str | Path,
    test_h5: str | Path,
    test_csv: str | Path,
    batch_size: int = 128,
    max_samples: int | None = None,
) -> tuple:
    """Prépare les datasets, les dataloaders, le modèle et le device.

    Parameters
    ----------
    train_h5 : str ou Path
        Chemin vers le fichier HDF5 d'entraînement.
    train_csv : str ou Path
        Chemin vers le CSV d'entraînement (questions, encodings, réponses).
    val_h5 : str ou Path
        Chemin vers le fichier HDF5 de validation.
    val_csv : str ou Path
        Chemin vers le CSV de validation.
    test_h5 : str ou Path
        Chemin vers le fichier HDF5 de test.
    test_csv : str ou Path
        Chemin vers le CSV de test.
    batch_size : int
        Taille des mini-batchs pour les dataloaders.
    max_samples : int ou None
        Nombre maximum de samples pour les datasets train et test.
        Si None, tous les samples sont utilisés.

    Returns
    -------
    tuple
        (model, train_loader, val_loader, test_loader, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = HDF5Dataset(
        str(train_h5),
        "data_train",
        str(train_csv),
        max_samples=max_samples)
    val_ds = HDF5Dataset(str(val_h5), "data_val", str(val_csv))
    test_ds = HDF5Dataset(
        str(test_h5),
        "data_test",
        str(test_csv),
        max_samples=max_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)

    model = SortOfClevrFiLMModel(num_answers=NUM_CLASSES).to(device)

    return model, train_loader, val_loader, test_loader, device


def run(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    lr: float = 0.001,
    epochs: int = 10,
    pretrain: bool = False,
    progress_queue: _queue.Queue | None = None,
    st_container: Any | None = None,
) -> tuple[dict, dict]:
    """Lance un entraînement complet et retourne l'historique et les résultats par classe.

    C'est la fonction principale appelée depuis la page Streamlit dans un thread
    séparé. Elle enchaîne l'entraînement et l'évaluation par classe, puis envoie
    les résultats dans la file partagée.

    Parameters
    ----------
    model : torch.nn.Module
        Le modèle à entraîner.
    train_loader : DataLoader
        Dataloader d'entraînement.
    val_loader : DataLoader
        Dataloader de validation.
    test_loader : DataLoader
        Dataloader de test (utilisé pour l'évaluation par classe).
    device : torch.device
        Appareil de calcul (CPU ou GPU).
    lr : float
        Learning rate pour l'optimiseur Adam.
    epochs : int
        Nombre d'epochs d'entraînement.
    pretrain : bool
        Si True, charge les poids pré-entraînés au lieu d'entraîner.
    progress_queue : queue.Queue ou None
        File partagée pour envoyer les mises à jour au thread Streamlit.
    st_container : Any ou None
        Conteneur Streamlit pour les barres de progression.

    Returns
    -------
    tuple[dict, dict]
        (history, per_class) — historique d'entraînement et accuracy par classe.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = train_model(
        model, train_loader, val_loader, optimizer, criterion, device,
        epochs=epochs, pretrain=pretrain,
        progress_queue=progress_queue, st_container=st_container,
    )
    per_class = evaluate_per_class(
        model, test_loader, device,
        progress_queue=progress_queue, st_container=st_container,
    )

    if progress_queue is not None:
        progress_queue.put({"history": history, "per_class": per_class})

    return history, per_class


def display_image(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple:
    """Retourne une image de test avec ses questions et encodings uniques.

    Prend le premier batch du dataloader, choisit une image au hasard,
    puis déduplique les questions pour n'en garder qu'une par type.

    Parameters
    ----------
    model : torch.nn.Module
        Le modèle évalué (mis en mode eval).
    test_loader : DataLoader
        Dataloader de test depuis lequel piocher l'image.
    device : torch.device
        Appareil de calcul (CPU ou GPU).

    Returns
    -------
    tuple
        (image, questions_uniques, encodings_uniques) où :

        - image : Tensor de forme (3, H, W)
        - questions_uniques : array de strings (questions sans doublons)
        - encodings_uniques : Tensor de forme (N_questions, 10)
    """
    model.eval()
    with torch.no_grad():
        questions, images, labels, encs = next(iter(test_loader))
        images = images.to(device)
        encs = encs.to(device)
        labels = labels.to(device)

        id_img = torch.randint(0, images.size(0), (1,)).item()
        image = images[id_img]

        questions_unique, qst_ind = np.unique(questions, return_index=True)
        encs_unique = encs[qst_ind]

        return image, questions_unique, encs_unique


if __name__ == "__main__":
    pass
