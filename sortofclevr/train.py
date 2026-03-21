"""Fonctions d'entraînement et d'évaluation pour Sort of CLEVR."""

from __future__ import annotations

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
    """Utilise stqdm si un conteneur Streamlit est fourni, sinon fallback sur tqdm."""
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
    """Entraîne le modèle, évalue sur val à chaque epoch, retourne l'historique.

    Si progress_queue est fourni, envoie un dict par epoch :
        {"epoch": int, "num_epochs": int, "train_loss", "train_acc", "val_loss", "val_acc"}
    Et à la fin :
        {"done": True, "history": dict}
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

    loop = _progress_bar(
        range(epochs),
        st_container=st_container,
        desc="Entraînement",
    )

    for epoch in loop:
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        batch_loop = _progress_bar(
            train_loader,
            st_container=st_container,
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False,
        )

        for _, images, labels, encodings in batch_loop:
            images    = images.to(device)
            encodings = encodings.to(device)
            labels    = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, encodings)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            total    += labels.size(0)
            correct  += (torch.argmax(outputs, dim=1) == labels).sum().item()

            batch_loop.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / total:.1%}",
            )

        t_loss = run_loss / len(train_loader)
        t_acc  = correct / total
        print("  Evaluation...", flush=True)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        print(f"  => Train {t_acc:.2%} | Val {v_acc:.2%}", flush=True)

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
    """Retourne (loss_moyenne, accuracy) sur le dataloader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for _, images, labels, encodings in dataloader:
            images    = images.to(device)
            encodings = encodings.to(device)
            labels    = labels.to(device)
            outputs   = model(images, encodings)
            total_loss += criterion(outputs, labels).item()
            correct    += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(dataloader), correct / total


def evaluate_per_class(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    progress_queue: _queue.Queue | None = None,
    st_container: Any | None = None,
) -> dict[str, float]:
    """Retourne {classe: accuracy} pour chaque classe."""
    model.eval()
    correct_pred = {c: 0 for c in CLASSES}
    total_pred   = {c: 0 for c in CLASSES}

    with torch.no_grad():
        loop = _progress_bar(
            enumerate(dataloader, start=1),
            st_container=st_container,
            total=len(dataloader),
            desc="Évaluation",
        )

        for batch_i, (_, images, labels, encodings) in loop:
            images    = images.to(device)
            encodings = encodings.to(device)
            labels    = labels.to(device)
            preds     = torch.argmax(model(images, encodings), dim=1)

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
    """Prépare les datasets, dataloaders, modèle et device pour l'entraînement."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Calculs effectués sur: {device}")

    train_ds = HDF5Dataset(str(train_h5), "data_train", str(train_csv), max_samples=max_samples)
    val_ds   = HDF5Dataset(str(val_h5),   "data_val",   str(val_csv))
    test_ds  = HDF5Dataset(str(test_h5),  "data_test",  str(test_csv), max_samples=max_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

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
    """Lance un entraînement complet et retourne (history, per_class).

    C'est la fonction à appeler depuis la page Streamlit.
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
    """Retourne une image aléatoire du dataloader avec ses questions et encodings uniques.

    Utilisé pour afficher une image et des questions dans Streamlit.
    """
    model.eval()
    with torch.no_grad():
        questions, images, labels, encs = next(iter(test_loader))
        images = images.to(device)
        encs   = encs.to(device)
        labels = labels.to(device)

        id_img = torch.randint(0, images.size(0), (1,)).item()
        image  = images[id_img]

        questions_unique, qst_ind = np.unique(questions, return_index=True)
        encs_unique = encs[qst_ind]

        return image, questions_unique, encs_unique
