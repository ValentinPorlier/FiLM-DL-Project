"""Fonctions d'entraînement et d'évaluation pour Sort of CLEVR."""

from __future__ import annotations

import queue as _queue

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import CLASSES, HDF5Dataset, NUM_CLASSES
from .model import SortOfClevrFiLMModel


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epochs: int = 10,
    progress_queue: _queue.Queue | None = None,
) -> dict:
    """Entraîne le modèle, évalue sur val à chaque epoch, retourne l'historique.

    Si progress_queue est fourni, envoie un dict par epoch :
        {"epoch": int, "num_epochs": int, "train_loss", "train_acc", "val_loss", "val_acc"}
    Et à la fin :
        {"done": True, "history": dict}
    """
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for _, images, labels, encodings in loop:
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
            correct  += (torch.argmax(outputs, 1) == labels).sum().item()

            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.1%}")

        t_loss = run_loss / len(train_loader)
        t_acc  = correct / total
        print(f"  Evaluation...", flush=True)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        print(f"  => Train {t_acc:.2%} | Val {v_acc:.2%}", flush=True)


        if progress_queue is not None:
            progress_queue.put({
                "epoch": epoch + 1, "num_epochs": epochs,
                "train_loss": t_loss, "train_acc": t_acc,
                "val_loss": v_loss, "val_acc": v_acc,
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
            correct    += (torch.argmax(outputs, 1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(dataloader), correct / total


def evaluate_per_class(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Retourne {classe: accuracy} pour chaque classe."""
    model.eval()
    correct_pred = {c: 0 for c in CLASSES}
    total_pred   = {c: 0 for c in CLASSES}

    with torch.no_grad():
        for _, images, labels, encodings in dataloader:
            images    = images.to(device)
            encodings = encodings.to(device)
            labels    = labels.to(device)
            preds     = torch.argmax(model(images, encodings), 1)

            for target, pred in zip(labels, preds):
                cls = CLASSES[target.item()]
                total_pred[cls]   += 1
                if target == pred:
                    correct_pred[cls] += 1

    return {cls: correct_pred[cls] / total_pred[cls] for cls in CLASSES if total_pred[cls] > 0}


def run(train_h5, train_csv, test_h5, test_csv, epochs=10, batch_size=128, lr=0.001, max_samples=None, num_workers=0):
    """Lance un entraînement complet et retourne (history, per_class).

    C'est la fonction à appeler depuis la page Streamlit.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = HDF5Dataset(str(train_h5), "data_train", str(train_csv), max_samples=max_samples)
    test_ds  = HDF5Dataset(str(test_h5),  "data_test",  str(test_csv),  max_samples=max_samples // 5 if max_samples else None)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model     = SortOfClevrFiLMModel(num_answers=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history   = train_model(model, train_loader, test_loader, optimizer, criterion, device, epochs=epochs)
    per_class = evaluate_per_class(model, test_loader, device)

    return history, per_class
