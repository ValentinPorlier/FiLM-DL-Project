"""Fonctions d'entraînement et d'évaluation pour Sort of CLEVR."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import CLASSES


def train_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epochs: int = 10,
) -> dict:
    """Entraîne le modèle et retourne l'historique des métriques.

    Returns
    -------
    history : dict avec clés 'train_loss' et 'train_acc' (listes par époque)
    """
    model.train()
    history = {"train_loss": [], "train_acc": []}

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total   = 0

        loop = tqdm(dataloader, desc=f"Époque [{epoch + 1}/{epochs}]", unit="batch")

        for _, images, labels, encodings in loop:
            images    = images.to(device)
            encodings = encodings.to(device)
            labels    = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, encodings)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted  = torch.max(outputs, 1)
            total        += labels.size(0)
            correct      += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=f"{100 * correct / total:.1f}%")

        epoch_loss = running_loss / len(dataloader)
        epoch_acc  = correct / total
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        print(f"Époque {epoch + 1} | Loss : {epoch_loss:.4f} | Acc : {epoch_acc:.2%}")

    return history


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Évalue le modèle sur un dataloader.

    Returns
    -------
    (loss_moyenne, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for _, images, labels, encodings in dataloader:
            images    = images.to(device)
            encodings = encodings.to(device)
            labels    = labels.to(device)

            outputs     = model(images, encodings)
            loss        = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total       += labels.size(0)
            correct     += (predicted == labels).sum().item()

    return total_loss / len(dataloader), correct / total


def evaluate_per_class(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Calcule l'accuracy par classe.

    Returns
    -------
    dict {classe: accuracy_float}
    """
    model.eval()
    correct_pred = {c: 0 for c in CLASSES}
    total_pred   = {c: 0 for c in CLASSES}

    with torch.no_grad():
        for _, images, labels, encodings in dataloader:
            images    = images.to(device)
            encodings = encodings.to(device)
            labels    = labels.to(device)

            outputs     = model(images, encodings)
            predictions = torch.argmax(outputs, 1)

            for target, pred in zip(labels, predictions):
                cls = CLASSES[target.item()]
                total_pred[cls]   += 1
                if target == pred:
                    correct_pred[cls] += 1

    return {
        cls: correct_pred[cls] / total_pred[cls]
        for cls in CLASSES
        if total_pred[cls] > 0
    }


def visualize_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_images: int = 6,
) -> plt.Figure:
    """Génère une figure matplotlib avec des prédictions d'exemple.

    Returns
    -------
    fig : Figure matplotlib
    """
    model.eval()
    shown = 0
    cols  = min(num_images, 3)
    rows  = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    with torch.no_grad():
        for question_strs, images, labels, encodings in dataloader:
            imgs_gpu = images.to(device)
            encs_gpu = encodings.to(device)
            outputs  = model(imgs_gpu, encs_gpu)
            preds    = torch.argmax(outputs, 1)

            for i in range(images.size(0)):
                if shown >= num_images:
                    break
                ax = list(axes_flat)[shown]
                img = images[i].permute(1, 2, 0).cpu().numpy()
                ax.imshow(img)
                ax.axis("off")

                true_lbl = CLASSES[labels[i].item()]
                pred_lbl = CLASSES[preds[i].item()]
                color    = "#2ecc71" if true_lbl == pred_lbl else "#e74c3c"

                ax.set_title(
                    f"Q : {question_strs[i]}\n"
                    f"Préd : {pred_lbl}  (Vrai : {true_lbl})",
                    color=color, fontsize=9,
                )
                shown += 1

            if shown >= num_images:
                break

    # Masquer les axes vides
    for ax in list(axes_flat)[shown:]:
        ax.set_visible(False)

    fig.patch.set_facecolor("#0d0d1a")
    plt.tight_layout()
    return fig
