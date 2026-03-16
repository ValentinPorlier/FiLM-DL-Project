"""Training and evaluation loops for the FiLM model on Mini-CLEVR.
 
Usage (CLI smoke-test):
    python -m src.train --config configs/default.yaml
 
The `train()` function is also called from the Streamlit UI via a
background thread.  It communicates progress through a shared
`progress_queue` (queue.Queue) whose items are dicts with keys:
  - "epoch"       : current epoch (1-based)
  - "train_loss"  : float
  - "train_acc"   : float
  - "val_loss"    : float
  - "val_acc"     : float
  - "done"        : bool (True on the last message)
  - "error"       : str  (only present if an exception occurred)
"""
 
from __future__ import annotations
 
import queue
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
 
from .dataset import MiniCLEVR, NUM_ANSWERS
from .model import FiLMModel
 
 
# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
 
@dataclass
class TrainConfig:
    # Dataset
    data_dir: str = "data/clevr"
    max_samples_train: int = 70000
    max_samples_val: int = 5000
    img_size: int = 128
    max_question_length: int = 45
 
    # Model (paper-faithful defaults)
    num_blocks: int = 4
    num_channels: int = 128
    embedding_dim: int = 300
    hidden_dim: int = 256
    rnn_num_layers: int = 2
    classifier_proj_dim: int = 512
    dropout: float = 0.1
    classifier_dropout: float = 0.5
    use_batchnorm: bool = True
    fix_gamma: bool = False
    fix_beta: bool = False
 
    # Training
    num_epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    grad_clip: float = 0.0  # <= 0 means no clipping (original default)
    seed: int = 42
 
    # I/O
    run_dir: str = "runs/default"
    save_gamma_beta: bool = True
 
    # Extra fields used at runtime (not part of YAML)
    extra: dict = field(default_factory=dict)
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
 
 
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()
 
 
# ---------------------------------------------------------------------------
# One-epoch loops
# ---------------------------------------------------------------------------
 
def train_one_epoch(
    model: FiLMModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 0.0,
    progress_callback=None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = len(dataloader)
 
    for batch_idx, (images, questions, answers, _, _) in enumerate(dataloader):
        images = images.to(device)
        questions = questions.to(device)
        answers = answers.to(device)
        lengths = (questions != 0).sum(dim=1)
 
        optimizer.zero_grad()
        logits = model(images, questions, lengths)
        loss = criterion(logits, answers)
        loss.backward()
 
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
 
        optimizer.step()
 
        total_loss += loss.item()
        total_acc += accuracy(logits, answers)
 
        if progress_callback is not None:
            progress_callback(batch_idx + 1, n_batches)
 
    return total_loss / n_batches, total_acc / n_batches
 
 
@torch.no_grad()
def evaluate(
    model: FiLMModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = len(dataloader)
 
    for images, questions, answers, _, _ in dataloader:
        images = images.to(device)
        questions = questions.to(device)
        answers = answers.to(device)
        lengths = (questions != 0).sum(dim=1)
 
        logits = model(images, questions, lengths)
        loss = criterion(logits, answers)
 
        total_loss += loss.item()
        total_acc += accuracy(logits, answers)
 
    return total_loss / n_batches, total_acc / n_batches
 
 
# ---------------------------------------------------------------------------
# Gamma/Beta collection
# ---------------------------------------------------------------------------
 
@torch.no_grad()
def collect_film_params(
    model: FiLMModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> dict[str, list]:
    model.eval()
    collected: dict[str, list] = {}
 
    for batch_idx, (images, questions, answers, _, _) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
 
        questions = questions.to(device)
        lengths = (questions != 0).sum(dim=1)
        film_params = model.film_generator(questions, lengths)
 
        for k, (gamma, beta) in enumerate(film_params):
            g_key = f"gamma_{k}"
            b_key = f"beta_{k}"
            if g_key not in collected:
                collected[g_key] = []
                collected[b_key] = []
            collected[g_key].append(gamma.cpu().numpy().ravel())
            collected[b_key].append(beta.cpu().numpy().ravel())
 
    return {k: np.concatenate(v) for k, v in collected.items()}
 
 
# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
 
def train(
    config: TrainConfig,
    progress_queue: Optional[queue.Queue] = None,
) -> dict:
    set_seed(config.seed)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
 
    # ---- Dataset --------------------------------------------------------
    train_dataset = MiniCLEVR(
        config.data_dir,
        split="train",
        max_samples=config.max_samples_train,
        img_size=config.img_size,
        max_question_length=config.max_question_length,
    )
    val_dataset = MiniCLEVR(
        config.data_dir,
        split="val",
        max_samples=config.max_samples_val,
        img_size=config.img_size,
        max_question_length=config.max_question_length,
        vocab=train_dataset.vocab,
    )
 
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
 
    # ---- Model ----------------------------------------------------------
    input_channels = train_dataset.feature_shape[0]
    feature_map_size = train_dataset.feature_shape[1]  # e.g. 14 for ResNet101

    model = FiLMModel(
        vocab_size=train_dataset.vocab_size,
        num_answers=NUM_ANSWERS,
        num_blocks=config.num_blocks,
        num_channels=config.num_channels,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        rnn_num_layers=config.rnn_num_layers,
        input_channels=input_channels,
        feature_map_size=feature_map_size,
        classifier_proj_dim=config.classifier_proj_dim,
        dropout=config.dropout,
        classifier_dropout=config.classifier_dropout,
        fix_gamma=config.fix_gamma,
        fix_beta=config.fix_beta,
    ).to(device)
 
    print(
        f"Input channels: {input_channels} "
        f"({'feature mode' if input_channels != 3 else 'raw image mode'})"
    )
 
    # Original uses Adam with no weight decay, no LR schedule
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss()
 
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model params: {n_params:,} | Input: {input_channels}ch | "
        f"Train samples: {len(train_dataset)}"
    )
 
    history: dict = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    epochs_no_improve = 0
    early_stop_patience = 10
    cfg_save = dict(config.__dict__)
    cfg_save["input_channels"] = input_channels
    cfg_save["feature_map_size"] = feature_map_size
    cfg_save["vocab_size"] = train_dataset.vocab_size

    # ---- Training loop --------------------------------------------------
    for epoch in range(1, config.num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip=config.grad_clip,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Save best checkpoint
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "vocab": train_dataset.vocab,
                "config": cfg_save,
                "history": history,
            }, run_dir / "checkpoint_best.pt")
        else:
            epochs_no_improve += 1

        msg = (
            f"Epoch {epoch}/{config.num_epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
            f"lr {current_lr:.2e} | {elapsed:.1f}s"
            + (" best" if improved else f" (no improve {epochs_no_improve}/{early_stop_patience})")
        )
        print(msg)

        if progress_queue is not None:
            progress_queue.put({
                "epoch": epoch,
                "num_epochs": config.num_epochs,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
                "done": epoch == config.num_epochs,
            })

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    # ---- Load best model for final eval/save ----------------------------
    best_ckpt = run_dir / "checkpoint_best.pt"
    if best_ckpt.exists():
        best_state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(best_state["model_state"])
        print(f"Restored best model (val_loss={best_val_loss:.4f})")

    # ---- Save final checkpoint ------------------------------------------
    checkpoint = {
        "model_state": model.state_dict(),
        "vocab": train_dataset.vocab,
        "config": cfg_save,
        "history": history,
    }
    torch.save(checkpoint, run_dir / "checkpoint.pt")
 
    # ---- Collect gamma / beta for visualisation -------------------------
    film_params = {}
    if config.save_gamma_beta:
        film_params = collect_film_params(model, val_loader, device)
 
    history["film_params"] = film_params
    history["run_dir"] = str(run_dir)
    history["vocab_size"] = train_dataset.vocab_size
 
    # ---- Collect sample predictions for display -------------------------
    history["sample_predictions"] = _collect_samples(
        model, val_dataset, device, n=10
    )
 
    # ---- Release H5 file handles ----------------------------------------
    train_dataset.close()
    val_dataset.close()
 
    return history
 
 
# ---------------------------------------------------------------------------
# Sample predictions helper
# ---------------------------------------------------------------------------
 
@torch.no_grad()
def _collect_samples(
    model: FiLMModel,
    dataset: MiniCLEVR,
    device: torch.device,
    n: int = 10,
) -> list[dict]:
    from .dataset import CLEVR_ANSWERS
 
    model.eval()
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    samples = []
 
    for idx in indices:
        image, question, answer_idx, question_str, answer_str = dataset[idx]
 
        q = question.unsqueeze(0).to(device)
        lengths = (q != 0).sum(dim=1)
 
        logit = model(image.unsqueeze(0).to(device), q, lengths)
        pred_idx = logit.argmax(dim=-1).item()
        pred_str = CLEVR_ANSWERS[pred_idx] if pred_idx < len(CLEVR_ANSWERS) else "?"
 
        samples.append({
            "image": image,
            "question": question_str,
            "answer": answer_str,
            "prediction": pred_str,
            "correct": pred_str == answer_str,
        })
 
    return samples
 
 
# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    import argparse
    import yaml
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
 
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
 
    config = TrainConfig(**{
        k: v
        for k, v in cfg_dict.items()
        if k in TrainConfig.__dataclass_fields__
    })
 
    train(config)