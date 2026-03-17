"""Recherche d'hyperparamètres avec Optuna pour FiLM sur CLEVR.

Chaque essai entraîne le modèle sur un sous-ensemble réduit (max_samples_train)
pendant quelques epochs, et retourne la val accuracy.

Usage :
    python -m src.optuna_search --data-dir data/clevr --n-trials 20 --epochs 5 --train-samples 20000

Résultats sauvegardés dans runs/optuna/
"""

import argparse
import json
from pathlib import Path

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import MiniCLEVR, NUM_ANSWERS
from .model import FiLMModel
from .train import train_one_epoch, evaluate, set_seed


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    set_seed(42 + trial.number)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Hyperparamètres à optimiser ────────────────────────────────────────
    lr           = trial.suggest_float("lr",           1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [32, 64, 128])
    num_channels = trial.suggest_categorical("num_channels", [64, 128, 256])
    hidden_dim   = trial.suggest_categorical("hidden_dim", [256, 1024, 4096])
    dropout      = trial.suggest_float("dropout", 0.0, 0.2)

    # ── Données ────────────────────────────────────────────────────────────
    train_ds = MiniCLEVR(
        args.data_dir, split="train",
        max_samples=args.train_samples,
        img_size=128, max_question_length=45,
    )
    val_ds = MiniCLEVR(
        args.data_dir, split="val",
        max_samples=args.val_samples,
        img_size=128, max_question_length=45,
        vocab=train_ds.vocab,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Modèle ─────────────────────────────────────────────────────────────
    input_channels  = train_ds.feature_shape[0]
    feature_map_size = train_ds.feature_shape[1]

    model = FiLMModel(
        vocab_size=train_ds.vocab_size,
        num_answers=NUM_ANSWERS,
        num_channels=num_channels,
        hidden_dim=hidden_dim,
        input_channels=input_channels,
        feature_map_size=feature_map_size,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # ── Entraînement ───────────────────────────────────────────────────────
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

        print(f"  Trial {trial.number} | Epoch {epoch}/{args.epochs} | "
              f"train_acc={train_acc:.3f} | val_acc={val_acc:.3f}")

        best_val_acc = max(best_val_acc, val_acc)

        # Élagage des essais peu prometteurs
        trial.report(val_acc, epoch)
        if trial.should_prune():
            train_ds.close()
            val_ds.close()
            raise optuna.TrialPruned()

    train_ds.close()
    val_ds.close()
    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",      default="data/clevr")
    parser.add_argument("--n-trials",      type=int, default=20)
    parser.add_argument("--epochs",        type=int, default=5,     help="epochs par essai")
    parser.add_argument("--train-samples", type=int, default=20000, help="samples train par essai")
    parser.add_argument("--val-samples",   type=int, default=3000,  help="samples val par essai")
    parser.add_argument("--out",           default="runs/optuna")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name="film_clevr",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        storage=f"sqlite:///{out_dir}/optuna.db",
        load_if_exists=True,
    )

    print(f"Démarrage de la recherche : {args.n_trials} essais, {args.epochs} epochs chacun")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    print("\n── Meilleurs hyperparamètres ──────────────────────────────────────")
    print(f"  val_acc : {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k:20s}: {v}")

    # Sauvegarde JSON
    result = {"best_val_acc": study.best_value, "best_params": study.best_params}
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nRésultats sauvegardés dans {out_dir}/best_params.json")


if __name__ == "__main__":
    main()
