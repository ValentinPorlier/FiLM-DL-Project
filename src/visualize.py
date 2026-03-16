"""Visualisation helpers for the FiLM Streamlit app.

Functions
---------
plot_training_curves      — Loss & accuracy curves (Plotly)
plot_gamma_beta_hist      — Distribution of γ/β per block (Plotly)
plot_feature_maps         — Feature maps before/after FiLM (matplotlib figure)
plot_predictions_grid     — Grid of sample predictions (matplotlib figure)
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict) -> go.Figure:
    """Plotly figure with loss and accuracy curves (train + val).

    Parameters
    ----------
    history : dict
        Keys: "train_loss", "val_loss", "train_acc", "val_acc" (lists).
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Loss", "Accuracy"),
    )

    # Loss
    fig.add_trace(
        go.Scatter(x=epochs, y=history["train_loss"], name="Train loss",
                   line=dict(color="#636EFA", width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history["val_loss"], name="Val loss",
                   line=dict(color="#EF553B", width=2)),
        row=1, col=1,
    )

    # Accuracy
    fig.add_trace(
        go.Scatter(x=epochs, y=history["train_acc"], name="Train acc",
                   line=dict(color="#636EFA", width=2, dash="dot")),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(x=epochs, y=history["val_acc"], name="Val acc",
                   line=dict(color="#EF553B", width=2, dash="dot")),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2, range=[0, 1])
    fig.update_layout(height=400, template="plotly_dark", legend=dict(x=0.5, y=-0.15, orientation="h"))
    return fig


# ---------------------------------------------------------------------------
# Gamma / Beta histograms
# ---------------------------------------------------------------------------

def plot_gamma_beta_hist(film_params: dict, num_blocks: int) -> go.Figure:
    """Plotly figure: one column per block, rows for gamma and beta.

    Parameters
    ----------
    film_params : dict
        Keys "gamma_<k>" and "beta_<k>" with flat numpy arrays.
    num_blocks : int
    """
    fig = make_subplots(
        rows=2, cols=num_blocks,
        row_titles=["γ (gamma)", "β (beta)"],
        column_titles=[f"Block {k}" for k in range(num_blocks)],
        horizontal_spacing=0.04,
        vertical_spacing=0.12,
    )

    colors_g = "#636EFA"
    colors_b = "#FFA15A"

    for k in range(num_blocks):
        gamma_vals = film_params.get(f"gamma_{k}", np.array([]))
        beta_vals = film_params.get(f"beta_{k}", np.array([]))

        if len(gamma_vals):
            fig.add_trace(
                go.Histogram(x=gamma_vals, nbinsx=40, name=f"γ_{k}",
                             marker_color=colors_g, showlegend=(k == 0)),
                row=1, col=k + 1,
            )
        if len(beta_vals):
            fig.add_trace(
                go.Histogram(x=beta_vals, nbinsx=40, name=f"β_{k}",
                             marker_color=colors_b, showlegend=(k == 0)),
                row=2, col=k + 1,
            )

    fig.update_layout(
        height=420,
        template="plotly_dark",
        bargap=0.05,
        title_text="Distribution of learned γ and β per block (val set)",
    )
    return fig


# ---------------------------------------------------------------------------
# Feature maps
# ---------------------------------------------------------------------------

def _denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalised CHW tensor to HWC uint8 numpy array."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


def plot_feature_maps(
    model,
    image: torch.Tensor,
    question: torch.Tensor,
    device: torch.device,
    block_idx: int = 0,
    n_maps: int = 6,
) -> plt.Figure:
    """Show feature maps before and after FiLM for a given block.

    Parameters
    ----------
    model : FiLMModel
    image : (3, H, W) tensor (normalised)
    question : (T,) tensor of word indices
    device
    block_idx : which residual block to probe
    n_maps : how many channels to display
    """
    model.eval()
    hooks_pre: list[torch.Tensor] = []
    hooks_post: list[torch.Tensor] = []

    def hook_pre(module, inp, out):
        hooks_pre.append(out.detach().cpu())

    def hook_post(module, inp, out):
        hooks_post.append(out.detach().cpu())

    block = model.blocks[block_idx]
    h1 = block.film.register_forward_pre_hook(lambda m, i: hooks_pre.append(i[0].detach().cpu()))
    h2 = block.film.register_forward_hook(lambda m, i, o: hooks_post.append(o.detach().cpu()))

    with torch.no_grad():
        q = question.unsqueeze(0).to(device)
        lengths = (q != 0).sum(dim=1)
        model(image.unsqueeze(0).to(device), q, lengths)

    h1.remove()
    h2.remove()

    pre = hooks_pre[0][0]   # (C, H, W)
    post = hooks_post[0][0] # (C, H, W)
    n_maps = min(n_maps, pre.shape[0])

    fig, axes = plt.subplots(3, n_maps, figsize=(n_maps * 2, 6))
    fig.patch.set_facecolor("#1e1e1e")

    # Row 0: input image repeated
    img_np = _denormalise(image)
    for j in range(n_maps):
        ax = axes[0, j]
        ax.imshow(img_np)
        ax.axis("off")
        if j == 0:
            ax.set_title("Input image", color="white", fontsize=9)

    # Row 1: before FiLM
    for j in range(n_maps):
        ax = axes[1, j]
        fm = pre[j].numpy()
        ax.imshow(fm, cmap="viridis")
        ax.axis("off")
        if j == 0:
            ax.set_title("Before FiLM", color="white", fontsize=9)

    # Row 2: after FiLM
    for j in range(n_maps):
        ax = axes[2, j]
        fm = post[j].numpy()
        ax.imshow(fm, cmap="viridis")
        ax.axis("off")
        if j == 0:
            ax.set_title("After FiLM", color="white", fontsize=9)

    for ax_row in axes:
        for ax in ax_row:
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.tight_layout(pad=0.5)
    return fig


# ---------------------------------------------------------------------------
# Prediction grid
# ---------------------------------------------------------------------------

def plot_predictions_grid(samples: list[dict], cols: int = 5) -> plt.Figure:
    """Display a grid of sample predictions with colour-coded correctness.

    Parameters
    ----------
    samples : list of dicts with keys image, question, answer, prediction, correct
    cols : grid columns
    """
    n = len(samples)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    fig.patch.set_facecolor("#1e1e1e")
    if rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, "__iter__") else [row])]

    for i, ax in enumerate(axes_flat):
        ax.set_facecolor("#1e1e1e")
        if i >= n:
            ax.axis("off")
            continue

        sample = samples[i]
        img_np = _denormalise(sample["image"])
        ax.imshow(img_np)
        ax.axis("off")

        color = "#00cc66" if sample["correct"] else "#ff4444"
        label = (
            f"Q: {sample['question'][:40]}\n"
            f"Pred: {sample['prediction']}  GT: {sample['answer']}"
        )
        ax.set_title(label, color=color, fontsize=7, wrap=True)

    plt.tight_layout(pad=0.3)
    return fig
