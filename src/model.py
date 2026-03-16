"""FiLM model for CLEVR VQA — adapted from github.com/ethanjperez/film.

This is a faithful port of the original FiLMedNet to modern PyTorch,
preserving the exact architecture that achieves 97%+ on CLEVR.

Architecture (paper-faithful)
------------------------------
  Stem: 2 × Conv3×3(feat_dim→C), no BN, stride 1
  → N × FiLMedResBlock:
      [x ++ coord] → input_proj Conv1×1(C+2→C) → ReLU
      → Conv3×3(C→C) → BN(affine=False) → FiLM(γ,β) → Dropout → ReLU
      + residual (from input_proj output)
  → Append coord → MaxPool2(k=2,s=2) → Conv1×1(C+2→proj_dim) → ReLU
  → GlobalAvgPool → FC(fc_dim) → ReLU → FC(num_answers)

Key details from the original:
  - stem has NO batchnorm (stem_batchnorm=False in defaults)
  - ResBlock BN has affine=False (FiLM provides the affine transform)
  - condition_method='bn-film': FiLM is applied AFTER BN
  - input_proj (1×1 conv) before the main 3×3 conv in each block
  - coord channels appended at input_proj AND before classifier
  - classifier uses maxpool2 + 1×1 proj + global avg pool + FC layers
  - module_batchnorm=False in original defaults → but bn-film still uses
    a BN layer with affine=False when condition_method='bn-film'

Modernisation from original:
  - Removed Variable() wrappers
  - Device-agnostic (no hardcoded .cuda())
  - Clean type annotations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .film_layer import FiLMLayer
from .film_generator import FiLMGenerator


# ── Coordinate maps ──────────────────────────────────────────────────────────

def coord_map(shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    """Return (2, H, W) coordinate feature maps in [−1, +1]."""
    H, W = shape
    x = torch.linspace(-1, 1, W, device=device).view(1, 1, W).expand(1, H, W)
    y = torch.linspace(-1, 1, H, device=device).view(1, H, 1).expand(1, H, W)
    return torch.cat([x, y], dim=0)


# ── FiLMed Residual Block ────────────────────────────────────────────────────

class FiLMedResBlock(nn.Module):
    """Single FiLM-conditioned residual block (paper-faithful).

    Architecture:
        [x ++ coord(2)] → Conv1×1(C+2, C) → ReLU     (input_proj)
                        → Conv3×3(C, C)               (main conv)
                        → BN(affine=False)             (normalise)
                        → FiLM(γ, β)                   (condition)
                        → Dropout2d                    (regularise)
                        → ReLU                         (activate)
                        + residual (input_proj output) (skip)
    """

    def __init__(
        self,
        num_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Input projection: reduces coord-augmented input back to C channels
        self.input_proj = nn.Conv2d(
            num_channels + 2, num_channels, kernel_size=1, bias=True
        )
        # Main convolution
        self.conv = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, padding=1, bias=False
        )
        # BN with affine=False: FiLM provides the affine parameters
        self.bn = nn.BatchNorm2d(num_channels, affine=False)
        self.film = FiLMLayer()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        batch_coords: torch.Tensor,
    ) -> torch.Tensor:
        # Input projection with coord channels
        x_aug = torch.cat([x, batch_coords], dim=1)   # (B, C+2, H, W)
        x_proj = F.relu(self.input_proj(x_aug))        # (B, C, H, W)

        # Main path
        out = self.conv(x_proj)                         # (B, C, H, W)
        out = self.bn(out)
        out = self.film(out, gamma, beta)
        out = self.drop(out)
        out = F.relu(out)

        # Residual from input_proj output
        out = x_proj + out
        return out


# ── Full FiLM model ──────────────────────────────────────────────────────────

class FiLMModel(nn.Module):
    """FiLM model for CLEVR VQA.

    Parameters
    ----------
    vocab_size       : question vocabulary size
    num_answers      : output classes (28 for CLEVR)
    num_blocks       : number of FiLM-ed residual blocks (default 4)
    num_channels     : feature-map channels (default 128)
    embedding_dim    : word embedding dim (default 200)
    hidden_dim       : GRU hidden dim (default 4096)
    rnn_num_layers   : GRU layers (default 1)
    rnn_dropout      : dropout between GRU layers
    input_channels   : 1024 for ResNet features, 3 for raw RGB
    classifier_proj_dim : 1×1 conv projection before global pool (default 512)
    classifier_fc_dim   : FC hidden dim in classifier (default 1024)
    dropout          : spatial dropout in ResBlocks (default 0.05)
    fix_gamma        : ablation — force γ = 1
    fix_beta         : ablation — force β = 0
    """

    def __init__(
        self,
        vocab_size: int,
        num_answers: int = 28,
        num_blocks: int = 4,
        num_channels: int = 128,
        embedding_dim: int = 200,
        hidden_dim: int = 4096,
        rnn_num_layers: int = 1,
        rnn_dropout: float = 0.0,
        input_channels: int = 1024,
        feature_map_size: int = 14,
        classifier_proj_dim: int = 512,
        classifier_fc_dim: int = 1024,
        dropout: float = 0.05,
        classifier_dropout: float = 0.5,
        fix_gamma: bool = False,
        fix_beta: bool = False,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.num_channels = num_channels
        self.fix_gamma = fix_gamma
        self.fix_beta = fix_beta

        # ── Stem: 2 × Conv3×3, no BN (matches original defaults) ────────
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # ── FiLM-ed residual blocks ─────────────────────────────────────
        self.blocks = nn.ModuleList([
            FiLMedResBlock(num_channels, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # ── FiLM generator (question encoder) ───────────────────────────
        self.film_generator = FiLMGenerator(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            rnn_num_layers=rnn_num_layers,
            rnn_dropout=rnn_dropout,
            num_blocks=num_blocks,
            num_channels=num_channels,
        )

        # ── Classifier (faithful to original build_classifier) ───────────
        # Original order: Conv1×1(C+2→proj) → ReLU → MaxPool2×2 → Flatten
        #                 → FC(proj*H/2*W/2 → fc_dim) → ReLU → FC(fc_dim→answers)
        # Stem has no stride → feature map stays feature_map_size × feature_map_size
        pooled_size = feature_map_size // 2  # e.g. 14 → 7
        fc_input_dim = classifier_proj_dim * pooled_size * pooled_size
        self.classifier_proj = nn.Conv2d(
            num_channels + 2, classifier_proj_dim, kernel_size=1
        )
        self.classifier_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_dim, classifier_fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_fc_dim, num_answers),
        )

    def forward(
        self,
        image: torch.Tensor,
        question: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        image    : (B, C_in, H, W)
        question : (B, T)
        lengths  : (B,)

        Returns
        -------
        logits : (B, num_answers)
        """
        B = image.size(0)

        # FiLM parameters from question
        film_params = self.film_generator(question, lengths)

        # Visual stem
        features = self.stem(image)  # (B, C, H, W)
        _, _, H, W = features.shape

        # Precompute coord maps (shared across all blocks)
        batch_coords = coord_map((H, W), features.device)
        batch_coords = batch_coords.unsqueeze(0).expand(B, -1, -1, -1)

        # FiLM-ed residual blocks
        for k, block in enumerate(self.blocks):
            gamma, beta = film_params[k]
            if self.fix_gamma:
                gamma = torch.ones_like(gamma)
            if self.fix_beta:
                beta = torch.zeros_like(beta)
            features = block(features, gamma, beta, batch_coords)

        # Classifier (original order: proj → pool → flatten → FC)
        features = torch.cat([features, batch_coords], dim=1)  # (B, C+2, H, W)
        features = F.relu(self.classifier_proj(features))       # (B, proj_dim, H, W)
        features = self.classifier_pool(features)               # (B, proj_dim, H/2, W/2)
        logits = self.classifier(features)                      # flatten → FC → num_answers

        return logits

    def get_film_params(
        self,
        question: torch.Tensor,
        lengths: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Extract FiLM parameters without running the visual pipeline."""
        with torch.no_grad():
            return self.film_generator(question, lengths)