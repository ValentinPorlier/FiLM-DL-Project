"""Architecture FiLM pour Sort of CLEVR.

Pipeline :
    Image RGB  →  CNN_feature_map  →  4 × FiLMResBlock  →  FiLMClassifier  →  logits

La question est encodée en un vecteur numérique de taille 10.
Chaque FiLMResBlock prédit localement ses propres γ/β via une couche Linear.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def add_coordinate_maps(x: torch.Tensor) -> torch.Tensor:
    """Ajoute deux canaux de coordonnées spatiales (x, y ∈ [−1, 1]) au tenseur.

    Parameters
    ----------
    x : (B, C, H, W)

    Returns
    -------
    (B, C+2, H, W)
    """
    B, _, H, W = x.shape
    y_coords = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
    x_coords = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
    return torch.cat([x, y_coords, x_coords], dim=1)


class CNN_feature_map(nn.Module):
    """Extracteur de features CNN — 4 convolutions stride-2 avec BatchNorm.

    Réduit la résolution spatiale par un facteur 16 (ex. 128→8).

    Parameters
    ----------
    in_channels  : canaux en entrée (3 pour RGB)
    out_channels : largeur des feature maps (128 par défaut)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 128) -> None:
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(4):
            layers += [
                nn.Conv2d(ch, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            ch = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLMResBlock(nn.Module):
    """Bloc résiduel FiLM pour Sort of CLEVR.

    Architecture :
        [x ++ coord]  →  Conv1×1  →  ReLU  (= résidu)
                      →  Conv3×3  →  BN(affine=False)  →  FiLM(γ,β)  →  ReLU
                      +  résidu

    Le générateur FiLM est ici une simple couche Linear(qst_dim → 2C)
    car la question est déjà sous forme d'un vecteur numérique.

    Parameters
    ----------
    in_channels : nombre de canaux en entrée (= out_channels du CNN)
    out_channels: nombre de canaux en sortie
    qst_dim     : dimension de l'encodage de question (10 par défaut)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        qst_dim: int = 10,
    ) -> None:
        super().__init__()
        self.conv1         = nn.Conv2d(in_channels + 2, out_channels, kernel_size=1)
        self.conv2         = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn            = nn.BatchNorm2d(out_channels, affine=False)
        self.film_generator = nn.Linear(qst_dim, out_channels * 2)

    def forward(self, x: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : (B, C, H, W)
        encoding : (B, qst_dim)
        """
        x   = add_coordinate_maps(x)        # (B, C+2, H, W)
        x   = torch.relu(self.conv1(x))     # (B, C, H, W)  ← résidu
        res = x

        x = self.conv2(x)
        x = self.bn(x)

        # Paramètres FiLM prédits par le générateur linéaire
        params       = self.film_generator(encoding).unsqueeze(2).unsqueeze(3)  # (B, 2C, 1, 1)
        gamma, beta  = params.chunk(2, dim=1)
        x = (1 + gamma) * x + beta  # formulation résiduelle : gamma=0 → identité au départ

        x = torch.relu(x) + res
        return x


class FiLMClassifier(nn.Module):
    """Tête de classification après les blocs FiLM.

    Pipeline : Conv1×1(C→512) → BN → ReLU → MaxPool global → MLP(512→1024→1024→K)

    Parameters
    ----------
    in_channels : canaux en entrée (128)
    num_answers : nombre de classes de sortie (11 pour Sort of CLEVR)
    """

    def __init__(self, in_channels: int = 128, num_answers: int = 11) -> None:
        super().__init__()
        self.conv_1x1 = nn.Conv2d(in_channels, 512, kernel_size=1)
        self.bn       = nn.BatchNorm2d(512)
        self.mlp      = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_answers),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn(self.conv_1x1(x)))            # (B, 512, H, W)
        x = x.view(x.size(0), x.size(1), -1).max(dim=2)[0]  # Global Max Pool → (B, 512)
        return self.mlp(x)


class SortOfClevrFiLMModel(nn.Module):
    """Modèle FiLM complet pour Sort of CLEVR.

    Parameters
    ----------
    num_answers : nombre de classes (11 par défaut)
    """

    def __init__(self, num_answers: int = 11) -> None:
        super().__init__()
        self.featuremap = CNN_feature_map(in_channels=3, out_channels=128)
        self.res_blocks = nn.ModuleList([FiLMResBlock(128, 128) for _ in range(4)])
        self.classifier  = FiLMClassifier(in_channels=128, num_answers=num_answers)

    def forward(self, img: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        img      : (B, 3, H, W)
        encoding : (B, 10)

        Returns
        -------
        logits : (B, num_answers)
        """
        x = self.featuremap(img)
        for block in self.res_blocks:
            x = block(x, encoding)
        return self.classifier(x)
