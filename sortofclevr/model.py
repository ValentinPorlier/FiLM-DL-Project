"""Architecture FiLM pour Sort of CLEVR.

Pipeline :
    Image RGB  →  CNN_feature_map  →  4 × FiLMResBlock  →  FiLMClassifier  →  logits

La question est encodée en un vecteur numérique de taille 10.
Chaque FiLMResBlock prédit localement ses propres γ/β via une couche Linear.
"""


import torch
import torch.nn as nn


def add_coordinate_maps(x: torch.Tensor) -> torch.Tensor:
    """Ajoute deux canaux de coordonnées spatiales au tenseur.

    Les coordonnées vont de -1 à 1, une pour la hauteur (y) et une pour la
    largeur (x). Ça permet au modèle de savoir où se trouve chaque pixel dans
    l'image.

    Parameters
    ----------
    x : torch.Tensor
        Tenseur de features de forme (B, C, H, W).

    Returns
    -------
    torch.Tensor
        Même tenseur avec 2 canaux en plus, de forme (B, C+2, H, W).
    """
    B, _, H, W = x.shape
    y_coords = torch.linspace(-1, 1, H, device=x.device).view(1,
                                                              1, H, 1).expand(B, 1, H, W)
    x_coords = torch.linspace(-1, 1, W, device=x.device).view(1,
                                                              1, 1, W).expand(B, 1, H, W)
    return torch.cat([x, y_coords, x_coords], dim=1)


class CNN_feature_map(nn.Module):
    """Extracteur de features CNN — 4 convolutions stride-2 avec BatchNorm.

    Réduit la résolution spatiale par un facteur 16 (ex. 128→8).

    Parameters
    ----------
    in_channels : int
        Nombre de canaux en entrée (3 pour une image RGB).
    out_channels : int
        Nombre de canaux en sortie de chaque couche (128 par défaut).
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
        """Passe l'image dans les 4 couches de convolution.

        Parameters
        ----------
        x : torch.Tensor
            Image en entrée de forme (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Carte de features de forme (B, out_channels, H/16, W/16).
        """
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
    in_channels : int
        Nombre de canaux en entrée (= out_channels du CNN).
    out_channels : int
        Nombre de canaux en sortie.
    qst_dim : int
        Dimension de l'encodage de question (10 par défaut).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        qst_dim: int = 10,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + 2, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.film_generator = nn.Linear(qst_dim, out_channels * 2)

    def forward(self, x: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        """Applique la modulation FiLM sur les features avec la question en entrée.

        Ajoute d'abord les coordonnées spatiales, puis applique les deux
        convolutions. Les paramètres γ et β sont prédits à partir de la
        question pour moduler les features avant l'activation finale.

        Parameters
        ----------
        x : torch.Tensor
            Carte de features de forme (B, C, H, W).
        encoding : torch.Tensor
            Encodage de la question de forme (B, qst_dim).

        Returns
        -------
        torch.Tensor
            Carte de features modulée, de même forme que x : (B, C, H, W).
        """
        x = add_coordinate_maps(x)        # (B, C+2, H, W)
        x = torch.relu(self.conv1(x))     # (B, C, H, W)  ← résidu
        res = x

        x = self.conv2(x)
        x = self.bn(x)

        # Paramètres FiLM prédits par le générateur linéaire
        params = self.film_generator(encoding).unsqueeze(
            2).unsqueeze(3)  # (B, 2C, 1, 1)
        gamma, beta = params.chunk(2, dim=1)
        # formulation résiduelle : gamma=0 → identité au départ
        x = (1 + gamma) * x + beta

        x = torch.relu(x) + res
        return x


class FiLMClassifier(nn.Module):
    """Tête de classification après les blocs FiLM.

    Pipeline : Conv1×1(C→512) → BN → ReLU → MaxPool global → MLP(512→1024→1024→K)

    Parameters
    ----------
    in_channels : int
        Nombre de canaux en entrée (128 par défaut).
    num_answers : int
        Nombre de classes de sortie (11 pour Sort of CLEVR).
    """

    def __init__(self, in_channels: int = 128, num_answers: int = 11) -> None:
        super().__init__()
        self.conv_1x1 = nn.Conv2d(in_channels, 512, kernel_size=1)
        self.bn = nn.BatchNorm2d(512)
        self.mlp = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_answers),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calcule les scores pour chaque réponse possible.

        Réduit les features spatiales en un seul vecteur via un max pooling
        global, puis passe ce vecteur dans le MLP pour obtenir les scores
        de chaque classe.

        Parameters
        ----------
        x : torch.Tensor
            Cartes de features en entrée de forme (B, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Scores (logits) de forme (B, num_answers), un par classe.
        """
        x = torch.relu(self.bn(self.conv_1x1(x)))            # (B, 512, H, W)
        x = x.view(x.size(0), x.size(1), -1).max(dim=2)[0]  # Global Max Pool → (B, 512)
        return self.mlp(x)


class SortOfClevrFiLMModel(nn.Module):
    """Modèle FiLM complet pour Sort of CLEVR.

    Parameters
    ----------
    num_answers : int
        Nombre de classes (11 par défaut).
    """

    def __init__(self, num_answers: int = 11) -> None:
        super().__init__()
        self.featuremap = CNN_feature_map(in_channels=3, out_channels=128)
        self.res_blocks = nn.ModuleList([FiLMResBlock(128, 128) for _ in range(4)])
        self.classifier = FiLMClassifier(in_channels=128, num_answers=num_answers)

    def forward(self, img: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        """Prédit la réponse à une question à partir d'une image.

        Extrait les features de l'image, les modifie avec la question via les
        4 blocs FiLM, puis classe le résultat.

        Parameters
        ----------
        img : torch.Tensor
            Image de forme (B, 3, H, W).
        encoding : torch.Tensor
            Encodage de la question de forme (B, 10).

        Returns
        -------
        torch.Tensor
            Logits de forme (B, num_answers).
        """
        x = self.featuremap(img)
        for block in self.res_blocks:
            x = block(x, encoding)
        return self.classifier(x)


if __name__ == "__main__":
    pass
