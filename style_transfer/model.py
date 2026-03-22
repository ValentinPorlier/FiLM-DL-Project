"""Architecture pour le transfert de style conditionnel via FiLM.

Pipeline :
    Image contenu  →  Encoder CNN  →  5 × ResBlock FiLM  →  Decoder CNN  →  Image stylisée

L'image de style est encodée par InceptionMixed6e (jusqu'à Mixed_6e) pour générer
les paramètres FiLM (γ, β) injectés dans chaque bloc via Instance Normalisation.
"""


import torch
import torch.nn as nn
import torchvision.models as models


class InceptionMixed6e(nn.Module):
    """Encodeur de style basé sur InceptionV3 tronqué après Mixed_6e.

    Les poids sont gelés (pré-entraînés ImageNet).
    Entrée : (B, 3, 256, 256) → Sortie : (B, 768, 14, 14)
    """

    def __init__(self) -> None:
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        inception.eval()
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode l'image de style.

        Parameters
        ----------
        x : (B, 3, 256, 256)

        Returns
        -------
        (B, 768, 14, 14)
        """
        return self.features(x)


class FiLMGenerator(nn.Module):
    """Génère les paramètres FiLM (γ, β) à partir d'une image de style.

    Pipeline : InceptionMixed6e → Global Average Pooling → Bottleneck MLP → (B, 2758)
    """

    def __init__(self) -> None:
        super().__init__()
        self.inception = InceptionMixed6e()
        self.bottleneck = nn.Sequential(
            nn.Linear(768, 100),
            nn.ReLU(),
            nn.Linear(100, 2758),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne un vecteur de paramètres FiLM.

        Parameters
        ----------
        x : (B, 3, 256, 256) — image de style

        Returns
        -------
        (B, 2758) — concaténation de tous les γ et β
        """
        parms = self.inception(x)                            # (B, 768, 14, 14)
        # Global Average Pooling → (B, 768, 1, 1)
        parms = torch.mean(parms, dim=[2, 3], keepdim=True)
        parms = self.bottleneck(parms.squeeze())             # (B, 768) → (B, 2758)
        return parms


class StyleTransferNetwork(nn.Module):
    """Réseau de transfert de style conditionnel.

    Architecture :
        Encoder : Conv9×9 → Conv3×3(stride 2) → Conv3×3(stride 2)
        5 × ResBlock FiLM : deux Conv3×3 par bloc avec Instance Norm FiLM
        Decoder : 2 × Upsample + Conv FiLM
        Sortie : Conv9×9 + FiLM → Sigmoid
    """

    def __init__(self) -> None:
        super().__init__()
        self.FiLMGenerator = FiLMGenerator()

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding="same")
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        # 5 blocs résiduels (2 convolutions chacun)
        self.res1conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same")
        self.res1conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same")
        self.res2conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same")
        self.res2conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same")
        self.res3conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same")
        self.res3conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same")
        self.res4conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same")
        self.res4conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same")
        self.res5conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same")
        self.res5conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same")

        # Decoder
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.convUpsample1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding="same")
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.convUpsample2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding="same")
        self.convfinal = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding="same")

    def apply_film(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Applique Instance Normalisation + modulation FiLM.

        Parameters
        ----------
        x     : (B, C, H, W)
        gamma : (B, C, 1, 1)
        beta  : (B, C, 1, 1)

        Returns
        -------
        (B, C, H, W)
        """
        eps = 1e-5
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True)
        return gamma * (x - mean) / torch.sqrt(var + eps) + beta

    def forward(self, x: torch.Tensor, vecteur_style: torch.Tensor) -> torch.Tensor:
        """Génère l'image stylisée.

        Parameters
        ----------
        x             : (B, 3, H, W) — image de contenu
        vecteur_style : (B, 3, H, W) — image de style

        Returns
        -------
        (B, 3, H, W) — image stylisée (valeurs ∈ [0, 1])
        """
        params = self.FiLMGenerator(vecteur_style).unsqueeze(
            2).unsqueeze(3)  # (B, 2758, 1, 1)
        pointer = 0

        def get_next_params(num_channels: int, ptr: int) -> tuple:
            """Extrait (gamma, beta, nouveau_pointeur) depuis le vecteur params."""
            mid = ptr + num_channels
            end = mid + num_channels
            return params[:, ptr:mid, :, :], params[:, mid:end, :, :], end

        # Encoder
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Bloc 1
        res = x
        x = self.res1conv1(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = torch.relu(self.apply_film(x, gamma, beta))
        x = self.res1conv2(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta) + res

        # Bloc 2
        res = x
        x = self.res2conv1(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = torch.relu(self.apply_film(x, gamma, beta))
        x = self.res2conv2(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta) + res

        # Bloc 3
        res = x
        x = self.res3conv1(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = torch.relu(self.apply_film(x, gamma, beta))
        x = self.res3conv2(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta) + res

        # Bloc 4
        res = x
        x = self.res4conv1(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = torch.relu(self.apply_film(x, gamma, beta))
        x = self.res4conv2(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta) + res

        # Bloc 5
        res = x
        x = self.res5conv1(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = torch.relu(self.apply_film(x, gamma, beta))
        x = self.res5conv2(x)
        gamma, beta, pointer = get_next_params(128, pointer)
        x = self.apply_film(x, gamma, beta) + res

        # Decoder
        x = self.upsample1(x)
        x = self.convUpsample1(x)
        gamma, beta, pointer = get_next_params(64, pointer)
        x = torch.relu(self.apply_film(x, gamma, beta))

        x = self.upsample2(x)
        x = self.convUpsample2(x)
        gamma, beta, pointer = get_next_params(32, pointer)
        x = torch.relu(self.apply_film(x, gamma, beta))

        x = self.convfinal(x)
        gamma, beta, pointer = get_next_params(3, pointer)
        return torch.sigmoid(self.apply_film(x, gamma, beta))


class VGGExtractor(nn.Module):
    """Extracteur de features VGG16 pour le calcul des losses contenu/style.

    Retourne les activations aux couches relu1_2, relu2_2, relu3_3, relu4_2
    (indices ``'3'``, ``'8'``, ``'15'``, ``'22'``).
    Les poids sont gelés (pré-entraînés ImageNet).
    """

    def __init__(self) -> None:
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg = vgg[:23]  # s'arrête après relu4_2
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        self.style_idx = ["3", "8", "15", "22"]
        self.content_idx = ["22"]  # relu4_2

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Retourne les feature maps aux couches cibles.

        Parameters
        ----------
        x : (B, 3, H, W)

        Returns
        -------
        dict[str, Tensor]
        """
        features: dict[str, torch.Tensor] = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.style_idx or name in self.content_idx:
                features[name] = x
        return features


def get_gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """Calcule la matrice de Gram normalisée d'un tenseur de features.

    Parameters
    ----------
    tensor : (B, C, H, W)

    Returns
    -------
    (B, C, C) — matrice de Gram normalisée par C × H × W
    """
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)


def compute_loss(
    vgg: VGGExtractor,
    gen_img: torch.Tensor,
    content_img: torch.Tensor,
    style_img: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calcule les losses de contenu et de style via VGG16.

    Parameters
    ----------
    vgg         : extracteur VGG16 gelé
    gen_img     : image générée (B, 3, H, W)
    content_img : image de contenu (B, 3, H, W)
    style_img   : image de style (B, 3, H, W)

    Returns
    -------
    tuple[Tensor, Tensor]
        (content_loss, style_loss)
    """
    gen_feats = vgg(gen_img)
    cont_feats = vgg(content_img)
    style_feats = vgg(style_img)

    mse = nn.MSELoss()

    content_loss = mse(gen_feats["22"], cont_feats["22"])

    style_layers = ["3", "8", "15"]
    style_loss = sum(
        mse(get_gram_matrix(gen_feats[idx]), get_gram_matrix(style_feats[idx]))
        for idx in style_layers
    ) / len(style_layers)

    return content_loss, style_loss


if __name__ == "__main__":
    pass
