"""FiLM Layer - Feature-wise Linear Modulation.

Implements the core FiLM operation: gamma * x + beta applied channel-wise
to feature maps, conditioning them on external information (e.g., a question).
"""

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """Applies Feature-wise Linear Modulation: gamma * x + beta.

    Parameters
    ----------
    None — this layer has no learnable parameters of its own.
    gamma and beta are provided externally by the FiLM generator.

    References
    ----------
    Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", 2018.
    https://arxiv.org/abs/1709.07871
    """

    def forward(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FiLM modulation.

        Parameters
        ----------
        x : torch.Tensor of shape (B, C, H, W)
            Input feature maps.
        gamma : torch.Tensor of shape (B, C)
            Per-channel scaling factors.
        beta : torch.Tensor of shape (B, C)
            Per-channel bias factors.

        Returns
        -------
        torch.Tensor of shape (B, C, H, W)
            Modulated feature maps.
        """
        # Reshape to (B, C, 1, 1) for spatial broadcast
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta
