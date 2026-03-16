"""FiLM Generator — encode la question et produit (γ, β) par bloc résiduel.

Correspond à film_gen.py du dépôt original ethanjperez/film.

Architecture :
    question tokens → Embedding → GRU → Linear → (γ_k, β_k) pour k = 0..N-1

Le décodeur linéaire produit un vecteur de taille N × 2C, qui est découpé en
N paires (γ_k, β_k) de dimension C (= num_channels).

Formulation résiduelle : γ = 1 + Δγ  (initialisé proche de 1, identité par défaut)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    """Encodeur GRU qui prédit les paramètres FiLM pour chaque bloc visuel.

    Parameters
    ----------
    vocab_size    : taille du vocabulaire de questions (incluant <pad>=0)
    embedding_dim : dimension des embeddings de mots
    hidden_dim    : dimension cachée du GRU
    rnn_num_layers: nombre de couches GRU
    rnn_dropout   : dropout inter-couches GRU (ignoré si rnn_num_layers=1)
    num_blocks    : nombre de blocs FiLMedResBlock
    num_channels  : nombre de canaux de chaque bloc (= C)

    Sortie de forward()
    -------------------
    Liste de num_blocks tuples (gamma_k, beta_k), chacun de shape (B, C).
    gamma_k = 1 + Δγ_k  (formulation résiduelle)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 200,
        hidden_dim: int = 4096,
        rnn_num_layers: int = 1,
        rnn_dropout: float = 0.0,
        num_blocks: int = 4,
        num_channels: int = 128,
    ) -> None:
        super().__init__()

        self.num_blocks   = num_blocks
        self.num_channels = num_channels

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_num_layers,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Décodeur linéaire : produit Δγ et β pour chaque bloc
        # Taille de sortie : num_blocks × 2 × num_channels
        self.decoder = nn.Linear(hidden_dim, num_blocks * 2 * num_channels)

    def forward(
        self,
        questions: torch.Tensor,
        lengths: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        questions : (B, T)  indices de tokens, paddés à 0
        lengths   : (B,)    longueurs réelles de chaque question

        Returns
        -------
        film_params : liste de num_blocks tuples (gamma_k, beta_k)
                      gamma_k, beta_k de shape (B, num_channels)
                      gamma_k = 1 + delta_gamma  (formulation résiduelle)
        """
        B = questions.size(0)

        emb = self.embedding(questions)  # (B, T, embedding_dim)

        # Pack pour ignorer le padding
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)  # h : (num_layers, B, hidden_dim)
        h = h[-1]                # (B, hidden_dim)  — dernière couche

        # Décodage → (B, num_blocks * 2 * num_channels)
        out = self.decoder(h)
        out = out.view(B, self.num_blocks, 2 * self.num_channels)

        # Découper en (delta_gamma, beta) par bloc
        film_params = []
        for k in range(self.num_blocks):
            delta_gamma = out[:, k, :self.num_channels]               # (B, C)
            beta        = out[:, k, self.num_channels:]               # (B, C)
            gamma       = 1.0 + delta_gamma                           # résiduel
            film_params.append((gamma, beta))

        return film_params
