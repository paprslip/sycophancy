"""
src/sae_model.py

Top-K Sparse Autoencoder with optional subspace constraints.

Architecture follows Venhoff et al. (2025):
- Encoder: Linear → Bias → Top-K activation
- Decoder: constrained to a subspace of the residual stream
- n_features in [5, 50] (far smaller than residual stream dim)
  → forces SAE to find the MOST FUNDAMENTAL variance dimensions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TopKSAE(nn.Module):
    """
    Top-K Sparse Autoencoder.

    Args:
        d_model: dimension of input activations (residual stream size)
        n_features: number of SAE features (dictionary size) - keep in [5, 50]
        k: number of features active per forward pass (sparsity)
        normalize_decoder: whether to L2-normalize decoder columns (recommended)
    """

    def __init__(
        self,
        d_model: int,
        n_features: int,
        k: int,
        normalize_decoder: bool = True,
    ):
        super().__init__()
        assert 1 <= k <= n_features, f"k={k} must be <= n_features={n_features}"
        assert 2 <= n_features <= 50, (
            f"n_features={n_features} should be in [2, 50] to force high-level features. "
            "Larger values lead to memorization rather than concept discovery."
        )

        self.d_model = d_model
        self.n_features = n_features
        self.k = k
        self.normalize_decoder = normalize_decoder

        # Encoder: d_model → n_features
        self.W_enc = nn.Parameter(torch.empty(d_model, n_features))
        self.b_enc = nn.Parameter(torch.zeros(n_features))

        # Decoder: n_features → d_model  (restricted subspace)
        self.W_dec = nn.Parameter(torch.empty(n_features, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Pre-encoder bias (subtract mean activation)
        self.b_pre = nn.Parameter(torch.zeros(d_model))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    @torch.no_grad()
    def normalize_decoder_weights(self):
        """Call after each optimizer step to keep decoder columns unit-norm."""
        if self.normalize_decoder:
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, d_model]
        Returns:
            z_topk: [batch, n_features] sparse latent (top-k values, rest 0)
            z_pre:  [batch, n_features] pre-sparsity latent (for aux loss)
        """
        x_centered = x - self.b_pre
        z_pre = x_centered @ self.W_enc + self.b_enc  # [batch, n_features]

        # Top-K: keep only the k largest activations, zero the rest
        topk_vals, topk_idx = torch.topk(z_pre, self.k, dim=-1)
        z_topk = torch.zeros_like(z_pre)
        z_topk.scatter_(-1, topk_idx, F.relu(topk_vals))

        return z_topk, z_pre

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch, n_features]
        Returns:
            x_hat: [batch, d_model] reconstruction
        """
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> dict:
        """
        Full forward pass.

        Returns dict with:
            x_hat:    reconstruction
            z:        sparse latent
            z_pre:    dense pre-sparsity latent (for aux loss)
            loss:     total loss (reconstruction + aux + constraint)
            loss_rec: reconstruction loss
            loss_aux: auxiliary dead-feature loss
            loss_con: subspace constraint loss
        """
        z, z_pre = self.encode(x)
        x_hat = self.decode(z)
        return dict(x_hat=x_hat, z=z, z_pre=z_pre)

    def get_feature_directions(self) -> torch.Tensor:
        """Return decoder directions [n_features, d_model] (the actual 'concepts')."""
        return F.normalize(self.W_dec, dim=1)

    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: return feature activation magnitudes for a batch."""
        z, _ = self.encode(x)
        return z


class SAETrainer:
    """
    Trains a TopKSAE with:
      - MSE reconstruction loss
      - Auxiliary loss to prevent dead features
      - Optional subspace constraint loss
    """

    def __init__(
        self,
        sae: TopKSAE,
        lr: float = 1e-3,
        aux_loss_coeff: float = 0.03,
        constraint_loss_coeff: float = 0.1,
        constraint_type: str = "orthogonal",  # "none" | "orthogonal" | "hierarchical"
        taxonomy_groups: Optional[list] = None,  # for hierarchical constraint
    ):
        self.sae = sae
        self.aux_loss_coeff = aux_loss_coeff
        self.constraint_loss_coeff = constraint_loss_coeff
        self.constraint_type = constraint_type
        self.taxonomy_groups = taxonomy_groups  # list of lists of feature indices

        self.optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

        # Track feature usage (for detecting dead features)
        self._feature_usage = torch.zeros(sae.n_features)

    def compute_losses(
        self, x: torch.Tensor, fwd: dict
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute all losses.

        Returns total loss and a dict of individual losses for logging.
        """
        x_hat = fwd["x_hat"]
        z = fwd["z"]
        z_pre = fwd["z_pre"]

        # 1. Reconstruction loss (normalized MSE)
        x_norm = (x - x.mean(dim=0, keepdim=True))
        loss_rec = F.mse_loss(x_hat, x)

        # 2. Auxiliary loss: penalize features that never activate
        #    (ghost grads approach: use pre-topk activations for dead features)
        with torch.no_grad():
            self._feature_usage *= 0.999  # exponential decay
            self._feature_usage += (z > 0).float().mean(0).cpu()
            dead_mask = (self._feature_usage < 1e-4).to(x.device)

        if dead_mask.any():
            # Encourage dead features to activate via their pre-sparsity values
            dead_z_pre = z_pre[:, dead_mask]
            loss_aux = (dead_z_pre ** 2).mean()
        else:
            loss_aux = torch.tensor(0.0, device=x.device)

        # 3. Subspace constraint loss
        loss_con = self._constraint_loss()

        total = loss_rec + self.aux_loss_coeff * loss_aux + self.constraint_loss_coeff * loss_con

        return total, {
            "loss_total": total.item(),
            "loss_rec": loss_rec.item(),
            "loss_aux": loss_aux.item(),
            "loss_con": loss_con.item(),
        }

    def _constraint_loss(self) -> torch.Tensor:
        """
        Subspace constraint on decoder directions.

        orthogonal:    penalize pairwise dot products between feature directions
                       → features must point in different directions
                       → encourages monosemanticity

        hierarchical:  features within the same taxonomy group are allowed to share
                       a subspace; features across groups are penalized for alignment
                       → encodes your taxonomy structure into the SAE
        """
        W = self.sae.W_dec  # [n_features, d_model]
        W_norm = F.normalize(W, dim=1)  # unit vectors

        if self.constraint_type == "none":
            return torch.tensor(0.0, device=W.device)

        elif self.constraint_type == "orthogonal":
            # Gram matrix: [n_features, n_features]
            G = W_norm @ W_norm.T
            # Off-diagonal elements should be 0
            mask = ~torch.eye(self.sae.n_features, dtype=bool, device=W.device)
            return (G[mask] ** 2).mean()

        elif self.constraint_type == "hierarchical":
            if self.taxonomy_groups is None:
                return torch.tensor(0.0, device=W.device)

            loss = torch.tensor(0.0, device=W.device)
            n_groups = len(self.taxonomy_groups)

            # Between groups: features should be orthogonal
            for i in range(n_groups):
                for j in range(i + 1, n_groups):
                    gi = torch.tensor(self.taxonomy_groups[i], device=W.device)
                    gj = torch.tensor(self.taxonomy_groups[j], device=W.device)
                    Wi = W_norm[gi]  # [|gi|, d_model]
                    Wj = W_norm[gj]  # [|gj|, d_model]
                    # Cross-group similarity should be low
                    cross = (Wi @ Wj.T) ** 2
                    loss = loss + cross.mean()

            # Within groups: no penalty (features can share subspace)
            # But optionally add mild within-group orthogonality for diversity
            for group in self.taxonomy_groups:
                if len(group) > 1:
                    gi = torch.tensor(group, device=W.device)
                    Wi = W_norm[gi]
                    G = Wi @ Wi.T
                    mask = ~torch.eye(len(group), dtype=bool, device=W.device)
                    if mask.any():
                        loss = loss + 0.1 * (G[mask] ** 2).mean()  # weaker within-group

            return loss

        else:
            raise ValueError(f"Unknown constraint_type: {self.constraint_type}")

    def step(self, x: torch.Tensor) -> dict:
        """Single training step. Returns loss dict."""
        self.sae.train()
        self.optimizer.zero_grad()

        fwd = self.sae(x)
        loss, loss_dict = self.compute_losses(x, fwd)

        loss.backward()
        self.optimizer.step()

        # Keep decoder columns unit-norm
        self.sae.normalize_decoder_weights()

        return loss_dict

    @torch.no_grad()
    def variance_explained(self, x: torch.Tensor) -> float:
        """Compute fraction of variance explained by reconstruction."""
        self.sae.eval()
        fwd = self.sae(x)
        x_hat = fwd["x_hat"]
        ss_res = ((x - x_hat) ** 2).sum()
        ss_tot = ((x - x.mean(0, keepdim=True)) ** 2).sum()
        return 1 - (ss_res / ss_tot).item()
