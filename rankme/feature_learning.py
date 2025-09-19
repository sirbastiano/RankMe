"""Feature learning metrics for representation quality assessment."""

from typing import Any, Optional

import torch

from rankme.base import StatelessMetric


class RankMe(StatelessMetric):
    """Compute the RankMe score for embeddings.

    The RankMe score quantifies how uniformly information is distributed across
    feature dimensions by measuring the normalized Shannon entropy of the
    singular-value spectrum of an embedding matrix Z ∈ R^{N×D}.

    Given singular values {s_i}_{i=1}^D of Z, define p_i = s_i / sum_j s_j and
    return R = -sum_i p_i * log(p_i) / log(D). By construction, R ∈ [0, 1].

    This implementation supports unbatched inputs of shape (N, D) and batched
    inputs of shape (B, N, D).

    Args:
        log_base: Base of the logarithm used in the entropy. The normalization
            uses the same base, so the score remains in [0, 1].
            If None, uses the natural logarithm.
        eps: Numerical floor to avoid log(0) and division by zero.
        center: If True, mean-center features along the batch axis before SVD:
            Z <- Z - mean(Z, dim=-2, keepdim=True). Default: False.
        detach: If True, computes score without tracking gradients.

    Returns:
        torch.Tensor: If input is (N, D), returns a scalar tensor.
                      If input is (B, N, D), returns shape (B,) with per-batch scores.

    Example:
        >>> import torch
        >>> from rankme import RankMe
        >>> Z = torch.randn(1024, 256)
        >>> rankme = RankMe(center=False)
        >>> score = rankme(Z)
        >>> float(score)  # normalized entropy in [0, 1]
        0.93
        
        >>> # Batched computation
        >>> Z_batch = torch.randn(8, 512, 128)
        >>> scores = rankme(Z_batch)  # shape (8,)
    """

    def __init__(
        self,
        log_base: Optional[float] = None,
        eps: float = 1e-12,
        center: bool = False,
        detach: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.log_base = log_base
        self.eps = eps
        self.center = center
        self.detach = detach

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Compute RankMe score.

        Args:
            Z: Embedding matrix of shape (N, D) or (B, N, D).

        Returns:
            torch.Tensor: Scalar (unbatched) or shape (B,) (batched) RankMe score.
            
        Raises:
            ValueError: If Z has invalid shape (not 2D or 3D).
        """
        if Z.dim() == 2:
            Z = Z.unsqueeze(0)  # -> (1, N, D)
            squeeze_back = True
        elif Z.dim() == 3:
            squeeze_back = False
        else:
            raise ValueError(
                f'Z must have shape (N,D) or (B,N,D), got {tuple(Z.shape)}'
            )

        if self.detach:
            Z = Z.detach()

        # Optional feature centering along the N (batch) axis
        if self.center:
            Z = Z - Z.mean(dim=1, keepdim=True)

        B, N, D = Z.shape

        # Compute singular values per batch item. Use svdvals for efficiency.
        # torch.linalg.svdvals supports batched inputs and returns shape (B, min(N, D)).
        s = torch.linalg.svdvals(Z)  # (B, min(N, D))

        # If N < D, pad singular values with zeros so we conceptually have D terms.
        if s.size(1) < D:
            pad = torch.zeros(B, D - s.size(1), dtype=s.dtype, device=s.device)
            s = torch.cat([s, pad], dim=1)

        # Convert to probability mass over D components
        s_sum = s.sum(dim=1, keepdim=True).clamp_min(self.eps)
        p = (s / s_sum).clamp_min(self.eps)

        # Entropy with chosen base; normalize by log(D) in the same base
        if self.log_base is None:
            logp = p.log()
            norm = torch.log(torch.tensor(D, dtype=p.dtype, device=p.device))
        else:
            base = torch.tensor(self.log_base, dtype=p.dtype, device=p.device)
            logp = p.log() / base.log()
            norm = (
                torch.tensor(D, dtype=p.dtype, device=p.device).log() / base.log()
            )

        H = -(p * logp).sum(dim=1)  # (B,)
        score = H / norm  # normalized entropy in [0, 1]

        return score.squeeze(0) if squeeze_back else score


class EffectiveRank(StatelessMetric):
    """Compute the effective rank of embedding matrices.
    
    The effective rank is computed as the exponential of the Shannon entropy
    of the normalized singular value spectrum. This gives an intuitive measure
    of the number of 'effective' dimensions being used.
    
    For a matrix with uniformly distributed singular values across D dimensions,
    the effective rank approaches D. For a rank-deficient matrix, it will be
    much smaller than the nominal rank.
    
    Args:
        eps: Numerical floor to avoid log(0) and division by zero.
        center: If True, mean-center features before SVD.
        detach: If True, computes without tracking gradients.
        
    Returns:
        torch.Tensor: Effective rank value(s). For input (N,D), returns scalar.
                      For input (B,N,D), returns shape (B,).
                      
    Example:
        >>> import torch
        >>> from rankme.feature_learning import EffectiveRank
        >>> Z = torch.randn(1000, 100)
        >>> eff_rank = EffectiveRank()
        >>> rank_value = eff_rank(Z)
        >>> float(rank_value)  # typically close to min(N, D) for random matrices
        99.2
    """
    
    def __init__(
        self,
        eps: float = 1e-12,
        center: bool = False,
        detach: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.eps = eps
        self.center = center
        self.detach = detach
        
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Compute effective rank.
        
        Args:
            Z: Embedding matrix of shape (N, D) or (B, N, D).
            
        Returns:
            torch.Tensor: Effective rank value(s).
        """
        if Z.dim() == 2:
            Z = Z.unsqueeze(0)
            squeeze_back = True
        elif Z.dim() == 3:
            squeeze_back = False
        else:
            raise ValueError(
                f'Z must have shape (N,D) or (B,N,D), got {tuple(Z.shape)}'
            )
            
        if self.detach:
            Z = Z.detach()
            
        if self.center:
            Z = Z - Z.mean(dim=1, keepdim=True)
            
        # Compute singular values
        s = torch.linalg.svdvals(Z)  # (B, min(N, D))
        
        # Normalize to probabilities
        s_sum = s.sum(dim=1, keepdim=True).clamp_min(self.eps)
        p = (s / s_sum).clamp_min(self.eps)
        
        # Compute Shannon entropy and convert to effective rank
        entropy = -(p * p.log()).sum(dim=1)
        effective_rank = entropy.exp()
        
        return effective_rank.squeeze(0) if squeeze_back else effective_rank


class SpectralEntropy(StatelessMetric):
    """Compute the spectral entropy of embedding matrices.
    
    This is the Shannon entropy of the normalized singular value spectrum,
    without the normalization by log(D) used in RankMe. This gives the
    raw entropy value in nats (natural log) or bits (log base 2).
    
    Args:
        log_base: Base of logarithm. If None, uses natural log (entropy in nats).
                  If 2, gives entropy in bits.
        eps: Numerical floor to avoid log(0) and division by zero.
        center: If True, mean-center features before SVD.
        detach: If True, computes without tracking gradients.
        
    Returns:
        torch.Tensor: Spectral entropy value(s).
        
    Example:
        >>> import torch
        >>> from rankme.feature_learning import SpectralEntropy
        >>> Z = torch.randn(1000, 100)
        >>> entropy = SpectralEntropy(log_base=2)  # entropy in bits
        >>> ent_value = entropy(Z)
        >>> float(ent_value)
        6.64  # entropy in bits
    """
    
    def __init__(
        self,
        log_base: Optional[float] = None,
        eps: float = 1e-12,
        center: bool = False,
        detach: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.log_base = log_base
        self.eps = eps
        self.center = center
        self.detach = detach
        
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Compute spectral entropy.
        
        Args:
            Z: Embedding matrix of shape (N, D) or (B, N, D).
            
        Returns:
            torch.Tensor: Spectral entropy value(s).
        """
        if Z.dim() == 2:
            Z = Z.unsqueeze(0)
            squeeze_back = True
        elif Z.dim() == 3:
            squeeze_back = False
        else:
            raise ValueError(
                f'Z must have shape (N,D) or (B,N,D), got {tuple(Z.shape)}'
            )
            
        if self.detach:
            Z = Z.detach()
            
        if self.center:
            Z = Z - Z.mean(dim=1, keepdim=True)
            
        # Compute singular values
        s = torch.linalg.svdvals(Z)  # (B, min(N, D))
        
        # Normalize to probabilities
        s_sum = s.sum(dim=1, keepdim=True).clamp_min(self.eps)
        p = (s / s_sum).clamp_min(self.eps)
        
        # Compute entropy
        if self.log_base is None:
            logp = p.log()
        else:
            base = torch.tensor(self.log_base, dtype=p.dtype, device=p.device)
            logp = p.log() / base.log()
            
        entropy = -(p * logp).sum(dim=1)
        
        return entropy.squeeze(0) if squeeze_back else entropy