"""Normalization Perturbation (NP) and NP+ from Fan et al., ICLR 2023.

The implementation follows Appendix B of "Towards Robust Object Detection
Invariant to Real-World Domain Shifts".  It has no learnable parameters and
is intended to be called only during training.
"""

from __future__ import annotations

import torch
from torch import nn


class NormalizationPerturbation(nn.Module):
    """Perturb per-image channel statistics without changing feature shape."""

    SUPPORTED_METHODS = {'np', 'np_plus'}

    def __init__(self, method='np', noise_std=0.75,
                 mean_scale_factor=1.5, eps=1e-6):
        super().__init__()
        method = str(method).lower()
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f'Unsupported normalization perturbation method: {method}'
            )
        self.method = method
        self.noise_std = float(noise_std)
        self.mean_scale_factor = float(mean_scale_factor)
        self.eps = float(eps)
        self.last_used_fallback = False

    def _np_beta(self, feature_mean):
        return torch.normal(
            mean=torch.ones_like(feature_mean),
            std=self.noise_std * torch.ones_like(feature_mean),
        )

    def _np_plus_beta(self, feature_mean):
        # Appendix B uses the sample standard deviation (the historical
        # torch.std default) over the current process-local mini-batch.
        if feature_mean.shape[0] < 2:
            self.last_used_fallback = True
            mean_scale = torch.zeros_like(feature_mean[:1])
        else:
            mean_diff = torch.std(
                feature_mean, dim=0, keepdim=True, unbiased=True
            )
            maximum = mean_diff.max()
            if not torch.isfinite(maximum) or maximum <= self.eps:
                self.last_used_fallback = True
                mean_scale = torch.zeros_like(mean_diff)
            else:
                self.last_used_fallback = False
                mean_scale = (
                    mean_diff / maximum * self.mean_scale_factor
                )
        noise = torch.normal(
            mean=torch.zeros_like(feature_mean),
            std=self.noise_std * torch.ones_like(feature_mean),
        )
        return 1.0 + noise * mean_scale

    def forward(self, features):
        """Apply Appendix-B NP/NP+ to ``[B, C, H, W]`` features."""
        if features.ndim != 4:
            raise ValueError(
                f'Expected [B, C, H, W], got shape={tuple(features.shape)}'
            )
        feature_mean = features.mean(dim=(2, 3), keepdim=True)
        alpha = torch.normal(
            mean=torch.ones_like(feature_mean),
            std=self.noise_std * torch.ones_like(feature_mean),
        )
        beta = (
            self._np_beta(feature_mean)
            if self.method == 'np'
            else self._np_plus_beta(feature_mean)
        )
        return (
            alpha * features
            - alpha * feature_mean
            + beta * feature_mean
        )

    def forward_tokens(self, tokens, hw_shape):
        """Perturb Swin tokens while preserving ``[B, H*W, C]`` layout."""
        if tokens.ndim != 3:
            raise ValueError(
                f'Expected [B, H*W, C], got shape={tuple(tokens.shape)}'
            )
        height, width = int(hw_shape[0]), int(hw_shape[1])
        batch_size, length, channels = tokens.shape
        if length != height * width:
            raise ValueError(
                f'Token length {length} does not match H*W={height * width}'
            )
        features = tokens.transpose(1, 2).reshape(
            batch_size, channels, height, width
        )
        perturbed = self.forward(features)
        return perturbed.flatten(2).transpose(1, 2).contiguous()
