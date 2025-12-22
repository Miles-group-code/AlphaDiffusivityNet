"""Variable-projection utilities for estimating b0 and data terms.

Includes differentiable interpolation and PPP likelihood helpers used across
methods, plus optional precomputation helpers for fixed interpolation grids.

Key entry points: interpolate_1d, project_b0_field, field_data_loss,
project_b0_ppp, ppp_nll.
"""

from __future__ import annotations

from typing import Literal

import torch


def interpolate_1d(
    y_src: torch.Tensor,
    x_src: torch.Tensor,
    x_dst: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Differentiable 1D linear interpolation from src grid to dst points.

    Assumes x_src is sorted ascending. Values outside the grid are clamped to
    the boundary. Intervals are located with torch.bucketize.

    Args:
        y_src: Source values on x_src.
        x_src: Source grid (1D, sorted).
        x_dst: Destination locations.
        eps: Clamp for zero-length intervals.

    Returns:
        Interpolated values shaped like x_dst.
    """
    x_src_flat = x_src.view(-1)
    y_src_flat = y_src.view(-1)
    x_dst_flat = x_dst.view(-1)
    if x_src_flat.numel() < 2:
        raise ValueError("x_src must have at least 2 points for interpolation.")

    x_min = x_src_flat[0]
    x_max = x_src_flat[-1]
    x_clamped = torch.clamp(x_dst_flat, x_min, x_max)

    idx = torch.bucketize(x_clamped, x_src_flat)
    idx_left = torch.clamp(idx - 1, 0, x_src_flat.numel() - 2)
    idx_right = idx_left + 1

    x_left = x_src_flat[idx_left]
    x_right = x_src_flat[idx_right]
    y_left = y_src_flat[idx_left]
    y_right = y_src_flat[idx_right]

    denom = (x_right - x_left).clamp_min(eps)
    w = (x_clamped - x_left) / denom
    y_dst_flat = (1.0 - w) * y_left + w * y_right
    return y_dst_flat.view(x_dst.shape)


def precompute_interp_1d(
    x_src: torch.Tensor,
    x_dst: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Size]:
    """Precompute 1D interpolation indices/weights for a fixed src/dst grid.

    This is useful when x_src/x_dst are constant across iterations; it avoids
    repeated torch.bucketize calls inside the training loop.
    """
    x_src_flat = x_src.view(-1)
    x_dst_flat = x_dst.view(-1)
    if x_src_flat.numel() < 2:
        raise ValueError("x_src must have at least 2 points for interpolation.")

    x_min = x_src_flat[0]
    x_max = x_src_flat[-1]
    x_clamped = torch.clamp(x_dst_flat, x_min, x_max)

    idx = torch.bucketize(x_clamped, x_src_flat)
    idx_left = torch.clamp(idx - 1, 0, x_src_flat.numel() - 2)
    idx_right = idx_left + 1

    x_left = x_src_flat[idx_left]
    x_right = x_src_flat[idx_right]
    denom = (x_right - x_left).clamp_min(eps)
    w = (x_clamped - x_left) / denom
    return idx_left, idx_right, w, x_dst.shape


def interpolate_1d_precomputed(
    y_src: torch.Tensor,
    idx_left: torch.Tensor,
    idx_right: torch.Tensor,
    w: torch.Tensor,
    out_shape: torch.Size,
) -> torch.Tensor:
    """Interpolate using precomputed indices/weights from precompute_interp_1d."""
    y_src_flat = y_src.view(-1)
    y_left = y_src_flat[idx_left]
    y_right = y_src_flat[idx_right]
    y_dst_flat = (1.0 - w) * y_left + w * y_right
    return y_dst_flat.view(out_shape)


def project_b0_field(
    u_hat: torch.Tensor,
    u_true: torch.Tensor,
    field_loss: Literal["mse", "rle"] = "mse",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Analytically project the optimal b0* for field data.

    For MSE, L = sum((b0 * u_hat - u_true)^2), the optimal projection is:
        b0* = <u_hat, u_true> / <u_hat, u_hat>

    For RLE, L = sum(((b0 * u_hat - u_true) / u_true)^2), this is weighted LS
    with w_i = 1 / u_true_i^2, giving:
        b0* = <w * u_hat, u_true> / <w * u_hat, u_hat>

    Note: b0* is NOT detached so gradients flow through the projection.
    """
    u_hat = u_hat.view(-1)
    u_true = u_true.view(-1)
    if field_loss == "mse":
        numerator = torch.sum(u_hat * u_true)
        denom = torch.sum(u_hat * u_hat) + eps
    elif field_loss == "rle":
        weights = 1.0 / (u_true + eps) ** 2
        numerator = torch.sum(weights * u_hat * u_true)
        denom = torch.sum(weights * u_hat * u_hat) + eps
    else:
        raise ValueError(f"Unsupported field_loss '{field_loss}'.")
    return numerator / denom


def field_data_loss(
    u_hat: torch.Tensor,
    u_true: torch.Tensor,
    b0_star: torch.Tensor,
    field_loss: Literal["mse", "rle"] = "mse",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the field data loss given b0_star."""
    resid = b0_star * u_hat - u_true
    if field_loss == "mse":
        return torch.mean(resid ** 2)
    if field_loss == "rle":
        denom = u_true + eps
        return torch.mean((resid / denom) ** 2)
    raise ValueError(f"Unsupported field_loss '{field_loss}'.")


def project_b0_ppp(
    n_obs: int,
    m_obs: int,
    integral_u_hat: torch.Tensor,
    clamp_min: float = 1e-12,
) -> torch.Tensor:
    """Project the optimal b0 for PPP observations."""
    denom = torch.clamp(integral_u_hat, min=clamp_min)
    return (float(n_obs) / float(m_obs)) / denom


def ppp_nll(
    u_hat_obs: torch.Tensor,
    b0_star: torch.Tensor,
    m_obs: int,
    integral_u_hat: torch.Tensor,
    clamp_min: float = 1e-12,
) -> torch.Tensor:
    """Compute the negative log-likelihood for PPP observations."""
    intensity_obs = torch.clamp(b0_star * u_hat_obs, min=clamp_min)
    nll = b0_star * float(m_obs) * integral_u_hat - torch.sum(torch.log(intensity_obs))
    return nll / float(m_obs)
