"""Variable-projection utilities for estimating b0 and data terms.

Includes differentiable interpolation and PPP likelihood helpers used across
methods, plus optional precomputation helpers for fixed interpolation grids.

Key entry points: interpolate_1d, project_b0_field, field_data_loss,
project_b0_ppp, ppp_nll, get_b0_field, get_b0_ppp.

Fixed b0 mode:
    When the source amplitude b0 is known a priori (e.g., from experimental
    calibration), you can bypass VarPro projection by passing b0_fixed_value
    to get_b0_field() or get_b0_ppp(). This eliminates the amplitude-diffusivity
    ambiguity and can improve inference when b0 is well-characterized.
"""

from __future__ import annotations

from typing import Literal, Optional

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
    """Compute the PPP negative log-likelihood using a numerically stable form.

    Theory
    ------
    For a Poisson Point Process with intensity λ(x) = b₀·û(x), the NLL is:
        NLL = ∫λ(x)dx - Σᵢ log(λ(xᵢ))

    After VarPro projects out the optimal b₀* = n_obs / (m_obs · ∫û), the first
    term becomes constant: b₀* · m_obs · ∫û = n_obs. Expanding the log:
        NLL = n_obs - Σᵢ[log(b₀*) + log(û(xᵢ))]
            = n_obs - n_obs·log(n_obs/(m_obs·∫û)) - Σᵢ log(û(xᵢ))

    The D-dependent terms (up to additive constants) are:
        NLL_reduced = n_obs · log(∫û) - Σᵢ log(û(xᵢ))

    Numerical stability
    -------------------
    The naive form [b₀*·m·∫û - Σlog(b₀*·û)] subtracts two large numbers
    (~n_obs each), causing catastrophic cancellation in float32. The reduced
    form avoids this by computing only the D-dependent terms directly.

    Returns NLL/m_obs for per-snapshot scaling (consistent with original API).
    """
    # Reconstruct n_obs from b0_star. This is data (constant w.r.t. D), so detach.
    # n_obs = b0_star * m_obs * integral_u_hat
    n_obs = (b0_star * float(m_obs) * integral_u_hat).detach()

    # Reduced NLL: n_obs · log(∫û) - Σᵢ log(û(xᵢ))
    # Clamp to avoid log(0)
    log_integral = torch.log(torch.clamp(integral_u_hat, min=clamp_min))
    log_u_hat = torch.log(torch.clamp(u_hat_obs, min=clamp_min))

    nll = n_obs * log_integral - torch.sum(log_u_hat)

    return nll / float(m_obs)


def get_b0_field(
    u_hat: torch.Tensor,
    u_true: torch.Tensor,
    field_loss: Literal["mse", "rle"] = "mse",
    b0_fixed_value: Optional[float] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Get b0 for field data, either via VarPro projection or fixed value.

    This is the recommended entry point for obtaining b0 in field mode. It
    handles both the standard VarPro projection case and the fixed b0 case
    with a unified interface.

    Args:
        u_hat: Predicted unit-source response (shape [...]).
        u_true: Observed field data (same shape as u_hat).
        field_loss: Loss type for VarPro projection ("mse" or "rle").
            Ignored when b0_fixed_value is set.
        b0_fixed_value: If set to a positive value, use this fixed b0 instead
            of VarPro projection. If None (default), compute b0 via VarPro.
        device: Torch device for the fixed b0 tensor. If None, uses u_hat's device.
        dtype: Torch dtype for the fixed b0 tensor. If None, uses u_hat's dtype.
        eps: Small constant for numerical stability in VarPro projection.

    Returns:
        b0_star: Scalar tensor containing the optimal (or fixed) b0 value.
            When b0_fixed_value is None, gradients flow through the projection.
            When b0_fixed_value is set, the tensor is constant (no gradient flow).

    Raises:
        ValueError: If b0_fixed_value is provided but non-positive.

    Example:
        # Standard VarPro projection
        b0_star = get_b0_field(u_hat, u_true, field_loss="mse")

        # Fixed b0 (known from experiment)
        b0_star = get_b0_field(u_hat, u_true, b0_fixed_value=100.0)
    """
    if b0_fixed_value is not None:
        # Validate fixed value
        if b0_fixed_value <= 0:
            raise ValueError(f"b0_fixed_value must be positive, got {b0_fixed_value}")

        # Create fixed b0 tensor (constant - no gradient flow through b0)
        _device = device if device is not None else u_hat.device
        _dtype = dtype if dtype is not None else u_hat.dtype
        return torch.tensor(b0_fixed_value, device=_device, dtype=_dtype)
    else:
        # Standard VarPro projection
        return project_b0_field(u_hat, u_true, field_loss=field_loss, eps=eps)


def get_b0_ppp(
    n_obs: int,
    m_obs: int,
    integral_u_hat: torch.Tensor,
    b0_fixed_value: Optional[float] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    clamp_min: float = 1e-12,
) -> torch.Tensor:
    """Get b0 for PPP data, either via VarPro projection or fixed value.

    This is the recommended entry point for obtaining b0 in particles mode. It
    handles both the standard VarPro projection case and the fixed b0 case
    with a unified interface.

    Args:
        n_obs: Total number of observed particles across all snapshots.
        m_obs: Number of snapshots.
        integral_u_hat: Integral of unit-source response (∫u_hat dx).
        b0_fixed_value: If set to a positive value, use this fixed b0 instead
            of VarPro projection. If None (default), compute b0 via VarPro.
        device: Torch device for the fixed b0 tensor. If None, uses
            integral_u_hat's device.
        dtype: Torch dtype for the fixed b0 tensor. If None, uses
            integral_u_hat's dtype.
        clamp_min: Minimum value for clamping integral_u_hat to avoid division
            by zero in VarPro projection.

    Returns:
        b0_star: Scalar tensor containing the optimal (or fixed) b0 value.
            When b0_fixed_value is None, gradients flow through the projection.
            When b0_fixed_value is set, the tensor is constant (no gradient flow).

    Raises:
        ValueError: If b0_fixed_value is provided but non-positive.

    Example:
        # Standard VarPro projection
        b0_star = get_b0_ppp(n_obs, m_obs, integral_u_hat)

        # Fixed b0 (known from experiment)
        b0_star = get_b0_ppp(n_obs, m_obs, integral_u_hat, b0_fixed_value=100.0)
    """
    if b0_fixed_value is not None:
        # Validate fixed value
        if b0_fixed_value <= 0:
            raise ValueError(f"b0_fixed_value must be positive, got {b0_fixed_value}")

        # Create fixed b0 tensor (constant - no gradient flow through b0)
        _device = device if device is not None else integral_u_hat.device
        _dtype = dtype if dtype is not None else integral_u_hat.dtype
        return torch.tensor(b0_fixed_value, device=_device, dtype=_dtype)
    else:
        # Standard VarPro projection
        return project_b0_ppp(n_obs, m_obs, integral_u_hat, clamp_min=clamp_min)
