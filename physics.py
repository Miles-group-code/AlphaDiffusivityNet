"""Physics and numerical helpers for the 1D alpha-PDE.

Includes flux-form finite-difference assembly, a NumPy Thomas solver for
ground-truth FDM, and regularizers for log D.

Key entry points: fdm_solve_alpha_dirichlet, h1_smoothness_logd,
tv_smoothness_logd, log_scale_anchor.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import torch


def _build_fdm_tridiag(
    logd: np.ndarray,
    alpha: float,
    mu: float,
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble tridiagonal coefficients for the alpha-PDE FDM system.

    Args:
        logd: Log diffusion values on the grid (length N).
        alpha: Stochastic convention in [0, 1].
        mu: Death rate.
        x: Uniform grid locations (length N).

    Returns:
        (lower, diag, upper) arrays for the interior system (length N-2).
    """
    n = x.size
    if n < 3:
        raise ValueError("Need at least 3 grid points for Dirichlet problem.")
    h = x[1] - x[0]

    g = np.exp((1.0 - alpha) * logd)
    logd_half = 0.5 * (logd[:-1] + logd[1:])
    d_half_alpha = np.exp(alpha * logd_half)
    a_half = d_half_alpha / h

    # Vectorized assembly
    # lower comes from a_half[i-1] * g[i-1] where i ranges 1..n-2 (indices of interior points)
    # interior points are at indices 1, 2, ..., n-2
    # lower[idx-1] corresponds to row corresponding to x[i] coupling with x[i-1]
    # For i=2 (first row with a lower coupling), we need a_half[1]*g[1]
    # lower array indices: 0..n-4 (length n-3)
    # logic in loop: if i > 1: lower[idx-1] = ...
    # i ranges 1 to n-2. i>1 means i ranges 2 to n-2.
    # idx = i-1 ranges 1 to n-3.
    # idx-1 ranges 0 to n-4.
    # We need a_half[1:n-2] * g[1:n-2]
    #
    # Let's map indices carefully from the original loop:
    # for i in range(1, n-1): (interior nodes 1 to n-2)
    #   idx = i - 1  (row index 0 to n-3)
    #   lower entry at idx-1 (exists if i > 1, i.e., rows 1 to n-3):
    #     lower[idx-1] = a_half[i-1] * g[i-1] / h
    #   upper entry at idx (exists if i < n-2, i.e., rows 0 to n-4):
    #     upper[idx] = a_half[i] * g[i+1] / h
    #   diag entry at idx:
    #     diag[idx] = -(a_half[i] + a_half[i-1]) * g[i] / h - mu
    #
    # Vectorized:
    # Diag (i=1..n-2):
    #   -(a_half[1:n-1] + a_half[0:n-2]) * g[1:n-1] / h - mu
    # Lower (i=2..n-2):
    #   a_half[1:n-2] * g[1:n-2] / h
    # Upper (i=1..n-3):
    #   a_half[1:n-2] * g[2:n-1] / h
    n_int = n - 2
    lower = np.zeros(n_int - 1, dtype=float)
    diag = np.zeros(n_int, dtype=float)
    upper = np.zeros(n_int - 1, dtype=float)
    for i in range(1, n - 1):
        idx = i - 1
        if i > 1:
            lower[idx - 1] = (a_half[i - 1] * g[i - 1]) / h
        if i < n - 2:
            upper[idx] = (a_half[i] * g[i + 1]) / h
        diag[idx] = -(a_half[i] + a_half[i - 1]) * g[i] / h - mu
    return lower, diag, upper


def _build_delta_rhs(
    x: np.ndarray,
    sources: Iterable[float],
    b0: float,
) -> np.ndarray:
    """Construct the interior RHS for point sources on a uniform grid.

    Each source contributes a -b0/h impulse to the nearest interior node.

    Args:
        x: Uniform grid locations (length N).
        sources: Source locations (assumed strictly inside the domain).
        b0: Source amplitude.

    Returns:
        RHS vector for interior nodes (length N-2).
    """
    n = x.size
    h = x[1] - x[0]
    rhs = np.zeros(n - 2, dtype=float)
    x0 = x[0]
    for z in sources:
        idx = int(round((z - x0) / h))
        idx = max(1, min(n - 2, idx))
        rhs[idx - 1] += -b0 / h
    return rhs


def _thomas_solve(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal system using the Thomas algorithm.

    NumPy implementation used for the ground-truth FDM solve; for the
    differentiable torch version see method_dto._thomas_solve.
    """
    n = diag.size
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return rhs / diag

    c_prime = np.zeros(n - 1, dtype=float)
    d_prime = np.zeros(n, dtype=float)

    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]
    for i in range(1, n - 1):
        den = diag[i] - lower[i - 1] * c_prime[i - 1]
        c_prime[i] = upper[i] / den
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / den
    den_last = diag[n - 1] - lower[n - 2] * c_prime[n - 2]
    d_prime[n - 1] = (rhs[n - 1] - lower[n - 2] * d_prime[n - 2]) / den_last

    x = np.zeros(n, dtype=float)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def fdm_solve_alpha_dirichlet(
    logd: Sequence[float],
    alpha: float,
    mu: float,
    x: Sequence[float],
    b0: float,
    sources: Iterable[float],
) -> np.ndarray:
    """Solve the steady 1D alpha-PDE on a uniform grid with Dirichlet BCs.

    Args:
        logd: Log diffusion values on the grid (length N).
        alpha: Stochastic convention in [0, 1].
        mu: Death rate.
        x: Uniform grid locations (length N).
        b0: Source amplitude.
        sources: Source locations (assumed strictly inside the domain).

    Returns:
        u(x) on the grid with boundary values set to zero.
    """
    logd = np.asarray(logd, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != logd.size:
        raise ValueError("logd and x must have the same length.")

    lower, diag, upper = _build_fdm_tridiag(logd, alpha, mu, x)
    rhs = _build_delta_rhs(x, sources, b0)

    u_int = _thomas_solve(lower, diag, upper, rhs)
    u = np.zeros_like(x)
    u[1:-1] = u_int
    return u


def h1_smoothness_logd(x: torch.Tensor, logd: torch.Tensor) -> torch.Tensor:
    """Compute H1 seminorm regularization using autograd gradients."""
    if not x.requires_grad:
        raise ValueError("x must have requires_grad=True for H1 regularization.")
    ones = torch.ones_like(logd)
    grad = torch.autograd.grad(logd, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    return torch.mean(grad ** 2)


def h1_smoothness_logd_discrete(logd: torch.Tensor, h: float = 1.0) -> torch.Tensor:
    """Discrete H1 regularization for log-D on a grid.

    NOTE: Use this with DTO, where logD is a raw parameter on grid nodes rather
    than a differentiable function of x. Autograd-based smoothness will not
    work when logd is indexed data rather than a function.
    """
    diffs = (logd[1:] - logd[:-1]) / h
    return torch.mean(diffs ** 2)


def tv_smoothness_logd(
    x: torch.Tensor,
    logd: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """TV regularization for log-D using a smoothed L1 penalty."""
    if not x.requires_grad:
        raise ValueError("x must have requires_grad=True for TV regularization.")
    ones = torch.ones_like(logd)
    grad = torch.autograd.grad(logd, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    return torch.mean(torch.sqrt(grad ** 2 + eps))


def tv_smoothness_logd_discrete(
    logd: torch.Tensor,
    h: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Discrete TV regularization for log-D on a grid."""
    diffs = (logd[1:] - logd[:-1]) / h
    return torch.mean(torch.sqrt(diffs ** 2 + eps))


def log_scale_anchor(logd: torch.Tensor, log_target: float) -> torch.Tensor:
    """Pointwise log-normal scale anchor that tethers log D to a target value."""
    return torch.mean((logd - log_target) ** 2)
