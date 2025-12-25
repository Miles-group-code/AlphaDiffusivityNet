"""Physics and numerical helpers for the 1D alpha-PDE.

Includes flux-form finite-difference assembly, a NumPy Thomas solver for
ground-truth FDM, and regularizers for D.

Key entry points: fdm_solve_alpha_dirichlet, h1_smoothness_d,
tv_smoothness_d, scale_anchor.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import torch


def _build_fdm_tridiag(
    d: np.ndarray,
    alpha: float,
    mu: float,
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble tridiagonal coefficients for the alpha-PDE FDM system.

    Handles non-uniform grids by computing local step sizes.

    Args:
        d: Diffusion values on the grid (length N).
        alpha: Stochastic convention in [0, 1].
        mu: Death rate.
        x: Grid locations (length N).

    Returns:
        (lower, diag, upper) arrays for the interior system (length N-2).
    """
    n = x.size
    if n < 3:
        raise ValueError("Need at least 3 grid points for Dirichlet problem.")
    
    # h[i] = x[i+1] - x[i]
    h = x[1:] - x[:-1]

    # Flux form: J = D^alpha * d/dx (D^(1-alpha) * u)
    # Let q = D^(1-alpha) * u. Then J = D^alpha * dq/dx.
    # We discretize q at nodes and J at half-nodes.
    
    g = d ** (1.0 - alpha)
    d_half = 0.5 * (d[:-1] + d[1:])
    
    # Flux coefficient at i+1/2: D_{i+1/2}^alpha / h_i
    a_half = (d_half ** alpha) / h

    # Vectorized assembly
    n_int = n - 2
    lower = np.zeros(n_int - 1, dtype=float)
    diag = np.zeros(n_int, dtype=float)
    upper = np.zeros(n_int - 1, dtype=float)
    
    # Voronoi volumes for interior nodes 1..N-2
    # vol[i] = (h[i-1] + h[i]) / 2  where i is global index
    # interior index k=0 corresponds to global i=1
    # vol[k] = (h[k] + h[k+1]) / 2
    
    vol = 0.5 * (h[:-1] + h[1:])

    for i in range(1, n - 1):
        idx = i - 1
        v = vol[idx]
        
        # Divergence is (J_{i+1/2} - J_{i-1/2}) / v
        
        if i > 1:
            lower[idx - 1] = (a_half[i - 1] * g[i - 1]) / v
        if i < n - 2:
            upper[idx] = (a_half[i] * g[i + 1]) / v
        
        diag[idx] = -(a_half[i] + a_half[i - 1]) * g[i] / v - mu
    return lower, diag, upper


def _build_delta_rhs(
    x: np.ndarray,
    sources: Iterable[float],
    b0: float,
) -> np.ndarray:
    """Construct the interior RHS for point sources on a non-uniform grid.

    Each source contributes -b0/vol to the nearest interior node,
    where vol is the Voronoi volume (average of adjacent steps).

    Args:
        x: Grid locations (length N).
        sources: Source locations (assumed strictly inside the domain).
        b0: Source amplitude.

    Returns:
        RHS vector for interior nodes (length N-2).
    """
    n = x.size
    h = x[1:] - x[:-1]
    vol = 0.5 * (h[:-1] + h[1:]) # Length N-2
    
    rhs = np.zeros(n - 2, dtype=float)
    
    for z in sources:
        # Find nearest interior node
        # Interior nodes are indices 1..N-2 in x
        idx_global = (np.abs(x - z)).argmin()
        
        # Force source to nearest interior node if it falls on boundary (unlikely given assumptions)
        idx_global = max(1, min(n - 2, idx_global))
        idx_int = idx_global - 1
        
        rhs[idx_int] += -b0 / vol[idx_int]
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
    d: Sequence[float],
    alpha: float,
    mu: float,
    x: Sequence[float],
    b0: float,
    sources: Iterable[float],
) -> np.ndarray:
    """Solve the steady 1D alpha-PDE on a uniform grid with Dirichlet BCs.

    Args:
        d: Diffusion values on the grid (length N).
        alpha: Stochastic convention in [0, 1].
        mu: Death rate.
        x: Uniform grid locations (length N).
        b0: Source amplitude.
        sources: Source locations (assumed strictly inside the domain).

    Returns:
        u(x) on the grid with boundary values set to zero.
    """
    d = np.asarray(d, dtype=float).reshape(-1)
    if np.any(d <= 0):
        raise ValueError("D must be strictly positive.")
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != d.size:
        raise ValueError("d and x must have the same length.")

    lower, diag, upper = _build_fdm_tridiag(d, alpha, mu, x)
    rhs = _build_delta_rhs(x, sources, b0)

    u_int = _thomas_solve(lower, diag, upper, rhs)
    u = np.zeros_like(x)
    u[1:-1] = u_int
    return u


def h1_smoothness_d(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Compute H1 seminorm regularization using autograd gradients."""
    if not x.requires_grad:
        raise ValueError("x must have requires_grad=True for H1 regularization.")
    ones = torch.ones_like(d)
    grad = torch.autograd.grad(d, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    return torch.mean(grad ** 2)


def h1_smoothness_d_discrete(d: torch.Tensor, h: float = 1.0) -> torch.Tensor:
    """Discrete H1 regularization for D on a grid."""
    diffs = (d[1:] - d[:-1]) / h
    return torch.mean(diffs ** 2)


def tv_smoothness_d(
    x: torch.Tensor,
    d: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """TV regularization for D using a smoothed L1 penalty."""
    if not x.requires_grad:
        raise ValueError("x must have requires_grad=True for TV regularization.")
    ones = torch.ones_like(d)
    grad = torch.autograd.grad(d, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    return torch.mean(torch.sqrt(grad ** 2 + eps))


def tv_smoothness_d_discrete(
    d: torch.Tensor,
    h: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Discrete TV regularization for D on a grid."""
    diffs = (d[1:] - d[:-1]) / h
    return torch.mean(torch.sqrt(diffs ** 2 + eps))


def scale_anchor(d: torch.Tensor, d_target: float) -> torch.Tensor:
    """Pointwise scale anchor that tethers D to a target value."""
    return torch.mean((d - d_target) ** 2)


def build_aligned_grid(
    domain: Tuple[float, float],
    n_res: int,
    z: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a grid that explicitly includes z as a grid point.

    This is used for PINN/BiLO where the source location must be exactly on the
    residual grid. The grid is constructed by splitting points proportionally
    on each side of z to maintain roughly uniform spacing throughout.

    Args:
        domain: (x_min, x_max) domain bounds.
        n_res: Total number of grid points.
        z: Source location to include as a grid point.
        device: Torch device.
        dtype: Torch dtype.

    Returns:
        1D tensor of grid points with z included.
    """
    if n_res < 3:
        raise ValueError("n_res must be >= 3.")
    x_min, x_max = domain
    if not (x_min < z < x_max):
        raise ValueError(f"z={z} must be strictly inside domain {domain}.")

    # Fraction of domain to the left of z
    frac = (z - x_min) / (x_max - x_min)
    # Allocate points proportionally (n_res - 1 intervals, plus 1 for endpoints)
    n_left = int(round(frac * (n_res - 1))) + 1
    n_right = n_res - n_left + 1  # +1 because z is shared

    # Ensure at least 2 points on each side
    n_left = max(2, min(n_res - 1, n_left))
    n_right = n_res - n_left + 1

    x_left = torch.linspace(x_min, z, n_left, device=device, dtype=dtype)
    x_right = torch.linspace(z, x_max, n_right, device=device, dtype=dtype)

    # Concatenate, excluding the duplicate z from x_right
    return torch.cat([x_left, x_right[1:]])
