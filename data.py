"""Data utilities for PPP sampling and SDE-based particle simulation.

Key structures:
- PPPData: particle positions and number of observations.
- estimate_ddi_scale(): data-driven diffusion scale initializer (re-exported from scale_estimation).
- sample_ppp_from_field(): PPP sampling from a field intensity.
- simulate_particles_alpha(): alpha-SDE simulation with births/deaths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

# Re-export estimate_ddi_scale for backward compatibility
# (it was originally in this module before being moved to scale_estimation.py)
from scale_estimation import estimate_ddi_scale

__all__ = [
    "PPPData",
    "estimate_ddi_scale",
    "sample_ppp_from_field",
    "simulate_particles_alpha",
]


@dataclass
class PPPData:
    """Poisson point process samples and observation count."""

    x_particles: torch.Tensor
    m_obs: int

    @property
    def n_obs(self) -> int:
        """Total number of particles across all observations."""
        return int(self.x_particles.numel())


def _invert_piecewise_linear_cdf(
    u_samples: np.ndarray,
    cdf: np.ndarray,
    x_flat: np.ndarray,
    u_flat: np.ndarray,
) -> np.ndarray:
    """Invert CDF assuming piecewise-linear PDF (consistent with trapezoidal CDF).

    For a cell [x_i, x_{i+1}] with u values [u_i, u_{i+1}], the PDF is linear:
        p(x) ∝ u_i + (u_{i+1} - u_i) * (x - x_i) / Δx

    The CDF within the cell is quadratic, so we use the quadratic formula to invert.
    This is consistent with how the likelihood evaluates u(x) via linear interpolation.

    The approach: normalize to a unit-mass problem within each cell.
    Let q = (U - cdf[i]) / (cdf[i+1] - cdf[i]) be the fraction through the cell.
    Then solve for t in [0, Δx] such that the normalized local CDF equals q.
    """
    # Find which cell each sample falls into
    # searchsorted gives the index where u_samples would be inserted to maintain order
    # We want the cell index, so subtract 1 and clamp to valid range
    cell_idx = np.searchsorted(cdf, u_samples, side="right") - 1
    cell_idx = np.clip(cell_idx, 0, len(x_flat) - 2)

    # Get cell properties
    x_left = x_flat[cell_idx]
    x_right = x_flat[cell_idx + 1]
    u_left = u_flat[cell_idx]
    u_right = u_flat[cell_idx + 1]
    cdf_left = cdf[cell_idx]
    cdf_right = cdf[cell_idx + 1]
    dx = x_right - x_left

    # Fraction through the cell (in [0, 1])
    cell_mass = cdf_right - cdf_left
    # Handle edge case of zero-mass cells
    cell_mass = np.maximum(cell_mass, 1e-30)
    q = (u_samples - cdf_left) / cell_mass
    q = np.clip(q, 0.0, 1.0)

    # For piecewise-linear PDF within cell:
    #   Local CDF: F(t) = u_left*t + (u_right-u_left)*t²/(2*dx)
    #   Cell mass: M = (u_left + u_right)*dx/2
    #   Normalized: F(t)/M = q
    #
    # Solving F(t) = q*M:
    #   u_left*t + (u_right-u_left)*t²/(2*dx) = q*(u_left+u_right)*dx/2
    #
    # This is quadratic: a*t² + b*t + c = 0 where:
    #   a = (u_right - u_left) / (2*dx)
    #   b = u_left
    #   c = -q * (u_left + u_right) * dx / 2
    #
    # Numerically stable solution:
    #   t = 2*|c| / (b + sqrt(b² + 4*a*|c|))  [taking positive root]

    u_sum = u_left + u_right
    du = u_right - u_left
    rhs = q * u_sum * dx / 2.0  # = |c| = q * cell_mass_unnormalized

    # Discriminant: b² + 4ac = u_left² + 4*(du/(2*dx))*rhs = u_left² + 2*du*rhs/dx
    discriminant = u_left * u_left + 2.0 * du * rhs / dx
    # Clamp to avoid sqrt of tiny negative due to floating point
    discriminant = np.maximum(discriminant, 0.0)

    # Numerically stable form that works even when du → 0:
    t = 2.0 * rhs / (u_left + np.sqrt(discriminant) + 1e-30)

    # Clamp t to [0, dx] for safety
    t = np.clip(t, 0.0, dx)

    return x_left + t


def sample_ppp_from_field(
    x_grid: torch.Tensor,
    u_field: torch.Tensor,
    m_obs: int,
    rng: np.random.Generator,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> PPPData:
    """Sample PPP particles from a (non-normalized) field intensity.

    Draws per-snapshot Poisson counts with mean integral(u_field) and samples
    positions via inverse-CDF. Uses quadratic CDF inversion within each cell
    to be consistent with piecewise-linear interpolation of u(x), which matches
    how the likelihood function evaluates u at particle locations.
    """
    if device is None:
        device = u_field.device
    if dtype is None:
        dtype = u_field.dtype
    x_flat = x_grid.view(-1).detach().cpu().numpy()
    u_flat = u_field.view(-1).detach().cpu().numpy()
    integral_u = np.trapz(u_flat, x_flat)
    if integral_u <= 0.0:
        return PPPData(x_particles=torch.empty(0, device=device, dtype=dtype), m_obs=m_obs)

    # Compute CDF at grid points using trapezoidal rule (same as before)
    cdf = np.cumsum((u_flat[:-1] + u_flat[1:]) / 2.0 * np.diff(x_flat))
    cdf = np.concatenate([[0.0], cdf])
    cdf = cdf / cdf[-1]

    positions = []
    for _ in range(m_obs):
        n_draw = rng.poisson(integral_u)
        if n_draw == 0:
            continue
        u_samples = rng.random(n_draw)
        # Use quadratic inversion instead of linear interpolation
        positions.append(_invert_piecewise_linear_cdf(u_samples, cdf, x_flat, u_flat))

    if not positions:
        return PPPData(x_particles=torch.empty(0, device=device, dtype=dtype), m_obs=m_obs)
    x_particles = torch.as_tensor(np.concatenate(positions), device=device, dtype=dtype)
    return PPPData(x_particles=x_particles, m_obs=m_obs)


def simulate_particles_alpha(
    d_func: Callable[[np.ndarray], np.ndarray],
    dprime_func: Callable[[np.ndarray], np.ndarray],
    z: float,
    birth_rate: float,
    death_rate: float,
    alpha: float,
    tmax: float,
    dt: float,
    rng: np.random.Generator,
    m_obs: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> PPPData:
    """Simulate particles under alpha diffusion with births and deaths.

    Uses Euler-Maruyama for the SDE dX = alpha * D'(X) dt + sqrt(2 D(X)) dW,
    with births at z and deaths as a Bernoulli survival per time step. The
    domain is [0, 1]; particles outside are discarded (absorbing boundary).

    Args:
        d_func: Callable returning D(x) for positions x.
        dprime_func: Callable returning dD/dx for positions x.
        z: Birth location in [0, 1].
        birth_rate: Births per unit time.
        death_rate: Deaths per unit time.
        alpha: Stochastic convention in [0, 1].
        tmax: Total simulation time (same units as dt).
        dt: Euler-Maruyama timestep.
        rng: NumPy random generator.
        m_obs: Number of independent snapshots to simulate.
        device: Torch device for output tensor.
        dtype: Torch dtype for output tensor.

    Returns:
        PPPData with concatenated particle positions and m_obs snapshots.
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    n_steps = int(round(tmax / dt))
    all_positions = []
    for _ in range(m_obs):
        particles = np.array([], dtype=float)
        for _ in range(n_steps):
            if particles.size > 0:
                d_vals = d_func(particles)
                dprime_vals = dprime_func(particles)
                noise = rng.standard_normal(particles.size)
                drift = alpha * dprime_vals * dt
                diffusion = np.sqrt(2.0 * d_vals * dt) * noise
                particles = particles + drift + diffusion
                particles = particles[(particles >= 0.0) & (particles <= 1.0)]

            n_births = rng.poisson(birth_rate * dt)
            if n_births > 0:
                particles = np.append(particles, np.ones(n_births) * z)

            if particles.size > 0:
                survival_prob = 1.0 - (death_rate * dt)
                mask = rng.random(particles.size) < survival_prob
                particles = particles[mask]
        all_positions.append(particles.copy())

    if all_positions:
        x_particles_np = np.concatenate(all_positions)
    else:
        x_particles_np = np.array([], dtype=float)
    x_particles = torch.as_tensor(x_particles_np, device=device, dtype=dtype)
    return PPPData(x_particles=x_particles, m_obs=m_obs)
