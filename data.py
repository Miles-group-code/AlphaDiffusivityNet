"""Data utilities for PPP sampling and SDE-based particle simulation.

Key structures:
- PPPData: particle positions and number of observations.
- estimate_ddi_scale(): data-driven diffusion scale initializer.
- sample_ppp_from_field(): PPP sampling from a field intensity.
- simulate_particles_alpha(): alpha-SDE simulation with births/deaths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass
class PPPData:
    """Poisson point process samples and observation count."""

    x_particles: torch.Tensor
    m_obs: int

    @property
    def n_obs(self) -> int:
        """Total number of particles across all observations."""
        return int(self.x_particles.numel())


def estimate_ddi_scale(
    mu: float,
    z: float,
    x_particles: torch.Tensor | None = None,
    u_field: torch.Tensor | None = None,
    x_grid: torch.Tensor | None = None,
    d_min: float = 1e-4,
    d_max: float = 10.0,
) -> float:
    """Estimate a diffusion scale from particles or field data.

    Data-driven initialization (DDI) uses the mean absolute deviation from the
    source location to set a plausible D scale: D ~ mu * MAD^2. This provides
    a stable starting point when D and b0 are only weakly identifiable.
    """
    if x_particles is not None and x_particles.numel() > 0:
        mad = torch.mean(torch.abs(x_particles - z)).item()
        d_est = mu * (mad ** 2)
    elif u_field is not None:
        if x_grid is None:
            x_grid = torch.linspace(
                0.0, 1.0, u_field.numel(), device=u_field.device, dtype=u_field.dtype
            )
        x_flat = x_grid.view(-1)
        u_flat = u_field.view(-1)
        mass = torch.trapezoid(u_flat, x_flat).item()
        mad = (
            torch.trapezoid(torch.abs(x_flat - z) * u_flat, x_flat).item()
            / (mass + 1e-9)
        )
        d_est = mu * (mad ** 2)
    else:
        d_est = 1.0
    return float(np.clip(d_est, d_min, d_max))


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
    positions via inverse-CDF on the trapezoidal mass.
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

    positions = []
    cdf = np.cumsum((u_flat[:-1] + u_flat[1:]) / 2.0 * np.diff(x_flat))
    cdf = np.concatenate([[0.0], cdf])
    cdf = cdf / cdf[-1]
    for _ in range(m_obs):
        n_draw = rng.poisson(integral_u)
        if n_draw == 0:
            continue
        u_samples = rng.random(n_draw)
        positions.append(np.interp(u_samples, cdf, x_flat))

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
