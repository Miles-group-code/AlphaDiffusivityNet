"""Scale estimation utilities for diffusivity inference.

Provides:
1) DDI (data-driven initialization) as a fast heuristic.
2) Scalar fit for D = const using a differentiable FDM solver.
"""

from __future__ import annotations

from typing import Iterable, Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

import physics
import varpro

if TYPE_CHECKING:
    from data import PPPData

# Minimum allowed diffusivity to prevent numerical instability in FDM solver.
# Values below this threshold can cause division by near-zero in the tridiagonal system.
D_MIN = 1e-6


def estimate_ddi_scale(
    mu: float,
    z: float,
    x_particles: Optional[torch.Tensor] = None,
    u_field: Optional[torch.Tensor] = None,
    x_grid: Optional[torch.Tensor] = None,
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


def fit_constant_d(
    x: torch.Tensor,
    alpha: float,
    mu: float,
    sources: Iterable[float],
    u_true: Optional[torch.Tensor] = None,
    ppp: Optional["PPPData"] = None,
    x_field: Optional[torch.Tensor] = None,
    x_int: Optional[torch.Tensor] = None,
    d_init: float = 0.1,
    max_iters: int = 500,
    lr: float = 0.1,
    field_loss: str = "mse",
    bc_type: str = "dirichlet",
    verbose: bool = False,
) -> float:
    """Fit a constant D by minimizing data loss via differentiable FDM.

    Solves:
        min_D L_data(u_hat(D), data), with u_hat from FDM and b0 projected.

    Args:
        x: Solver grid for the FDM solve.
        alpha: Stochastic convention parameter.
        mu: Death rate.
        sources: Source locations.
        u_true: Observed field data (field mode).
        ppp: Particle data (PPP mode).
        x_field: Observation grid for u_true (defaults to x).
        x_int: Integration grid for PPP normalization (defaults to x).
        d_init: Initial guess for D.
        max_iters: Optimization iterations.
        lr: Adam learning rate.
        field_loss: "mse" or "rle" (field mode).
        bc_type: Boundary condition type ("dirichlet" or "neumann").
        verbose: Print optimization progress.

    Returns:
        Best-fit constant D value as a float.
    """
    if (u_true is None) == (ppp is None):
        raise ValueError("Provide exactly one of u_true or ppp.")

    device = x.device
    dtype = x.dtype

    x = x.view(-1)
    if x_field is not None:
        x_field = x_field.to(device=device, dtype=dtype).view(-1)
    if x_int is not None:
        x_int = x_int.to(device=device, dtype=dtype).view(-1)

    # Prepare observation data on the correct device/dtype
    if u_true is not None:
        u_true = u_true.to(device=device, dtype=dtype)
    if ppp is not None:
        # Flatten particle positions once here to avoid repeated view(-1) calls in the loop
        x_particles = ppp.x_particles.to(device=device, dtype=dtype).view(-1)
    else:
        x_particles = None

    # Clamp initial guess to valid range
    d_init_value = max(float(d_init), D_MIN)

    # Use a single trainable scalar parameter for the constant D value
    d_const = nn.Parameter(torch.tensor(d_init_value, device=device, dtype=dtype))

    # Adam works well for this 1D optimization problem; LBFGS would be overkill
    optimizer = torch.optim.Adam([d_const], lr=lr)

    # Track the best solution found (optimizer may overshoot and recover)
    best_loss = float("inf")
    best_d = d_init_value

    if verbose:
        mode_str = "field" if u_true is not None else "particles"
        print(f"[scalar_fit] Starting: d_init={d_init_value:.6f}, mode={mode_str}, max_iters={max_iters}")

    for step in range(max_iters):
        # set_to_none=True is slightly faster than zero_grad() for single-param optimization
        optimizer.zero_grad(set_to_none=True)

        d_vec = d_const.expand(x.numel())
        u_hat = physics.fdm_solve_alpha_torch(
            d_vec, alpha, mu, x, 1.0, sources, bc_type=bc_type
        )

        if u_true is not None:
            u_true_flat = u_true.view(-1)
            if x_field is None or (x_field.numel() == x.numel() and torch.allclose(x_field, x)):
                u_hat_field = u_hat
            else:
                u_hat_field = varpro.interpolate_1d(u_hat, x, x_field)
            b0_star = varpro.project_b0_field(u_hat_field, u_true_flat, field_loss=field_loss)
            loss = varpro.field_data_loss(
                u_hat_field, u_true_flat, b0_star, field_loss=field_loss
            )
        else:
            # Particles mode: interpolate u_hat to particle locations
            if x_particles.numel() == 0:
                # Handle edge case of no particles (empty tensor)
                u_hat_obs = x_particles
            else:
                u_hat_obs = varpro.interpolate_1d(u_hat, x, x_particles)

            # Compute integral for PPP normalization (on fine grid if provided)
            if x_int is None:
                integral_unit = torch.trapezoid(u_hat.view(-1), x.view(-1))
            else:
                u_hat_int = varpro.interpolate_1d(u_hat, x, x_int)
                integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))

            # VarPro projection and PPP negative log-likelihood
            b0_star = varpro.project_b0_ppp(ppp.n_obs, ppp.m_obs, integral_unit)
            loss = varpro.ppp_nll(u_hat_obs, b0_star, ppp.m_obs, integral_unit)

        # Backprop through the differentiable FDM solver
        loss.backward()
        optimizer.step()

        # Enforce positivity constraint via projection (D >= D_MIN)
        with torch.no_grad():
            d_const.clamp_(min=D_MIN)

        # Track best solution (loss may not decrease monotonically due to Adam momentum)
        loss_value = float(loss.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_d = float(d_const.item())

        if verbose and step % 100 == 0:
            print(
                f"[scalar_fit] iter {step}: D={float(d_const.item()):.6f}, loss={loss_value:.6e}"
            )

    if verbose:
        print(f"[scalar_fit] Final: D={best_d:.6f}, loss={best_loss:.6e}")

    return best_d
