"""DTO (direct-to-optimization) solver for the 1D alpha-PDE inverse problem.

Defines DTOData/DTOResult, a differentiable tridiagonal solver, and the
single-level optimization loop for grid-parameterized D.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from data import PPPData
import physics, varpro
from scale_estimation import estimate_ddi_scale, fit_constant_d
from training_logger import TrainingHistory, format_dto_progress


# =============================================================================
# D PARAMETERIZATION: DIRECT + PROJECTION
# =============================================================================
# We parameterize D(x) directly rather than through softplus or exp(logD).
#
# RATIONALE: Both softplus and logD parameterizations have Jacobians proportional
# to D itself, which causes catastrophic gradient suppression for small D values:
#
#   - logD: D = exp(theta) -> dD/dtheta = D
#   - softplus: D = softplus(theta) -> dD/dtheta = sigmoid(theta) ~ D (for small D)
#
# When D = 0.01, the gradient w.r.t. theta is suppressed by 100x. Combined with the
# already-flat Output Error landscape (see dto_issues.md), this makes optimization
# nearly impossible--LBFGS reports param_change -> 0 within ~20 steps.
#
# SOLUTION: Use direct parameterization with projection:
#   - Store theta = D - D_min as the parameter
#   - Forward: D = theta + D_min (dD/dtheta = 1, no suppression!)
#   - After step: clamp theta >= 0 to enforce D >= D_min
# =============================================================================

D_MIN = 1e-6


@dataclass
class DTOData:
    """Input bundle for DTO training.

    Fields:
        mode: "field" or "particles".
        x_res: Residual/solver grid (1D).
        x_field: Observation grid for field mode (1D).
        u_true: Observed field values on x_field (field mode).
        ppp: Particle observations for PPP mode (particles mode).
    """

    mode: str  # "field" or "particles"
    x_res: torch.Tensor
    x_field: Optional[torch.Tensor] = None  # Field observation grid (field mode)
    u_true: Optional[torch.Tensor] = None
    ppp: Optional[PPPData] = None


@dataclass
class DTOResult:
    """Outputs and training history for DTO fitting.

    Returned tensors are detached and on CPU for convenience.

    Fields:
        x_res: Solver grid (1D).
        d_pred: D(x) on x_res (1D).
        u_hat_unit: Unit-source response u_hat for b0=1 on x_res (1D).
        u_pred: Predicted field b0* * u_hat_unit on x_res (1D).
        b0_star: Projected source amplitude.
        history: Loss curves and diagnostics recorded every log_every steps.
            Typical keys: iter, total, data, reg_smooth, reg_scale, b0_star,
            mean_d, d_snap_iters, d_snapshots.
    """

    x_res: torch.Tensor
    d_pred: torch.Tensor
    u_hat_unit: torch.Tensor
    u_pred: torch.Tensor
    b0_star: float
    history: Dict[str, List[float]]


def _init_d_profile(
    x: torch.Tensor,
    base: float,
    scale: float,
    freq: float,
) -> torch.Tensor:
    """Build a sinusoidal D initialization on the grid."""
    if scale >= 1.0:
        raise ValueError("pert_scale must be < 1 to keep D_init positive.")
    d_init = base * (1.0 + scale * torch.sin(2.0 * torch.pi * freq * x))
    return d_init


def _thomas_solve(
    lower: torch.Tensor,
    diag: torch.Tensor,
    upper: torch.Tensor,
    rhs: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Solve a tridiagonal system with the Thomas algorithm."""
    n = diag.shape[0]
    if n == 1:
        return rhs / (diag + eps)

    # NOTE: Use Python lists to avoid in-place writes on preallocated tensors,
    # which can break autograd for tridiagonal solves.
    c_prime = []
    d_prime = []

    den0 = diag[0]
    c_prime.append(upper[0] / (den0 + eps))
    d_prime.append(rhs[0] / (den0 + eps))

    for i in range(1, n - 1):
        den = diag[i] - lower[i - 1] * c_prime[i - 1]
        c_prime.append(upper[i] / (den + eps))
        d_prime.append((rhs[i] - lower[i - 1] * d_prime[i - 1]) / (den + eps))

    den_last = diag[n - 1] - lower[n - 2] * c_prime[n - 2]
    d_prime.append((rhs[n - 1] - lower[n - 2] * d_prime[n - 2]) / (den_last + eps))

    x_list = [None] * n
    x_list[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x_list[i] = d_prime[i] - c_prime[i] * x_list[i + 1]
    return torch.stack(x_list, dim=0)


def _build_tridiag_alpha(
    d: torch.Tensor,
    alpha: float,
    mu: float,
    h: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble tridiagonal coefficients for the alpha-PDE discretization.

    h is now a vector of step sizes: h[i] = x[i+1] - x[i].

    NOTE: We use the harmonic mean of D^alpha at cell interfaces rather than
    (arithmetic_mean(D))^alpha. This preserves flux continuity across interfaces
    when D is discontinuous (e.g., step profiles). For smooth D, both approaches
    give second-order accuracy, but the harmonic mean is more physically correct
    for heterogeneous media.
    """
    g = d ** (1.0 - alpha)

    # Harmonic mean of D^alpha at cell interfaces for flux continuity
    d_alpha_left = d[:-1] ** alpha
    d_alpha_right = d[1:] ** alpha
    a_half = 2.0 * d_alpha_left * d_alpha_right / (d_alpha_left + d_alpha_right + eps) / h

    # Interior volumes: vol[i] = (h[i] + h[i+1]) / 2
    vol = 0.5 * (h[:-1] + h[1:])

    diag = -((a_half[1:] + a_half[:-1]) * g[1:-1] / vol) - mu
    lower_full = (a_half[:-1] * g[:-2]) / vol
    upper_full = (a_half[1:] * g[2:]) / vol
    lower = lower_full[1:]
    upper = upper_full[:-1]
    return lower, diag, upper


def _build_tridiag_alpha_neumann(
    d: torch.Tensor,
    alpha: float,
    mu: float,
    h: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble Neumann tridiagonal coefficients (full grid, zero flux).

    Uses half-cell control volumes at the boundaries to enforce zero flux,
    while keeping the interior stencil identical to the Dirichlet scheme.
    """
    g = d ** (1.0 - alpha)

    d_alpha_left = d[:-1] ** alpha
    d_alpha_right = d[1:] ** alpha
    a_half = 2.0 * d_alpha_left * d_alpha_right / (d_alpha_left + d_alpha_right + eps) / h

    # Volumes: half-cell at boundaries, full-cell in interior
    vol_0 = h[0] / 2.0
    vol_int = 0.5 * (h[:-1] + h[1:])
    vol_n1 = h[-1] / 2.0

    # Diagonal
    diag_0 = -(a_half[0] * g[0]) / vol_0 - mu
    diag_int = -((a_half[1:] + a_half[:-1]) * g[1:-1] / vol_int) - mu
    diag_n1 = -(a_half[-1] * g[-1]) / vol_n1 - mu
    diag = torch.cat([diag_0.unsqueeze(0), diag_int, diag_n1.unsqueeze(0)])

    # Upper diagonal
    upper_0 = (a_half[0] * g[1]) / vol_0
    upper_int = (a_half[1:] * g[2:]) / vol_int
    upper = torch.cat([upper_0.unsqueeze(0), upper_int])

    # Lower diagonal
    lower_int = (a_half[:-1] * g[:-2]) / vol_int
    lower_n1 = (a_half[-1] * g[-2]) / vol_n1
    lower = torch.cat([lower_int, lower_n1.unsqueeze(0)])

    return lower, diag, upper


class DtoAlphaVarPro(nn.Module):
    """Finite-difference solver with a learnable D profile (Dirichlet/Neumann)."""

    def __init__(
        self,
        x_res: torch.Tensor,
        alpha: float,
        mu: float,
        sources: List[float],
        d_init: torch.Tensor,
        d_min: float = D_MIN,
        bc_type: str = "dirichlet",
    ) -> None:
        super().__init__()
        self.x_res = x_res
        self.alpha = float(alpha)
        self.mu = float(mu)
        self.sources = list(sources)
        self.d_min = d_min
        self.n = int(x_res.numel())
        self.bc_type = bc_type.strip().lower()
        if self.n < 3:
            raise ValueError("Need n_res >= 3.")
        if self.bc_type not in {"dirichlet", "neumann"}:
            raise ValueError(f"Unsupported bc_type '{bc_type}'.")

        # h is a vector of interval lengths
        self.h = x_res[1:] - x_res[:-1]

        # Direct parameterization: theta = D - D_min
        if self.bc_type == "dirichlet":
            self.theta_int = nn.Parameter(d_init[1:-1] - d_min)
        else:
            self.theta_int = nn.Parameter(d_init - d_min)

        # Build RHS for unit source
        with torch.no_grad():
            if self.bc_type == "dirichlet":
                rhs = torch.zeros(self.n - 2, device=self.x_res.device, dtype=self.x_res.dtype)
                vol = 0.5 * (self.h[:-1] + self.h[1:])

                for z in self.sources:
                    z_t = torch.tensor(float(z), device=self.x_res.device, dtype=self.x_res.dtype)
                    idx_right = int(torch.searchsorted(self.x_res, z_t, side="right").item())
                    idx_left = idx_right - 1

                    if idx_left < 0:
                        idx_left = 0
                    if idx_right >= self.n:
                        idx_right = self.n - 1

                    x_left = self.x_res[idx_left]
                    x_right = self.x_res[idx_right]
                    h_interval = x_right - x_left

                    if h_interval < 1e-12:
                        if 1 <= idx_left <= self.n - 2:
                            rhs[idx_left - 1] -= 1.0 / vol[idx_left - 1]
                    else:
                        w_left = 1.0 - torch.abs(x_left - z_t) / h_interval
                        w_right = 1.0 - torch.abs(x_right - z_t) / h_interval

                        if 1 <= idx_left <= self.n - 2:
                            rhs[idx_left - 1] -= w_left / vol[idx_left - 1]
                        if 1 <= idx_right <= self.n - 2:
                            rhs[idx_right - 1] -= w_right / vol[idx_right - 1]
            else:
                rhs = torch.zeros(self.n, device=self.x_res.device, dtype=self.x_res.dtype)
                vol = torch.empty(self.n, device=self.x_res.device, dtype=self.x_res.dtype)
                vol[0] = self.h[0] / 2.0
                vol[-1] = self.h[-1] / 2.0
                if self.n > 2:
                    vol[1:-1] = 0.5 * (self.h[:-1] + self.h[1:])

                for z in self.sources:
                    z_t = torch.tensor(float(z), device=self.x_res.device, dtype=self.x_res.dtype)
                    idx_right = int(torch.searchsorted(self.x_res, z_t, side="right").item())
                    idx_left = idx_right - 1

                    if idx_left < 0:
                        idx_left = 0
                    if idx_right >= self.n:
                        idx_right = self.n - 1

                    x_left = self.x_res[idx_left]
                    x_right = self.x_res[idx_right]
                    h_interval = x_right - x_left

                    if h_interval < 1e-12:
                        rhs[idx_left] -= 1.0 / vol[idx_left]
                    else:
                        w_left = 1.0 - torch.abs(x_left - z_t) / h_interval
                        w_right = 1.0 - torch.abs(x_right - z_t) / h_interval
                        rhs[idx_left] -= w_left / vol[idx_left]
                        rhs[idx_right] -= w_right / vol[idx_right]

        self.register_buffer("rhs_unit", rhs)

    def build_d_full(self) -> torch.Tensor:
        """Build full D array from interior parameters."""
        if self.bc_type == "dirichlet":
            d = torch.empty(self.n, device=self.x_res.device, dtype=self.x_res.dtype)
            d[1:-1] = self.theta_int + self.d_min
            d[0] = d[1]
            d[-1] = d[-2]
            return d
        return self.theta_int + self.d_min

    def solve_unit_u_hat(self, d_full: torch.Tensor) -> torch.Tensor:
        """Solve for the unit-source response u_hat given D."""
        if self.bc_type == "dirichlet":
            lower, diag, upper = _build_tridiag_alpha(d_full, self.alpha, self.mu, self.h)
            u_int = _thomas_solve(lower, diag, upper, self.rhs_unit)

            u = torch.zeros(self.n, device=self.x_res.device, dtype=d_full.dtype)
            u[1:-1] = u_int
            return u

        lower, diag, upper = _build_tridiag_alpha_neumann(d_full, self.alpha, self.mu, self.h)
        return _thomas_solve(lower, diag, upper, self.rhs_unit)


def fit(data_bundle: DTOData, cfg: Config, verbose: bool = True) -> DTOResult:
    """Fit D with DTO and return the reconstructed fields.

    Args:
        data_bundle: DTOData specifying observations and grids.
        cfg: Full configuration (physics/data/grid/train/reg/arch/run).
        verbose: Print progress during training.

    Returns:
        DTOResult with predictions on the solver grid (x_res) and training history.
    """
    # =========================================================================
    # CONFIG EXTRACTION
    # =========================================================================
    # Extract all config values upfront for clarity. This makes dependencies
    # explicit and reduces visual clutter in the core algorithm.
    device = cfg.run.torch_device
    dtype = cfg.run.torch_dtype

    # Physics
    alpha = cfg.physics.alpha
    mu = cfg.physics.mu
    sources = cfg.physics.sources
    bc_type = cfg.physics.bc_type

    # Data
    mode = data_bundle.mode
    field_loss_type = cfg.data.field_loss
    b0_fixed_value = cfg.data.b0_fixed_value

    # D profile initialization
    use_ddi = cfg.d_profile.use_ddi
    ddi_d_min, ddi_d_max = cfg.d_profile.ddi_d_min, cfg.d_profile.ddi_d_max
    pert_scale, pert_freq = cfg.d_profile.pert_scale, cfg.d_profile.pert_freq

    # Regularization
    wreg_smooth, wreg_scale = cfg.reg.wreg_smooth, cfg.reg.wreg_scale
    smoothness_type = cfg.reg.smoothness_type

    # Training
    finetune_iters = cfg.train.finetune_iters
    lr_d_fine = cfg.train.lr_d_fine
    use_lbfgs = cfg.train.optimizer == "lbfgs"
    lbfgs_lr, lbfgs_max_iter = cfg.train.lbfgs_lr, cfg.train.lbfgs_max_iter
    use_scheduler = cfg.train.use_scheduler
    log_every = cfg.train.log_every
    scalar_fit_iters = cfg.train.scalar_fit_iters

    # Early stopping
    early_burnin = cfg.train.early_burnin
    early_patience = cfg.train.early_patience
    early_tol = cfg.train.early_tol

    # Architecture
    d_min = getattr(cfg.arch, "d_min", D_MIN)

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================
    if len(sources) != 1:
        raise NotImplementedError("DTO currently supports a single source.")

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================
    x_res = data_bundle.x_res.to(device=device, dtype=dtype)
    if x_res.ndim != 1:
        x_res = x_res.view(-1)
    x_field = x_res
    if data_bundle.x_field is not None:
        x_field = data_bundle.x_field.to(device=device, dtype=dtype)
        if x_field.ndim != 1:
            x_field = x_field.view(-1)

    if mode == "field":
        u_true = data_bundle.u_true.to(device=device, dtype=dtype).view(-1)
        ppp = None
        interp_particles = None
    else:
        ppp = PPPData(
            x_particles=data_bundle.ppp.x_particles.to(device=device, dtype=dtype).view(-1),
            m_obs=data_bundle.ppp.m_obs,
        )
        u_true = None
        interp_particles = varpro.precompute_interp_1d(x_res, ppp.x_particles)

    # =========================================================================
    # SCALE ESTIMATION
    # =========================================================================
    if use_ddi:
        d_ddi = estimate_ddi_scale(
            mu=mu,
            z=sources[0],
            x_particles=ppp.x_particles if ppp is not None else None,
            u_field=u_true if u_true is not None else None,
            x_grid=x_field,
            d_min=ddi_d_min,
            d_max=ddi_d_max,
        )
    else:
        d_ddi = float(cfg.d_profile.params[0])

    if scalar_fit_iters > 0:
        d_scale = fit_constant_d(
            x=x_res,
            alpha=alpha,
            mu=mu,
            sources=sources,
            u_true=u_true if u_true is not None else None,
            ppp=ppp if ppp is not None else None,
            x_field=x_field if u_true is not None else None,
            d_init=d_ddi,
            max_iters=scalar_fit_iters,
            field_loss=field_loss_type,
            bc_type=bc_type,
            verbose=verbose,
        )
    else:
        d_scale = d_ddi

    if verbose:
        print(f"[DTO] DDI scale: {d_ddi:.3e}")
        print(f"[DTO] Scalar fit scale: {d_scale:.3e}")

    d_target = d_scale

    # =========================================================================
    # MODEL INITIALIZATION
    # =========================================================================
    if pert_scale > 0.0:
        d_init = _init_d_profile(x_res, base=d_scale, scale=pert_scale, freq=pert_freq)
    else:
        d_init = d_scale * torch.ones_like(x_res)

    model = DtoAlphaVarPro(
        x_res=x_res,
        alpha=alpha,
        mu=mu,
        sources=list(sources),
        d_init=d_init,
        d_min=d_min,
        bc_type=bc_type,
    ).to(device=device, dtype=dtype)

    # =========================================================================
    # OPTIMIZER SETUP
    # =========================================================================
    if use_lbfgs:
        optimizer = torch.optim.LBFGS(
            [model.theta_int],
            lr=lbfgs_lr,
            max_iter=lbfgs_max_iter,
            history_size=10,
            line_search_fn="strong_wolfe",
        )
        scheduler = None
    else:
        optimizer = torch.optim.Adam([model.theta_int], lr=lr_d_fine)
        scheduler = None
        if use_scheduler and finetune_iters > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=finetune_iters, eta_min=lr_d_fine * 0.1
            )

    # =========================================================================
    # TRAINING HISTORY
    # =========================================================================
    history = TrainingHistory.for_dto()

    # Early stopping state
    best_total: Optional[float] = None
    patience = 0

    # =========================================================================
    # LOSS COMPUTATION
    # =========================================================================
    def _compute_losses() -> Tuple[torch.Tensor, ...]:
        d_full = model.build_d_full()
        u_hat_unit = model.solve_unit_u_hat(d_full)

        if mode == "field":
            if x_field.numel() == x_res.numel() and torch.allclose(x_field, x_res):
                u_hat_field = u_hat_unit
            else:
                u_hat_field = varpro.interpolate_1d(u_hat_unit, x_res, x_field)
            b0_star = varpro.get_b0_field(
                u_hat_field, u_true, field_loss=field_loss_type, b0_fixed_value=b0_fixed_value
            )
            data_loss = varpro.field_data_loss(
                u_hat_field, u_true, b0_star, field_loss=field_loss_type
            )
        else:
            integral_unit = torch.trapezoid(u_hat_unit.view(-1), x_res.view(-1))
            b0_star = varpro.get_b0_ppp(
                ppp.n_obs, ppp.m_obs, integral_unit, b0_fixed_value=b0_fixed_value
            )
            u_hat_obs = varpro.interpolate_1d_precomputed(u_hat_unit, *interp_particles)
            data_loss = varpro.ppp_nll(u_hat_obs, b0_star, ppp.m_obs, integral_unit)

        if smoothness_type == "tv":
            reg_smooth = physics.tv_smoothness_d_discrete(d_full, h=float(model.h[0].item()))
        else:
            reg_smooth = physics.h1_smoothness_d_discrete(d_full, h=float(model.h[0].item()))

        reg_scale = physics.scale_anchor(d_full, d_target)
        total_loss = data_loss + wreg_smooth * reg_smooth + wreg_scale * reg_scale

        return d_full, u_hat_unit, b0_star, data_loss, reg_smooth, reg_scale, total_loss

    # =========================================================================
    # MAIN TRAINING LOOP
    # =========================================================================
    try:
        for step in range(finetune_iters + 1):
            if not use_lbfgs:
                optimizer.zero_grad(set_to_none=True)

            # Forward pass
            d_full, u_hat_unit, b0_star, data_loss, reg_smooth, reg_scale, total_loss = (
                _compute_losses()
            )

            # -----------------------------------------------------------------
            # LOGGING
            # -----------------------------------------------------------------
            if step % log_every == 0:
                with torch.no_grad():
                    mean_d = torch.mean(d_full).item()
                    integral_unit = torch.trapezoid(u_hat_unit.view(-1), x_res.view(-1))
                    d_snapshot = d_full.detach().cpu().numpy()

                history.log(
                    step=step,
                    total=total_loss.item(),
                    data=data_loss.item(),
                    reg_smooth=reg_smooth.item(),
                    reg_scale=reg_scale.item(),
                    b0_star=b0_star.item(),
                    mean_d=mean_d,
                )
                history.log_snapshot(step, d_snapshot)

                if verbose:
                    loss_name = field_loss_type if mode == "field" else "ppp"
                    print(format_dto_progress(
                        step=step,
                        total=total_loss.item(),
                        data=data_loss.item(),
                        reg_smooth=reg_smooth.item(),
                        reg_scale=reg_scale.item(),
                        wreg_smooth=wreg_smooth,
                        wreg_scale=wreg_scale,
                        b0_star=b0_star.item(),
                        integral_unit=integral_unit.item(),
                        mean_d=mean_d,
                        loss_name=loss_name,
                    ))

            # -----------------------------------------------------------------
            # EARLY STOPPING
            # -----------------------------------------------------------------
            stop_training = False
            if step >= early_burnin:
                total_val = total_loss.item()
                if best_total is None:
                    best_total = total_val
                    patience = 0
                else:
                    denom = max(abs(best_total), 1e-12)
                    improvement = (best_total - total_val) / denom
                    if improvement > early_tol:
                        best_total = total_val
                        patience = 0
                    else:
                        patience += 1
                        if patience >= early_patience:
                            stop_training = True

            # -----------------------------------------------------------------
            # OPTIMIZER STEP
            # -----------------------------------------------------------------
            if step < finetune_iters and not stop_training:
                if use_lbfgs:
                    closure_calls = 0

                    def _lbfgs_closure() -> torch.Tensor:
                        nonlocal closure_calls
                        closure_calls += 1
                        optimizer.zero_grad(set_to_none=True)
                        loss_value = _compute_losses()[-1]
                        loss_value.backward()
                        return loss_value.detach()

                    theta_before = model.theta_int.data.clone()
                    optimizer.step(_lbfgs_closure)

                    # Project to enforce D >= D_min
                    with torch.no_grad():
                        model.theta_int.data.clamp_(min=0)

                    if step % log_every == 0 and verbose:
                        theta_after = model.theta_int.data
                        param_change = (theta_after - theta_before).abs().max().item()
                        grad_norm = model.theta_int.grad.norm().item() if model.theta_int.grad is not None else 0.0
                        print(f"  [LBFGS DEBUG] grad_norm: {grad_norm:.3e} | param_change: {param_change:.3e} | closure_calls: {closure_calls}")
                else:
                    total_loss.backward()
                    optimizer.step()

                    # Project to enforce D >= D_min
                    with torch.no_grad():
                        model.theta_int.data.clamp_(min=0)

                    if scheduler is not None:
                        scheduler.step()
            else:
                if stop_training and verbose:
                    print(f"[DTO] Early stopping triggered at step {step}.")
                break
    except KeyboardInterrupt:
        if verbose:
            print(f"\n[DTO] Training interrupted by user at step {step}. Continuing to post-processing...")

    # =========================================================================
    # FINAL RESULT EXTRACTION
    # =========================================================================
    with torch.no_grad():
        d_full = model.build_d_full()
        u_hat_unit = model.solve_unit_u_hat(d_full)
        if mode == "field":
            if x_field.numel() == x_res.numel() and torch.allclose(x_field, x_res):
                u_hat_field = u_hat_unit
            else:
                u_hat_field = varpro.interpolate_1d(u_hat_unit, x_res, x_field)
            b0_star = varpro.get_b0_field(
                u_hat_field, u_true, field_loss=field_loss_type, b0_fixed_value=b0_fixed_value
            )
        else:
            integral_unit = torch.trapezoid(u_hat_unit.view(-1), x_res.view(-1))
            b0_star = varpro.get_b0_ppp(
                ppp.n_obs, ppp.m_obs, integral_unit, b0_fixed_value=b0_fixed_value
            )
        u_pred = b0_star * u_hat_unit
        d_pred = d_full

    return DTOResult(
        x_res=x_res.detach().cpu(),
        d_pred=d_pred.detach().cpu(),
        u_hat_unit=u_hat_unit.detach().cpu(),
        u_pred=u_pred.detach().cpu(),
        b0_star=float(b0_star.item()),
        history=history.to_dict(),
    )
