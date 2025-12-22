"""DTO (direct-to-optimization) solver for the 1D alpha-PDE inverse problem.

Defines DTOData/DTOResult, a differentiable tridiagonal solver, and the
single-level optimization loop for grid-parameterized logD.
"""

# NOTE: This file intentionally duplicates some utilities (LogD init helpers)
# from other method files. This keeps each method self-contained and readable.
# If you modify shared logic, update all three method files.
#
# DTO overview:
# - Unique: grid-parameterized logD with a differentiable tridiagonal solver.
# - Loss: data loss via VarPro + smoothness (H1/TV) + log-normal scale anchor.
# - Optimization: single-level gradient descent on interior logD parameters.
# - Key knobs: wreg_smooth, wreg_scale, smoothness_type, lr_d_fine.

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from config import Config
from data import PPPData, estimate_ddi_scale
import physics, varpro


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
        logd: log D(x) on x_res (1D).
        d_pred: D(x) = exp(logd) on x_res (1D).
        u_hat_unit: Unit-source response u_hat for b0=1 on x_res (1D).
        u_pred: Predicted field b0* * u_hat_unit on x_res (1D).
        b0_star: Projected source amplitude.
        history: Loss curves and diagnostics recorded every log_every steps.
            Typical keys: iter, total, data, reg_smooth, reg_scale, b0_star,
            mean_d, d_snap_iters, d_snapshots.
    """

    x_res: torch.Tensor
    logd: torch.Tensor
    d_pred: torch.Tensor
    u_hat_unit: torch.Tensor
    u_pred: torch.Tensor
    b0_star: float
    history: Dict[str, List[float]]


def _init_logd_profile(
    x: torch.Tensor,
    base: float,
    scale: float,
    freq: float,
) -> torch.Tensor:
    """Build a sinusoidal log-D initialization on the grid."""
    if scale >= 1.0:
        raise ValueError("d_init_pert_scale must be < 1 to keep D_init positive.")
    d_init = base * (1.0 + scale * torch.sin(2.0 * torch.pi * freq * x))
    return torch.log(d_init)


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
    logd: torch.Tensor,
    alpha: float,
    mu: float,
    h: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble tridiagonal coefficients for the alpha-PDE discretization."""
    g = torch.exp((1.0 - alpha) * logd)
    logd_half = 0.5 * (logd[:-1] + logd[1:])
    a_half = torch.exp(alpha * logd_half) / h

    diag = -((a_half[1:] + a_half[:-1]) * g[1:-1] / h) - mu
    lower_full = (a_half[:-1] * g[:-2]) / h
    upper_full = (a_half[1:] * g[2:]) / h
    lower = lower_full[1:]
    upper = upper_full[:-1]
    return lower, diag, upper


class DtoAlphaVarPro(nn.Module):
    """Finite-difference solver with a learnable interior log-D profile."""

    def __init__(
        self,
        x_res: torch.Tensor,
        alpha: float,
        mu: float,
        sources: List[float],
        logd_init: torch.Tensor,
    ) -> None:
        super().__init__()
        self.x_res = x_res
        self.alpha = float(alpha)
        self.mu = float(mu)
        self.sources = list(sources)
        self.n = int(x_res.numel())
        if self.n < 3:
            raise ValueError("Need n_res >= 3.")
        self.h = x_res[1] - x_res[0]
        self.theta_int = nn.Parameter(logd_init[1:-1].clone().detach())

        with torch.no_grad():
            self.z_indices = []
            for z in self.sources:
                idx = torch.argmin(torch.abs(self.x_res - float(z))).item()
                self.z_indices.append(int(idx))
            rhs = torch.zeros(self.n - 2, device=self.x_res.device, dtype=self.x_res.dtype)
            for idx in self.z_indices:
                idx = max(1, min(self.n - 2, idx))
                rhs[idx - 1] = rhs[idx - 1] - (1.0 / self.h)
        # Unit-source RHS is constant across iterations; cache as a buffer.
        self.register_buffer("rhs_unit", rhs)

    def build_logd_full(self) -> torch.Tensor:
        """Mirror interior log-D to the boundaries for Dirichlet conditions."""
        logd = torch.empty(self.n, device=self.x_res.device, dtype=self.x_res.dtype)
        logd[1:-1] = self.theta_int
        logd[0] = logd[1]
        logd[-1] = logd[-2]
        return logd

    def solve_unit_u_hat(self, logd_full: torch.Tensor) -> torch.Tensor:
        """Solve for the unit-source response u_hat given log-D.

        Uses a cached unit-source RHS vector to avoid per-iteration rebuilds.
        """
        lower, diag, upper = _build_tridiag_alpha(logd_full, self.alpha, self.mu, self.h)
        u_int = _thomas_solve(lower, diag, upper, self.rhs_unit)
        u = torch.zeros(self.n, device=self.x_res.device, dtype=self.x_res.dtype)
        u[1:-1] = u_int
        return u


def fit(data_bundle: DTOData, cfg: Config, verbose: bool = True) -> DTOResult:
    """Fit log-D with DTO and return the reconstructed fields.

    Args:
        data_bundle: DTOData specifying observations and grids.
        cfg: Full configuration (physics/data/grid/train/reg/arch/run).
        verbose: Print progress during training.

    Returns:
        DTOResult with predictions on the solver grid (x_res) and training history.
    """
    device = cfg.run.torch_device
    dtype = cfg.run.torch_dtype

    if len(cfg.physics.sources) != 1:
        raise NotImplementedError("DTO currently supports a single source.")

    x_res = data_bundle.x_res.to(device=device, dtype=dtype)
    if x_res.ndim != 1:
        x_res = x_res.view(-1)
    x_field = x_res
    if data_bundle.x_field is not None:
        x_field = data_bundle.x_field.to(device=device, dtype=dtype)
        if x_field.ndim != 1:
            x_field = x_field.view(-1)

    if data_bundle.mode == "field":
        u_true = data_bundle.u_true.to(device=device, dtype=dtype).view(-1)
        ppp = None
        x_int = None
    else:
        ppp = PPPData(
            x_particles=data_bundle.ppp.x_particles.to(device=device, dtype=dtype).view(-1),
            m_obs=data_bundle.ppp.m_obs,
        )
        u_true = None
        domain = cfg.physics.domain
        x_int = torch.linspace(domain[0], domain[1], cfg.grid.n_int, device=device, dtype=dtype)
        # Precompute interpolation weights for fixed PPP grids.
        interp_int = varpro.precompute_interp_1d(x_res, x_int)
        interp_particles = varpro.precompute_interp_1d(x_res, ppp.x_particles)

    d_init_base = cfg.d_profile.d_init_base
    if cfg.d_profile.use_ddi:
        z0 = cfg.physics.sources[0]
        d_init_base = estimate_ddi_scale(
            mu=cfg.physics.mu,
            z=z0,
            x_particles=ppp.x_particles if ppp is not None else None,
            u_field=u_true if u_true is not None else None,
            x_grid=x_field,
            d_min=cfg.d_profile.ddi_d_min,
            d_max=cfg.d_profile.ddi_d_max,
        )

    if verbose:
        print(f"[DTO] Initialized ⟨D⟩_base: {d_init_base:.3e}")
    log_target = math.log(d_init_base)

    logd_init = _init_logd_profile(
        x_res,
        base=d_init_base,
        scale=cfg.d_profile.d_init_pert_scale,
        freq=cfg.d_profile.d_init_pert_freq,
    )

    model = DtoAlphaVarPro(
        x_res=x_res,
        alpha=cfg.physics.alpha,
        mu=cfg.physics.mu,
        sources=list(cfg.physics.sources),
        logd_init=logd_init,
    ).to(device)

    optimizer = torch.optim.Adam([model.theta_int], lr=cfg.train.lr_d_fine)
    scheduler = None
    if cfg.train.use_scheduler and cfg.train.finetune_iters > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.train.finetune_iters, eta_min=cfg.train.lr_d_fine * 0.1
        )

    history: Dict[str, List[float]] = {
        "iter": [],
        "total": [],
        "data": [],
        "reg_smooth": [],
        "reg_scale": [],
        "b0_star": [],
        "mean_d": [],
        "d_snap_iters": [],
        "d_snapshots": [],
    }

    best_total: Optional[float] = None
    patience = 0

    for step in range(cfg.train.finetune_iters + 1):
        optimizer.zero_grad(set_to_none=True)

        logd_full = model.build_logd_full()
        u_hat_unit = model.solve_unit_u_hat(logd_full)

        if data_bundle.mode == "field":
            if x_field.numel() == x_res.numel() and torch.allclose(x_field, x_res):
                u_hat_field = u_hat_unit
            else:
                u_hat_field = varpro.interpolate_1d(u_hat_unit, x_res, x_field)
            b0_star = varpro.project_b0_field(
                u_hat_field,
                u_true,
                field_loss=cfg.data.field_loss,
            )
            data_loss = varpro.field_data_loss(
                u_hat_field,
                u_true,
                b0_star,
                field_loss=cfg.data.field_loss,
            )
        else:
            u_hat_int = varpro.interpolate_1d_precomputed(u_hat_unit, *interp_int)
            integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
            b0_star = varpro.project_b0_ppp(ppp.n_obs, ppp.m_obs, integral_unit)
            u_hat_obs = varpro.interpolate_1d_precomputed(u_hat_unit, *interp_particles)
            data_loss = varpro.ppp_nll(
                u_hat_obs,
                b0_star,
                ppp.m_obs,
                integral_unit,
            )

        if cfg.reg.smoothness_type == "tv":
            reg_smooth = physics.tv_smoothness_logd_discrete(
                logd_full, h=float(model.h.item())
            )
        else:
            reg_smooth = physics.h1_smoothness_logd_discrete(
                logd_full, h=float(model.h.item())
            )
        reg_scale = physics.log_scale_anchor(logd_full, log_target)
        total_loss = (
            data_loss
            + cfg.reg.wreg_smooth * reg_smooth
            + cfg.reg.wreg_scale * reg_scale
        )

        if step % cfg.train.log_every == 0:
            with torch.no_grad():
                mean_d = torch.mean(torch.exp(logd_full)).item()
                if data_bundle.mode == "particles":
                    u_hat_int = varpro.interpolate_1d_precomputed(u_hat_unit, *interp_int)
                    integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
                else:
                    integral_unit = torch.trapezoid(u_hat_unit.view(-1), x_res.view(-1))
                u_int = (b0_star * integral_unit).item()
                d_snapshot = torch.exp(logd_full).detach().cpu().numpy()
            history["iter"].append(step)
            history["total"].append(total_loss.item())
            history["data"].append(data_loss.item())
            history["reg_smooth"].append(reg_smooth.item())
            history["reg_scale"].append(reg_scale.item())
            history["b0_star"].append(b0_star.item())
            history["mean_d"].append(mean_d)
            history["d_snap_iters"].append(step)
            history["d_snapshots"].append(d_snapshot)
            if verbose:
                loss_name = cfg.data.field_loss if data_bundle.mode == "field" else "ppp"
                reg_smooth_eff = cfg.reg.wreg_smooth * reg_smooth
                reg_scale_eff = cfg.reg.wreg_scale * reg_scale
                print(
                    f"[DTO] Iter {step:05d} | Ltot: {total_loss.item():.3e}\n"
                    f"  Ldata({loss_name}): {data_loss.item():.3e} | "
                    f"RegSmooth: {reg_smooth.item():.3e} (eff: {reg_smooth_eff.item():.3e}) | "
                    f"RegScale: {reg_scale.item():.3e} (eff: {reg_scale_eff.item():.3e})\n"
                    f"  b₀*: {b0_star.item():.2f} | ∫u: {u_int:.3e} | ⟨D⟩: {mean_d:.3e}"
                )

        stop_training = False
        if step >= cfg.train.early_burnin:
            total_val = total_loss.item()
            if best_total is None:
                best_total = total_val
                patience = 0
            else:
                denom = max(abs(best_total), 1e-12)
                improvement = (best_total - total_val) / denom
                if improvement > cfg.train.early_tol:
                    best_total = total_val
                    patience = 0
                else:
                    patience += 1
                    if patience >= cfg.train.early_patience:
                        stop_training = True

        if step < cfg.train.finetune_iters and not stop_training:
            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            if stop_training and verbose:
                 print(f"[DTO] Early stopping triggered at step {step}.")
            break

    with torch.no_grad():
        logd_full = model.build_logd_full()
        u_hat_unit = model.solve_unit_u_hat(logd_full)
        if data_bundle.mode == "field":
            if x_field.numel() == x_res.numel() and torch.allclose(x_field, x_res):
                u_hat_field = u_hat_unit
            else:
                u_hat_field = varpro.interpolate_1d(u_hat_unit, x_res, x_field)
            b0_star = varpro.project_b0_field(
                u_hat_field,
                u_true,
                field_loss=cfg.data.field_loss,
            )
        else:
            u_hat_int = varpro.interpolate_1d(u_hat_unit, x_res, x_int)
            integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
            b0_star = varpro.project_b0_ppp(ppp.n_obs, ppp.m_obs, integral_unit)
        u_pred = b0_star * u_hat_unit
        d_pred = torch.exp(logd_full)

    return DTOResult(
        x_res=x_res.detach().cpu(),
        logd=logd_full.detach().cpu(),
        d_pred=d_pred.detach().cpu(),
        u_hat_unit=u_hat_unit.detach().cpu(),
        u_pred=u_pred.detach().cpu(),
        b0_star=float(b0_star.item()),
        history=history,
    )
