"""DTO (direct-to-optimization) solver for the 1D alpha-PDE inverse problem.

Defines DTOData/DTOResult, a differentiable tridiagonal solver, and the
single-level optimization loop for grid-parameterized D.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from data import PPPData, estimate_ddi_scale
import physics, varpro


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
        raise ValueError("d_init_pert_scale must be < 1 to keep D_init positive.")
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble tridiagonal coefficients for the alpha-PDE discretization.
    
    h is now a vector of step sizes: h[i] = x[i+1] - x[i].
    """
    g = d ** (1.0 - alpha)
    d_half = 0.5 * (d[:-1] + d[1:])
    a_half = (d_half ** alpha) / h

    # Interior volumes: vol[i] = (h[i] + h[i+1]) / 2 (indices shifted)
    # The 'h' vector has length N-1.
    # a_half has length N-1.
    # We need volumes for interior nodes 1..N-2.
    # vol[k] corresponds to node k+1.
    vol = 0.5 * (h[:-1] + h[1:])

    diag = -((a_half[1:] + a_half[:-1]) * g[1:-1] / vol) - mu
    lower_full = (a_half[:-1] * g[:-2]) / vol
    upper_full = (a_half[1:] * g[2:]) / vol
    lower = lower_full[1:]
    upper = upper_full[:-1]
    return lower, diag, upper


class DtoAlphaVarPro(nn.Module):
    """Finite-difference solver with a learnable interior D profile."""

    def __init__(
        self,
        x_res: torch.Tensor,
        alpha: float,
        mu: float,
        sources: List[float],
        d_init: torch.Tensor,
        d_min: float = D_MIN,
    ) -> None:
        super().__init__()
        self.x_res = x_res
        self.alpha = float(alpha)
        self.mu = float(mu)
        self.sources = list(sources)
        self.d_min = d_min
        self.n = int(x_res.numel())
        if self.n < 3:
            raise ValueError("Need n_res >= 3.")
        
        # h is now a vector of interval lengths
        self.h = x_res[1:] - x_res[:-1]

        d_init_shifted = d_init[1:-1] - d_min
        # Inverse softplus: x = log(exp(y) - 1)
        self.theta_int = nn.Parameter(
            torch.where(
                d_init_shifted > 20,
                d_init_shifted,
                torch.log(torch.expm1(d_init_shifted.clamp(min=1e-8)))
            )
        )

        with torch.no_grad():
            rhs = torch.zeros(self.n - 2, device=self.x_res.device, dtype=self.x_res.dtype)
            # vol is needed for RHS scaling
            vol = 0.5 * (self.h[:-1] + self.h[1:])
            
            for z in self.sources:
                z_t = torch.tensor(float(z), device=self.x_res.device, dtype=self.x_res.dtype)
                # p2h-s1 hat delta: distribute source to the two nearest grid points
                idx_right = int(torch.searchsorted(self.x_res, z_t, side="right").item())
                idx_left = idx_right - 1
                
                # Check bounds
                if idx_left < 0: idx_left = 0
                if idx_right >= self.n: idx_right = self.n - 1
                
                # We need to map global indices to interior RHS indices (0..N-3).
                # The source contribution is -b0 * w / vol, effectively smearing the
                # Dirac delta into a hat function with unit integral over the
                # Voronoi cells.
                
                x_left = self.x_res[idx_left]
                x_right = self.x_res[idx_right]
                h_interval = x_right - x_left
                
                if h_interval < 1e-12:
                     # Coincides with node?
                     # If idx_left is an interior node (1..N-2)
                     if 1 <= idx_left <= self.n - 2:
                         rhs[idx_left - 1] -= 1.0 / vol[idx_left - 1]
                else:
                    w_left = 1.0 - torch.abs(x_left - z_t) / h_interval
                    w_right = 1.0 - torch.abs(x_right - z_t) / h_interval
                    
                    if 1 <= idx_left <= self.n - 2:
                        rhs[idx_left - 1] -= w_left / vol[idx_left - 1]
                    if 1 <= idx_right <= self.n - 2:
                        rhs[idx_right - 1] -= w_right / vol[idx_right - 1]
                        
        self.register_buffer("rhs_unit", rhs)

    def build_d_full(self) -> torch.Tensor:
        """Mirror interior D to the boundaries for Dirichlet conditions."""
        d = torch.empty(self.n, device=self.x_res.device, dtype=self.x_res.dtype)
        d[1:-1] = F.softplus(self.theta_int) + self.d_min
        d[0] = d[1]
        d[-1] = d[-2]
        return d

    def solve_unit_u_hat(self, d_full: torch.Tensor) -> torch.Tensor:
        """Solve for the unit-source response u_hat given D.

        Uses a cached unit-source RHS vector to avoid per-iteration rebuilds.
        """
        lower, diag, upper = _build_tridiag_alpha(d_full, self.alpha, self.mu, self.h)
        u_int = _thomas_solve(lower, diag, upper, self.rhs_unit)
        u = torch.zeros(self.n, device=self.x_res.device, dtype=self.x_res.dtype)
        u[1:-1] = u_int
        return u


def fit(data_bundle: DTOData, cfg: Config, verbose: bool = True) -> DTOResult:
    """Fit D with DTO and return the reconstructed fields.

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
    
    # We use d_init_base directly for regularization target
    d_target = d_init_base

    d_init = _init_d_profile(
        x_res,
        base=d_init_base,
        scale=cfg.d_profile.d_init_pert_scale,
        freq=cfg.d_profile.d_init_pert_freq,
    )

    d_min = getattr(cfg.arch, "d_min", D_MIN)
    model = DtoAlphaVarPro(
        x_res=x_res,
        alpha=cfg.physics.alpha,
        mu=cfg.physics.mu,
        sources=list(cfg.physics.sources),
        d_init=d_init,
        d_min=d_min,
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

        d_full = model.build_d_full()
        u_hat_unit = model.solve_unit_u_hat(d_full)

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
            reg_smooth = physics.tv_smoothness_d_discrete(
                d_full, h=float(model.h[0].item())
            )
        else:
            reg_smooth = physics.h1_smoothness_d_discrete(
                d_full, h=float(model.h[0].item())
            )
        reg_scale = physics.scale_anchor(d_full, d_target)
        total_loss = (
            data_loss
            + cfg.reg.wreg_smooth * reg_smooth
            + cfg.reg.wreg_scale * reg_scale
        )

        if step % cfg.train.log_every == 0:
            with torch.no_grad():
                mean_d = torch.mean(d_full).item()
                if data_bundle.mode == "particles":
                    u_hat_int = varpro.interpolate_1d_precomputed(u_hat_unit, *interp_int)
                    integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
                else:
                    integral_unit = torch.trapezoid(u_hat_unit.view(-1), x_res.view(-1))
                u_int = (b0_star * integral_unit).item()
                d_snapshot = d_full.detach().cpu().numpy()
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
                    f"  b₀*: {b0_star.item():.2f} | ∫û: {integral_unit.item():.3e} | ∫u: {u_int:.3e} | ⟨D⟩: {mean_d:.3e}"
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
        d_full = model.build_d_full()
        u_hat_unit = model.solve_unit_u_hat(d_full)
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
        d_pred = d_full

    return DTOResult(
        x_res=x_res.detach().cpu(),
        d_pred=d_pred.detach().cpu(),
        u_hat_unit=u_hat_unit.detach().cpu(),
        u_pred=u_pred.detach().cpu(),
        b0_star=float(b0_star.item()),
        history=history,
    )
