"""PINN solver for the 1D alpha-PDE inverse problem.

Defines PINNData/PINNResult, network modules for logD and u, and the
single-level training loop that couples data and physics losses.
"""

# NOTE: This file intentionally duplicates some utilities (LogD init helpers)
# from other method files. This keeps each method self-contained and readable.
# If you modify shared logic, update all three method files.
#
# PINN overview:
# - Unique: joint training of logD_net and u_net with PDE residual loss.
# - Loss: w_data * data + w_phys * physics residual/jump + smoothness/scale anchor.
# - Optimization: single-level gradient descent on both networks.
# - Key knobs: w_data, w_phys, w_jump, wreg_smooth, wreg_scale, smoothness_type, lr_d/lr_lower.

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


@dataclass
class PINNData:
    """Input bundle for PINN training.

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
class PINNResult:
    """Outputs and training history for PINN fitting.

    Returned tensors are detached and on CPU for convenience.

    Fields:
        x_res: Solver grid (1D).
        logd: log D(x) on x_res (1D).
        d_pred: D(x) = exp(logd) on x_res (1D).
        u_hat_unit: Unit-source response u_hat for b0=1 on x_res (1D).
        u_pred: Predicted field b0* * u_hat_unit on x_res (1D).
        b0_star: Projected source amplitude.
        history: Loss curves and diagnostics recorded every log_every steps.
            Typical keys: iter, total, data, phys, res, jump, reg_smooth,
            reg_scale, b0_star, mean_d, d_snap_iters, d_snapshots.
    """

    x_res: torch.Tensor
    logd: torch.Tensor
    d_pred: torch.Tensor
    u_hat_unit: torch.Tensor
    u_pred: torch.Tensor
    b0_star: float
    history: Dict[str, List[float]]


class LogDNet(nn.Module):
    """RFF-embedded MLP that parameterizes log D(x)."""

    def __init__(self, width: int = 128, use_rff: bool = True) -> None:
        super().__init__()
        self.use_rff = use_rff
        self.embed = nn.Linear(1, width)
        if use_rff:
            for param in self.embed.parameters():
                param.requires_grad = False
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate log D(x) at given coordinates."""
        x = x.view(-1, 1)
        if self.use_rff:
            feat = torch.sin(2.0 * torch.pi * self.embed(x))
        else:
            feat = self.embed(x)
        return self.net(feat)


class LocalOperator(nn.Module):
    """Local operator network for the unit response u_hat(x)."""

    def __init__(self, width: int = 128, use_rff: bool = True) -> None:
        super().__init__()
        # NOTE: Unlike BiLO, the PINN u-network does NOT condition on logD.
        # This is standard PINN practice: u_net and logD_net are trained jointly.
        self.use_rff = use_rff
        self.geom_layer = nn.Linear(2, width)
        if use_rff:
            self.geom_layer.weight.requires_grad = False
            if self.geom_layer.bias is not None:
                self.geom_layer.bias.requires_grad = False
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(3)])
        self.output = nn.Linear(width, 1)
        self.activation = torch.tanh

    def forward(self, x: torch.Tensor, z_known: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate u_hat and return the |x-z| feature for jump constraints."""
        x = x.view(-1, 1)
        phi_z = torch.abs(x - z_known)
        if phi_z.is_leaf and not phi_z.requires_grad:
            phi_z.requires_grad_(True)
        geom_in = torch.cat([x, phi_z], dim=1)
        geom_lin = self.geom_layer(geom_in)
        if self.use_rff:
            h = self.activation(torch.sin(2.0 * torch.pi * geom_lin))
        else:
            h = self.activation(geom_lin)
        for layer in self.hidden:
            h = self.activation(layer(h))
        u_raw = self.output(h)
        u = F.softplus(u_raw) * x * (1.0 - x)
        return u, phi_z


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


def _alpha_flux_residual(
    x: torch.Tensor,
    logd: torch.Tensor,
    u: torch.Tensor,
    alpha: float,
    mu: float,
) -> torch.Tensor:
    """Flux-form PDE residual L_alpha(u) - mu*u using autograd."""
    if not x.requires_grad:
        raise ValueError("x must have requires_grad=True for autograd residuals.")
    ones = torch.ones_like(u)
    q = torch.exp((1.0 - alpha) * logd) * u
    q_x = torch.autograd.grad(q, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    J = torch.exp(alpha * logd) * q_x
    J_x = torch.autograd.grad(J, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    return J_x - mu * u


def _compute_physics_losses(
    logd_net: nn.Module,
    u_net: nn.Module,
    x_res: torch.Tensor,
    z_tensor: torch.Tensor,
    alpha: float,
    mu: float,
    w_jump: float,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute PDE residual and jump losses for the PINN objective.

    Args:
        mask: Precomputed boolean mask that excludes source-adjacent points.
    """
    x_pde = x_res.clone().detach().requires_grad_(True)
    logd = logd_net(x_pde)
    u_hat, _ = u_net(x_pde, z_tensor)
    residual = _alpha_flux_residual(x_pde, logd, u_hat, alpha, mu)
    res_loss = torch.mean(residual[mask] ** 2)

    z_probe = z_tensor.clone().detach().requires_grad_(True)
    logd_z = logd_net(z_probe)
    u_z, phi_z = u_net(z_probe, z_tensor)
    du_dphi = torch.autograd.grad(
        u_z,
        phi_z,
        grad_outputs=torch.ones_like(u_z),
        create_graph=True,
    )[0]
    jump_res = torch.exp(logd_z) * (2.0 * du_dphi) + 1.0
    jump_loss = torch.mean(jump_res ** 2)

    phys_loss = res_loss + w_jump * jump_loss
    return phys_loss, res_loss, jump_loss


def fit(data_bundle: PINNData, cfg: Config, verbose: bool = True) -> PINNResult:
    """Fit log-D with a physics-informed neural network.

    Args:
        data_bundle: PINNData specifying observations and grids.
        cfg: Full configuration (physics/data/grid/train/reg/arch/run).
        verbose: Print progress during training.

    Returns:
        PINNResult with predictions on the solver grid (x_res) and training history.
    """
    device = cfg.run.torch_device
    dtype = cfg.run.torch_dtype
    if len(cfg.physics.sources) != 1:
        raise NotImplementedError("PINN currently supports a single source.")

    x_res = data_bundle.x_res.to(device=device, dtype=dtype).view(-1, 1)
    x_field = x_res
    if data_bundle.x_field is not None:
        x_field = data_bundle.x_field.to(device=device, dtype=dtype).view(-1, 1)
    if data_bundle.mode == "field":
        u_true = data_bundle.u_true.to(device=device, dtype=dtype).view(-1, 1)
        ppp = None
    else:
        ppp = PPPData(
            x_particles=data_bundle.ppp.x_particles.to(device=device, dtype=dtype).view(-1, 1),
            m_obs=data_bundle.ppp.m_obs,
        )
        u_true = None

    z_tensor = torch.tensor(cfg.physics.sources[0], device=device, dtype=dtype).view(1, 1)
    # Mask out source-adjacent points once to avoid per-iteration recomputation.
    pde_mask = torch.abs(x_res - z_tensor) > 1e-4

    d_init_base = cfg.d_profile.d_init_base
    if cfg.d_profile.use_ddi:
        d_init_base = estimate_ddi_scale(
            mu=cfg.physics.mu,
            z=cfg.physics.sources[0],
            x_particles=ppp.x_particles if ppp is not None else None,
            u_field=u_true if u_true is not None else None,
            x_grid=x_field.view(-1),
            d_min=cfg.d_profile.ddi_d_min,
            d_max=cfg.d_profile.ddi_d_max,
        )

    if verbose:
        print(f"[PINN] Initialized ⟨D⟩_base: {d_init_base:.3e}")
    log_target = math.log(d_init_base)

    logd_init = _init_logd_profile(
        x_res.view(-1),
        base=d_init_base,
        scale=cfg.d_profile.d_init_pert_scale,
        freq=cfg.d_profile.d_init_pert_freq,
    ).view(-1, 1)

    logd_net = LogDNet(width=cfg.arch.rff_width, use_rff=cfg.arch.use_rff_logd).to(device)
    u_net = LocalOperator(width=cfg.arch.rff_width, use_rff=cfg.arch.use_rff_geom).to(device)

    history: Dict[str, List[float]] = {
        "iter": [],
        "total": [],
        "data": [],
        "phys": [],
        "res": [],
        "jump": [],
        "reg_smooth": [],
        "reg_scale": [],
        "b0_star": [],
        "mean_d": [],
        "d_snap_iters": [],
        "d_snapshots": [],
    }

    if cfg.train.pretrain_iters > 0:
        optim_pre = torch.optim.Adam(
            [
                {"params": logd_net.parameters(), "lr": cfg.train.lr_d_pre},
                {"params": u_net.parameters(), "lr": cfg.train.lr_lower_pre},
            ]
        )
        for step in range(cfg.train.pretrain_iters + 1):
            optim_pre.zero_grad(set_to_none=True)
            phys_loss, res_loss, jump_loss = _compute_physics_losses(
                logd_net,
                u_net,
                x_res,
                z_tensor,
                cfg.physics.alpha,
                cfg.physics.mu,
                cfg.reg.w_jump,
                pde_mask,
            )
            logd_pred = logd_net(x_res)
            anchor_loss = physics.log_scale_anchor(logd_pred, log_target)
            pre_loss = cfg.reg.w_phys * phys_loss + anchor_loss

            if verbose and step % cfg.train.log_every == 0:
                with torch.no_grad():
                    mean_d = torch.mean(torch.exp(logd_pred)).item()
                print(
                    f"[PINN|pretrain] Iter {step:05d} | Ltot: {pre_loss.item():.3e}\n"
                    f"  Lphys: {phys_loss.item():.3e} | Lanchor: {anchor_loss.item():.3e}\n"
                    f"  Lres: {res_loss.item():.3e} | Ljump: {jump_loss.item():.3e}\n"
                    f"  ⟨D⟩: {mean_d:.3e}"
                )

            if step < cfg.train.pretrain_iters:
                pre_loss.backward()
                optim_pre.step()

    optim_fine = torch.optim.Adam(
        [
            {"params": logd_net.parameters(), "lr": cfg.train.lr_d_fine},
            {"params": u_net.parameters(), "lr": cfg.train.lr_lower_fine},
        ]
    )
    sched = None
    if cfg.train.use_scheduler and cfg.train.finetune_iters > 0:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim_fine, T_max=cfg.train.finetune_iters, eta_min=cfg.train.lr_d_fine * 0.1
        )

    x_int = torch.linspace(
        cfg.physics.domain[0], cfg.physics.domain[1], cfg.grid.n_int, device=device, dtype=dtype
    ).view(-1, 1)

    best_total: Optional[float] = None
    patience = 0

    for step in range(cfg.train.finetune_iters + 1):
        optim_fine.zero_grad(set_to_none=True)

        integral_unit = None
        if data_bundle.mode == "field":
            u_hat_field, _ = u_net(x_field, z_tensor)
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
            integral_unit = None
        else:
            u_hat_int, _ = u_net(x_int, z_tensor)
            integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
            b0_star = varpro.project_b0_ppp(ppp.n_obs, ppp.m_obs, integral_unit)
            u_hat_obs, _ = u_net(ppp.x_particles, z_tensor)
            data_loss = varpro.ppp_nll(
                u_hat_obs.view(-1),
                b0_star,
                ppp.m_obs,
                integral_unit,
            )

        phys_loss, res_loss, jump_loss = _compute_physics_losses(
            logd_net,
            u_net,
            x_res,
            z_tensor,
            cfg.physics.alpha,
            cfg.physics.mu,
            cfg.reg.w_jump,
            pde_mask,
        )

        x_reg = x_res.clone().detach().requires_grad_(True)
        logd_reg = logd_net(x_reg)
        if cfg.reg.smoothness_type == "tv":
            reg_smooth = physics.tv_smoothness_logd(x_reg, logd_reg)
        else:
            reg_smooth = physics.h1_smoothness_logd(x_reg, logd_reg)
        # Reuse logd_reg (same x locations) to avoid an extra forward pass.
        reg_scale = physics.log_scale_anchor(logd_reg, log_target)

        total_loss = (
            cfg.reg.w_data * data_loss
            + cfg.reg.w_phys * phys_loss
            + cfg.reg.wreg_smooth * reg_smooth
            + cfg.reg.wreg_scale * reg_scale
        )

        if step % cfg.train.log_every == 0:
            with torch.no_grad():
                d_vals = torch.exp(logd_reg.detach())
                mean_d = torch.mean(d_vals).item()
                d_snapshot = d_vals.detach().cpu().numpy().reshape(-1)
                if data_bundle.mode == "field":
                    # Integral is only needed for logging in field mode.
                    u_hat_res, _ = u_net(x_res, z_tensor)
                    integral_unit = torch.trapezoid(u_hat_res.view(-1), x_res.view(-1))
                u_int = (b0_star * integral_unit).item() if integral_unit is not None else 0.0
            history["iter"].append(step)
            history["total"].append(total_loss.item())
            history["data"].append(data_loss.item())
            history["phys"].append(phys_loss.item())
            history["res"].append(res_loss.item())
            history["jump"].append(jump_loss.item())
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
                    f"[PINN|finetune] Iter {step:05d} | Ltot: {total_loss.item():.3e}\n"
                    f"  Ldata({loss_name}): {data_loss.item():.3e} | Lphys: {phys_loss.item():.3e} | "
                    f"RegSmooth: {reg_smooth.item():.3e} (eff: {reg_smooth_eff.item():.3e}) | "
                    f"RegScale: {reg_scale.item():.3e} (eff: {reg_scale_eff.item():.3e})\n"
                    f"  Lres: {res_loss.item():.3e} | Ljump: {jump_loss.item():.3e}\n"
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
            optim_fine.step()
            if sched is not None:
                sched.step()
        else:
            if stop_training and verbose:
                 print(f"[PINN] Early stopping triggered at step {step}.")
            break

    with torch.no_grad():
        logd_final = logd_net(x_res)
        u_hat_res, _ = u_net(x_res, z_tensor)
        if data_bundle.mode == "field":
            u_hat_field, _ = u_net(x_field, z_tensor)
            b0_star = varpro.project_b0_field(
                u_hat_field,
                u_true,
                field_loss=cfg.data.field_loss,
            )
        else:
            u_hat_int, _ = u_net(x_int, z_tensor)
            integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
            b0_star = varpro.project_b0_ppp(ppp.n_obs, ppp.m_obs, integral_unit)
        u_pred = b0_star * u_hat_res
        d_pred = torch.exp(logd_final)

    return PINNResult(
        x_res=x_res.detach().cpu().view(-1),
        logd=logd_final.detach().cpu().view(-1),
        d_pred=d_pred.detach().cpu().view(-1),
        u_hat_unit=u_hat_res.detach().cpu().view(-1),
        u_pred=u_pred.detach().cpu().view(-1),
        b0_star=float(b0_star.item()),
        history=history,
    )
