"""BiLO (bilevel optimization) solver for the 1D alpha-PDE inverse problem.

Defines BiLOData/BiLOResult, network modules for logD and the local operator,
plus bilevel pretrain/finetune training loops.
"""

# NOTE: This file intentionally duplicates some utilities (LogD init helpers)
# from other method files. This keeps each method self-contained and readable.
# If you modify shared logic, update all three method files.
#
# BiLO overview:
# - Unique: bilevel training with a learnable local operator for u_hat(x).
# - Loss: upper (data + smoothness/scale anchor) and lower (physics residual/jump/rgrad).
# - Optimization: simultaneous updates on disjoint parameter groups each iteration.
# - Key knobs: w_jump, w_resgrad, wreg_smooth, wreg_scale, smoothness_type.

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
class BiLOData:
    """Input bundle for BiLO training.

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
class BiLOResult:
    """Outputs and training history for BiLO fitting.

    Returned tensors are detached and on CPU for convenience.

    Fields:
        x_res: Solver grid (1D).
        logd: log D(x) on x_res (1D).
        d_pred: D(x) = exp(logd) on x_res (1D).
        u_hat_unit: Unit-source response u_hat for b0=1 on x_res (1D).
        u_pred: Predicted field b0* * u_hat_unit on x_res (1D).
        b0_star: Projected source amplitude.
        history: Loss curves and diagnostics recorded every log_every steps.
            Typical keys: iter, upper, data, reg_smooth, reg_scale, lower, res,
            jump, rgrad, jump_rgrad, b0_star, mean_d, d_snap_iters, d_snapshots.
        logd_net: Trained logD network (for reuse or inspection).
        local_op: Trained local operator network (for reuse or inspection).
    """

    x_res: torch.Tensor
    logd: torch.Tensor
    d_pred: torch.Tensor
    u_hat_unit: torch.Tensor
    u_pred: torch.Tensor
    b0_star: float
    history: Dict[str, List[float]]
    logd_net: Optional[nn.Module] = None
    local_op: Optional[nn.Module] = None


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
        self.use_rff = use_rff
        self.geom_layer = nn.Linear(2, width)
        if use_rff:
            self.geom_layer.weight.requires_grad = False
            if self.geom_layer.bias is not None:
                self.geom_layer.bias.requires_grad = False
        # NOTE: logd_embed is intentionally trainable. The original BiLO draft
        # froze this embedding, but trainable embeddings are more flexible.
        self.logd_embed = nn.Linear(1, width, bias=False)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(3)])
        self.output = nn.Linear(width, 1)
        self.activation = torch.tanh

    def forward(
        self, x: torch.Tensor, logd: torch.Tensor, z_known: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate u_hat and return the |x-z| feature for jump constraints."""
        x = x.view(-1, 1)
        logd = logd.view(-1, 1)
        phi_z = torch.abs(x - z_known)
        if phi_z.is_leaf and not phi_z.requires_grad:
            phi_z.requires_grad_(True)
        geom_in = torch.cat([x, phi_z], dim=1)
        geom_lin = self.geom_layer(geom_in)
        embed = self.logd_embed(logd)
        if self.use_rff:
            h = self.activation(torch.sin(2.0 * torch.pi * geom_lin) + embed)
        else:
            h = self.activation(geom_lin + embed)
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


def _trainable_params(module: nn.Module) -> List[nn.Parameter]:
    """Return list of trainable parameters (excludes frozen RFF layers)."""
    return [p for p in module.parameters() if p.requires_grad]


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


def _calc_data_loss(
    logd_net: nn.Module,
    local_op: nn.Module,
    x_res: torch.Tensor,
    x_int: torch.Tensor,
    x_field: torch.Tensor,
    z_tensor: torch.Tensor,
    mode: str,
    u_true: Optional[torch.Tensor],
    ppp: Optional[PPPData],
    field_loss: str,
    log_target: float,
    smoothness_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute data and regularization terms for the upper-level objective."""
    logd_int = logd_net(x_int)
    u_hat_int, _ = local_op(x_int, logd_int, z_tensor)

    if mode == "field":
        logd_field = logd_net(x_field)
        u_hat_field, _ = local_op(x_field, logd_field, z_tensor)
        b0_star = varpro.project_b0_field(u_hat_field, u_true, field_loss=field_loss)
        data_loss = varpro.field_data_loss(u_hat_field, u_true, b0_star, field_loss=field_loss)
    else:
        integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
        b0_star = varpro.project_b0_ppp(ppp.n_obs, ppp.m_obs, integral_unit)
        logd_data = logd_net(ppp.x_particles)
        u_hat_data, _ = local_op(ppp.x_particles, logd_data, z_tensor)
        data_loss = varpro.ppp_nll(u_hat_data.view(-1), b0_star, ppp.m_obs, integral_unit)

    x_reg = x_res.clone().detach().requires_grad_(True)
    logd_reg = logd_net(x_reg)
    if smoothness_type == "tv":
        reg_smooth = physics.tv_smoothness_logd(x_reg, logd_reg)
    else:
        reg_smooth = physics.h1_smoothness_logd(x_reg, logd_reg)
    # Reuse logd_reg (same x locations) to avoid an extra forward pass.
    reg_scale = physics.log_scale_anchor(logd_reg, log_target)
    return b0_star, data_loss, reg_smooth, reg_scale


def _calc_physics_loss(
    logd_net: nn.Module,
    local_op: nn.Module,
    x_res: torch.Tensor,
    z_tensor: torch.Tensor,
    z_idx: int,
    alpha: float,
    mu: float,
    w_jump: float,
    w_resgrad: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute PDE residual and jump penalties for the lower-level objective.

    Args:
        z_idx: Index of the source location on the grid (excluded from residual).
    """
    x_pde = x_res.clone().detach().requires_grad_(True)
    logd_pde = logd_net(x_pde)
    u_hat_pde, _ = local_op(x_pde, logd_pde, z_tensor)
    residual = _alpha_flux_residual(x_pde, logd_pde, u_hat_pde, alpha, mu)
    # Exclude the source point from residual loss
    n = residual.shape[0]
    res_loss = (torch.sum(residual ** 2) - residual[z_idx] ** 2) / (n - 1)

    z_probe = z_tensor.clone().detach().requires_grad_(True)
    logd_z = logd_net(z_probe)
    u_hat_z, phi_z = local_op(z_probe, logd_z, z_tensor)
    du_dphi = torch.autograd.grad(
        u_hat_z,
        phi_z,
        grad_outputs=torch.ones_like(u_hat_z),
        create_graph=True,
    )[0]
    jump_res = torch.exp(logd_z) * (2.0 * du_dphi) + 1.0
    jump_loss = torch.mean(jump_res ** 2)

    if w_resgrad > 0.0:
        # NOTE: grad_outputs=ones is sufficient here because the residual is
        # evaluated pointwise, so the Jacobian is diagonal in practice.
        grad_jump = torch.autograd.grad(
            jump_res,
            logd_z,
            grad_outputs=torch.ones_like(jump_res),
            create_graph=True,
            allow_unused=True,
        )[0]
        jump_rgrad = torch.mean(grad_jump ** 2) if grad_jump is not None else torch.tensor(
            0.0, device=x_res.device, dtype=x_res.dtype
        )
        # Zero out the source point in grad_outputs for resgrad
        grad_outputs = torch.ones_like(residual)
        grad_outputs[z_idx] = 0.0
        grad_res = torch.autograd.grad(
            residual,
            logd_pde,
            grad_outputs=grad_outputs,
            create_graph=True,
            allow_unused=True,
        )[0]
        rgrad = torch.mean(grad_res ** 2) if grad_res is not None else torch.tensor(
            0.0, device=x_res.device, dtype=x_res.dtype
        )
    else:
        rgrad = torch.tensor(0.0, device=x_res.device, dtype=x_res.dtype)
        jump_rgrad = torch.tensor(0.0, device=x_res.device, dtype=x_res.dtype)

    lower_loss = res_loss + w_jump * jump_loss + w_resgrad * (rgrad + jump_rgrad)
    return lower_loss, res_loss, jump_loss, rgrad, jump_rgrad


def fit(data_bundle: BiLOData, cfg: Config, verbose: bool = True) -> BiLOResult:
    """Fit log-D with bilevel optimization and return reconstructed fields.

    Args:
        data_bundle: BiLOData specifying observations and grids.
        cfg: Full configuration (physics/data/grid/train/reg/arch/run).
        verbose: Print progress during training.

    Returns:
        BiLOResult with predictions on the solver grid (x_res) and training history.
    """
    device = cfg.run.torch_device
    dtype = cfg.run.torch_dtype
    domain = cfg.physics.domain

    if len(cfg.physics.sources) != 1:
        raise NotImplementedError("BiLO currently supports a single source.")

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
    # Since z is exactly on the aligned grid, find the single index to exclude from residual.
    z_idx = int(torch.argmin(torch.abs(x_res - z_tensor)).item())

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
        print(f"[BiLO] Initialized ⟨D⟩_base: {d_init_base:.3e}")
    log_target = math.log(d_init_base)

    logd_init = _init_logd_profile(
        x_res.view(-1),
        base=d_init_base,
        scale=cfg.d_profile.d_init_pert_scale,
        freq=cfg.d_profile.d_init_pert_freq,
    ).view(-1, 1)

    logd_net = LogDNet(width=cfg.arch.rff_width, use_rff=cfg.arch.use_rff_logd).to(device)
    local_op = LocalOperator(width=cfg.arch.rff_width, use_rff=cfg.arch.use_rff_geom).to(device)

    x_int = torch.linspace(
        domain[0], domain[1], cfg.grid.n_int, device=device, dtype=dtype
    ).view(-1, 1)

    history: Dict[str, List[float]] = {
        "iter": [],
        "upper": [],
        "data": [],
        "reg_smooth": [],
        "reg_scale": [],
        "lower": [],
        "res": [],
        "jump": [],
        "rgrad": [],
        "jump_rgrad": [],
        "b0_star": [],
        "mean_d": [],
        "d_snap_iters": [],
        "d_snapshots": [],
    }

    # Pretraining (still sequential for supervised warmup)
    if cfg.train.pretrain_iters > 0:
        u_init_np = physics.fdm_solve_alpha_dirichlet(
            logd_init.detach().cpu().numpy().reshape(-1),
            cfg.physics.alpha,
            cfg.physics.mu,
            x_res.view(-1).detach().cpu().numpy(),
            1.0,
            cfg.physics.sources,
        )
        u_init_target = torch.tensor(u_init_np, device=device, dtype=dtype).view(-1, 1)
        opt_d = torch.optim.Adam(logd_net.parameters(), lr=cfg.train.lr_d_pre)
        opt_l = torch.optim.Adam(local_op.parameters(), lr=cfg.train.lr_lower_pre)
        for step in range(cfg.train.pretrain_iters):
            opt_d.zero_grad(set_to_none=True)
            anchor_loss = torch.mean((logd_net(x_res) - logd_init) ** 2)
            anchor_loss.backward()
            opt_d.step()

            opt_l.zero_grad(set_to_none=True)
            lower, res_loss, jump_loss, rgrad, jump_rgrad = _calc_physics_loss(
                logd_net,
                local_op,
                x_res,
                z_tensor,
                z_idx,
                cfg.physics.alpha,
                cfg.physics.mu,
                cfg.reg.w_jump,
                cfg.reg.w_resgrad,
            )
            logd_curr = logd_net(x_res).detach()
            u_pred, _ = local_op(x_res, logd_curr, z_tensor)
            loss_sup = torch.mean((u_pred - u_init_target) ** 2)
            (lower + loss_sup).backward()

            if verbose and step % cfg.train.log_every == 0:
                with torch.no_grad():
                    mean_d = torch.mean(torch.exp(logd_curr)).item()
                    pre_total = (anchor_loss + lower + loss_sup).item()
                print(
                    f"[BiLO|pretrain] Iter {step:05d} | Ltot: {pre_total:.3e}\n"
                    f"  Lanchor: {anchor_loss.item():.3e} | Llower: {lower.item():.3e} | Lsup: {loss_sup.item():.3e}\n"
                    f"  Lres: {res_loss.item():.3e} | Ljump: {jump_loss.item():.3e} | "
                    f"Lrgrad: {rgrad.item():.3e} | Ljump_rgrad: {jump_rgrad.item():.3e}\n"
                    f"  ⟨D⟩: {mean_d:.3e}"
                )

            opt_l.step()

    # Final pretrain log
    if cfg.train.pretrain_iters > 0 and verbose:
        # NOTE: _calc_physics_loss uses autograd.grad, so keep gradients enabled here.
        with torch.enable_grad():
            anchor_loss = torch.mean((logd_net(x_res) - logd_init) ** 2)
            lower, res_loss, jump_loss, rgrad, jump_rgrad = _calc_physics_loss(
                logd_net,
                local_op,
                x_res,
                z_tensor,
                z_idx,
                cfg.physics.alpha,
                cfg.physics.mu,
                cfg.reg.w_jump,
                cfg.reg.w_resgrad,
            )
            logd_curr = logd_net(x_res).detach()
            u_pred, _ = local_op(x_res, logd_curr, z_tensor)
            loss_sup = torch.mean((u_pred - u_init_target) ** 2)
            mean_d = torch.mean(torch.exp(logd_curr)).item()
            pre_total = (anchor_loss + lower + loss_sup).item()
        print(
            f"[BiLO|pretrain] Iter {cfg.train.pretrain_iters:05d} | Ltot: {pre_total:.3e}\n"
            f"  Lanchor: {anchor_loss.item():.3e} | Llower: {lower.item():.3e} | Lsup: {loss_sup.item():.3e}\n"
            f"  Lres: {res_loss.item():.3e} | Ljump: {jump_loss.item():.3e} | "
            f"Lrgrad: {rgrad.item():.3e} | Ljump_rgrad: {jump_rgrad.item():.3e}\n"
            f"  ⟨D⟩: {mean_d:.3e}"
        )

    # Fine-tuning: Simultaneous Optimization.
    # We intentionally compute isolated gradients for each loss and assign them to
    # disjoint parameter groups before a single optimizer step. This matches the
    # BiLO simultaneous (Jacobi-style) updates and avoids cross-terms from a joint
    # backward (do NOT replace with (upper_loss + lower_loss).backward()).
    logd_params = _trainable_params(logd_net)
    local_op_params = _trainable_params(local_op)

    # Single optimizer with parameter groups is equivalent to two optimizers
    # stepped once per iteration, while keeping a single scheduler interface.
    optimizer = torch.optim.Adam([
        {"params": logd_params, "lr": cfg.train.lr_d_fine},
        {"params": local_op_params, "lr": cfg.train.lr_lower_fine},
    ])

    scheduler = None
    if cfg.train.use_scheduler and cfg.train.finetune_iters > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.finetune_iters,
            eta_min=min(cfg.train.lr_d_fine, cfg.train.lr_lower_fine) * 0.1,
        )

    best_total: Optional[float] = None
    patience = 0

    for step in range(cfg.train.finetune_iters + 1):
        optimizer.zero_grad(set_to_none=True)

        # 1. Upper Level: Minimize Data Loss (w.r.t logd_net)
        b0_star, data_loss, reg_smooth, reg_scale = _calc_data_loss(
            logd_net,
            local_op,
            x_res,
            x_int,
            x_field,
            z_tensor,
            data_bundle.mode,
            u_true,
            ppp,
            cfg.data.field_loss,
            log_target,
            cfg.reg.smoothness_type,
        )
        upper_loss = (
            data_loss + cfg.reg.wreg_smooth * reg_smooth + cfg.reg.wreg_scale * reg_scale
        )

        # Compute upper gradients for logd_net only (treat local_op as constant).
        grads_upper = torch.autograd.grad(
            upper_loss,
            logd_params,
            create_graph=False,
            allow_unused=True,
        )
        for param, grad in zip(logd_params, grads_upper):
            if grad is not None:
                param.grad = grad

        # 2. Lower Level: Minimize Physics Loss (w.r.t local_op)
        lower_loss, res_loss, jump_loss, rgrad, jump_rgrad = _calc_physics_loss(
            logd_net,
            local_op,
            x_res,
            z_tensor,
            z_idx,
            cfg.physics.alpha,
            cfg.physics.mu,
            cfg.reg.w_jump,
            cfg.reg.w_resgrad,
        )

        # Compute lower gradients for local_op only (treat logd_net as constant).
        grads_lower = torch.autograd.grad(
            lower_loss,
            local_op_params,
            create_graph=False,
            allow_unused=True,
        )
        for param, grad in zip(local_op_params, grads_lower):
            if grad is not None:
                param.grad = grad

        # Logging
        total_loss = upper_loss + lower_loss
        if step % cfg.train.log_every == 0:
            with torch.no_grad():
                logd_res = logd_net(x_res)
                d_vals = torch.exp(logd_res)
                mean_d = torch.mean(d_vals).item()
                u_hat_res, _ = local_op(x_res, logd_res, z_tensor)
                if data_bundle.mode == "particles":
                    logd_int = logd_net(x_int)
                    u_hat_int, _ = local_op(x_int, logd_int, z_tensor)
                    integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
                else:
                    integral_unit = torch.trapezoid(u_hat_res.view(-1), x_res.view(-1))
                u_int = (b0_star * integral_unit).item()
                d_snapshot = d_vals.detach().cpu().numpy().reshape(-1)
            history["iter"].append(step)
            history["upper"].append(upper_loss.item())
            history["data"].append(data_loss.item())
            history["reg_smooth"].append(reg_smooth.item())
            history["reg_scale"].append(reg_scale.item())
            history["lower"].append(lower_loss.item())
            history["res"].append(res_loss.item())
            history["jump"].append(jump_loss.item())
            history["rgrad"].append(rgrad.item())
            history["jump_rgrad"].append(jump_rgrad.item())
            history["b0_star"].append(b0_star.item())
            history["mean_d"].append(mean_d)
            history["d_snap_iters"].append(step)
            history["d_snapshots"].append(d_snapshot)
            if verbose:
                loss_name = cfg.data.field_loss if data_bundle.mode == "field" else "ppp"
                reg_smooth_eff = cfg.reg.wreg_smooth * reg_smooth
                reg_scale_eff = cfg.reg.wreg_scale * reg_scale
                print(
                    f"[BiLO|finetune] Iter {step:05d} | Ltot: {total_loss.item():.3e}\n"
                    f"  Upper: {upper_loss.item():.3e} | Ldata({loss_name}): {data_loss.item():.3e} | "
                    f"RegSmooth: {reg_smooth.item():.3e} (eff: {reg_smooth_eff.item():.3e}) | "
                    f"RegScale: {reg_scale.item():.3e} (eff: {reg_scale_eff.item():.3e})\n"
                    f"  Lower: {lower_loss.item():.3e} | Lres: {res_loss.item():.3e} | Ljump: {jump_loss.item():.3e} | "
                    f"Lrgrad: {rgrad.item():.3e} | Ljump_rgrad: {jump_rgrad.item():.3e}\n"
                    f"  b₀*: {b0_star.item():.2f} | ∫u: {u_int:.3e} | ⟨D⟩: {mean_d:.3e}"
                )

        # Early Stopping check
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

        # 3. Simultaneous Update
        if step < cfg.train.finetune_iters and not stop_training:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            if stop_training and verbose:
                 print(f"[BiLO] Early stopping triggered at step {step}.")
            break

    with torch.no_grad():
        logd_final = logd_net(x_res)
        u_hat_res, _ = local_op(x_res, logd_final, z_tensor)
        if data_bundle.mode == "field":
            logd_field = logd_net(x_field)
            u_hat_field, _ = local_op(x_field, logd_field, z_tensor)
            b0_star = varpro.project_b0_field(
                u_hat_field, u_true, field_loss=cfg.data.field_loss
            )
        else:
            u_hat_int, _ = local_op(x_int, logd_net(x_int), z_tensor)
            integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
            b0_star = varpro.project_b0_ppp(ppp.n_obs, ppp.m_obs, integral_unit)
        u_pred = b0_star * u_hat_res
        d_pred = torch.exp(logd_final)

    return BiLOResult(
        x_res=x_res.detach().cpu().view(-1),
        logd=logd_final.detach().cpu().view(-1),
        d_pred=d_pred.detach().cpu().view(-1),
        u_hat_unit=u_hat_res.detach().cpu().view(-1),
        u_pred=u_pred.detach().cpu().view(-1),
        b0_star=float(b0_star.item()),
        history=history,
        logd_net=logd_net,
        local_op=local_op,
    )
