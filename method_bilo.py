"""BiLO (bilevel optimization) solver for the 1D alpha-PDE inverse problem.

Defines BiLOData/BiLOResult, network modules for D and the local operator,
plus bilevel pretrain/finetune training loops.
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
from training_logger import (
    TrainingHistory,
    format_bilo_progress,
    format_bilo_pretrain_progress,
)

# =============================================================================
# D PARAMETERIZATION: Softplus + offset
# =============================================================================
# We use softplus for positivity: D = softplus(raw) + D_min
# Softplus has mild gradient suppression but avoids catastrophic gradient death.
# =============================================================================

D_MIN = 1e-6


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
    x_field: Optional[torch.Tensor] = None
    u_true: Optional[torch.Tensor] = None
    ppp: Optional[PPPData] = None


@dataclass
class BiLOResult:
    """Outputs and training history for BiLO fitting.

    Returned tensors are detached and on CPU for convenience.
    """

    x_res: torch.Tensor
    d_pred: torch.Tensor
    u_hat_unit: torch.Tensor
    u_pred: torch.Tensor
    b0_star: float
    history: Dict[str, List[float]]
    d_net: Optional[nn.Module] = None
    local_op: Optional[nn.Module] = None


class DNet(nn.Module):
    """RFF-embedded MLP that parameterizes D(x) with softplus + offset."""

    def __init__(
        self, width: int = 128, use_rff: bool = True, rff_scale: float = 1.0, d_min: float = D_MIN
    ) -> None:
        super().__init__()
        self.use_rff = use_rff
        self.rff_scale = rff_scale
        self.d_min = d_min
        self.embed = nn.Linear(1, width)
        if use_rff:
            for param in self.embed.parameters():
                param.requires_grad = False
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        if self.use_rff:
            feat = torch.sin(2.0 * torch.pi * self.rff_scale * self.embed(x))
        else:
            feat = self.embed(x)
        raw = self.net(feat)
        return F.softplus(raw) + self.d_min


class LocalOperator(nn.Module):
    """Local operator network for the unit response u_hat(x), conditioned on D."""

    def __init__(
        self,
        width: int = 128,
        use_rff: bool = True,
        rff_scale: float = 1.0,
        bc_type: str = "dirichlet",
    ) -> None:
        super().__init__()
        self.use_rff = use_rff
        self.rff_scale = rff_scale
        self.bc_type = bc_type.strip().lower()
        if self.bc_type not in {"dirichlet", "neumann"}:
            raise ValueError(f"Unsupported bc_type '{bc_type}'.")
        self.geom_layer = nn.Linear(2, width)
        if use_rff:
            self.geom_layer.weight.requires_grad = False
            if self.geom_layer.bias is not None:
                self.geom_layer.bias.requires_grad = False
        self.d_embed = nn.Linear(1, width, bias=False)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(3)])
        self.output = nn.Linear(width, 1)
        self.activation = F.silu

    def forward(
        self, x: torch.Tensor, d: torch.Tensor, z_known: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, 1)
        d = d.view(-1, 1)

        phi_z = torch.abs(x - z_known)
        if phi_z.is_leaf and not phi_z.requires_grad:
            phi_z.requires_grad_(True)
        geom_in = torch.cat([x, phi_z], dim=1)
        geom_lin = self.geom_layer(geom_in)
        embed = self.d_embed(torch.log(d))
        if self.use_rff:
            h = self.activation(torch.sin(2.0 * torch.pi * self.rff_scale * geom_lin) + embed)
        else:
            h = self.activation(geom_lin + embed)
        for layer in self.hidden:
            h = self.activation(layer(h))
        u_raw = self.output(h)
        u_pos = F.softplus(u_raw)
        if self.bc_type == "dirichlet":
            u = u_pos * x * (1.0 - x)
        else:
            u = u_pos
        return u, phi_z


def _init_d_profile(
    x: torch.Tensor, base: float, scale: float, freq: float
) -> torch.Tensor:
    """Build a sinusoidal D initialization on the grid."""
    if scale >= 1.0:
        raise ValueError("pert_scale must be < 1 to keep D_init positive.")
    return base * (1.0 + scale * torch.sin(2.0 * torch.pi * freq * x))


def _trainable_params(module: nn.Module) -> List[nn.Parameter]:
    """Return list of trainable parameters (excludes frozen RFF layers)."""
    return [p for p in module.parameters() if p.requires_grad]


def _alpha_flux_residual(
    x: torch.Tensor, d: torch.Tensor, u: torch.Tensor, alpha: float, mu: float
) -> torch.Tensor:
    """Flux-form PDE residual L_alpha(u) - mu*u using autograd."""
    if not x.requires_grad:
        raise ValueError("x must have requires_grad=True for autograd residuals.")
    ones = torch.ones_like(u)
    q = (d ** (1.0 - alpha)) * u
    q_x = torch.autograd.grad(q, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    J = (d ** alpha) * q_x
    J_x = torch.autograd.grad(J, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    return J_x - mu * u


def _compute_bc_loss_neumann(
    d_net: nn.Module,
    local_op: nn.Module,
    z_tensor: torch.Tensor,
    domain: Tuple[float, float],
    device: torch.device,
    dtype: torch.dtype,
):
    """Penalize boundary derivatives for Neumann BCs: u'(x_min)=u'(x_max)=0."""
    x0 = torch.tensor([[domain[0]]], device=device, dtype=dtype, requires_grad=True)
    x1 = torch.tensor([[domain[1]]], device=device, dtype=dtype, requires_grad=True)

    d0 = d_net(x0)
    d1 = d_net(x1)
    u0, _ = local_op(x0, d0, z_tensor)
    u1, _ = local_op(x1, d1, z_tensor)

    u0_x = torch.autograd.grad(u0, x0, grad_outputs=torch.ones_like(u0), create_graph=True)[0]
    u1_x = torch.autograd.grad(u1, x1, grad_outputs=torch.ones_like(u1), create_graph=True)[0]

    Du0_x_D = torch.autograd.grad(d0 * u0_x, d0, grad_outputs=torch.ones_like(u0), create_graph=True)[0]
    Du1_x_D = torch.autograd.grad(d1 * u1_x, d1, grad_outputs=torch.ones_like(u1), create_graph=True)[0]

    bc_loss = torch.mean(u0_x ** 2 + u1_x ** 2)

    bc_grad_loss =  torch.mean(Du0_x_D ** 2 + Du1_x_D ** 2)

    return bc_loss, bc_grad_loss


def _calc_data_loss(
    d_net: nn.Module,
    local_op: nn.Module,
    x_res: torch.Tensor,
    x_int: torch.Tensor,
    x_field: torch.Tensor,
    z_tensor: torch.Tensor,
    mode: str,
    u_true: Optional[torch.Tensor],
    ppp: Optional[PPPData],
    field_loss: str,
    d_target: float,
    smoothness_type: str,
    b0_fixed_value: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute data and regularization terms for the upper-level objective."""
    d_int = d_net(x_int)
    u_hat_int, _ = local_op(x_int, d_int, z_tensor)

    if mode == "field":
        d_field = d_net(x_field)
        u_hat_field, _ = local_op(x_field, d_field, z_tensor)
        b0_star = varpro.get_b0_field(
            u_hat_field, u_true, field_loss=field_loss, b0_fixed_value=b0_fixed_value
        )
        data_loss = varpro.field_data_loss(u_hat_field, u_true, b0_star, field_loss=field_loss)
    else:
        integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
        b0_star = varpro.get_b0_ppp(ppp.n_obs, ppp.m_obs, integral_unit, b0_fixed_value=b0_fixed_value)
        d_data = d_net(ppp.x_particles)
        u_hat_data, _ = local_op(ppp.x_particles, d_data, z_tensor)
        data_loss = varpro.ppp_nll(u_hat_data.view(-1), b0_star, ppp.m_obs, integral_unit)

    x_reg = x_res.clone().detach().requires_grad_(True)
    d_reg = d_net(x_reg)
    if smoothness_type == "tv":
        reg_smooth = physics.tv_smoothness_d(x_reg, d_reg)
    else:
        reg_smooth = physics.h1_smoothness_d(x_reg, d_reg)

    reg_scale = physics.scale_anchor(d_reg, d_target)
    return b0_star, data_loss, reg_smooth, reg_scale


def _calc_physics_loss(
    d_net: nn.Module,
    local_op: nn.Module,
    x_res: torch.Tensor,
    z_tensor: torch.Tensor,
    z_idx: int,
    alpha: float,
    mu: float,
    w_jump: float,
    w_resgrad: float,
    bc_type: str,
    domain: Tuple[float, float],
    w_bc: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute PDE residual, jump, and BC penalties for the lower-level objective."""
    x_pde = x_res.clone().detach().requires_grad_(True)
    d_pde = d_net(x_pde)
    u_hat_pde, _ = local_op(x_pde, d_pde, z_tensor)
    residual = _alpha_flux_residual(x_pde, d_pde, u_hat_pde, alpha, mu)
    n = residual.shape[0]
    res_loss = (torch.sum(residual ** 2) - residual[z_idx] ** 2) / (n - 1)

    z_probe = z_tensor.clone().detach().requires_grad_(True)
    d_z = d_net(z_probe)
    u_hat_z, phi_z = local_op(z_probe, d_z, z_tensor)
    du_dphi = torch.autograd.grad(
        u_hat_z, phi_z, grad_outputs=torch.ones_like(u_hat_z), create_graph=True
    )[0]
    jump_res = d_z * (2.0 * du_dphi) + 1.0
    jump_loss = torch.mean(jump_res ** 2)

    bc_loss = torch.tensor(0.0, device=x_res.device, dtype=x_res.dtype)
    bc_grad_loss = torch.tensor(0.0, device=x_res.device, dtype=x_res.dtype)
    if bc_type == "neumann":
        bc_loss, bc_grad_loss = _compute_bc_loss_neumann(d_net, local_op, z_tensor, domain, x_res.device, x_res.dtype)

    if w_resgrad > 0.0:
        grad_jump = torch.autograd.grad(
            jump_res, d_z, grad_outputs=torch.ones_like(jump_res), create_graph=True, allow_unused=True
        )[0]
        d_scale_jump = d_z.clamp(min=D_MIN)
        jump_rgrad = torch.mean((grad_jump * d_scale_jump) ** 2) if grad_jump is not None else torch.tensor(0.0, device=x_res.device, dtype=x_res.dtype)

        grad_outputs = torch.ones_like(residual)
        grad_outputs[z_idx] = 0.0
        grad_res = torch.autograd.grad(
            residual, d_pde, grad_outputs=grad_outputs, create_graph=True, allow_unused=True
        )[0]
        d_scale_pde = d_pde.clamp(min=D_MIN)
        rgrad = torch.mean((grad_res * d_scale_pde) ** 2) if grad_res is not None else torch.tensor(0.0, device=x_res.device, dtype=x_res.dtype)
    else:
        rgrad = torch.tensor(0.0, device=x_res.device, dtype=x_res.dtype)
        jump_rgrad = torch.tensor(0.0, device=x_res.device, dtype=x_res.dtype)

    lower_loss = res_loss + w_jump * jump_loss + w_resgrad * (rgrad + jump_rgrad)
    if bc_type == "neumann":
        lower_loss = lower_loss + w_bc * (bc_loss) + w_resgrad * bc_grad_loss
    return lower_loss, res_loss, jump_loss, bc_loss, rgrad, jump_rgrad, bc_grad_loss


def fit(data_bundle: BiLOData, cfg: Config, verbose: bool = True) -> BiLOResult:
    """Fit D with bilevel optimization and return reconstructed fields.

    Args:
        data_bundle: BiLOData specifying observations and grids.
        cfg: Full configuration (physics/data/grid/train/reg/arch/run).
        verbose: Print progress during training.

    Returns:
        BiLOResult with predictions on the solver grid (x_res) and training history.
    """
    # =========================================================================
    # CONFIG EXTRACTION
    # =========================================================================
    device = cfg.run.torch_device
    dtype = cfg.run.torch_dtype

    # Physics
    alpha = cfg.physics.alpha
    mu = cfg.physics.mu
    sources = cfg.physics.sources
    domain = cfg.physics.domain
    bc_type = cfg.physics.bc_type.lower()

    # Data
    mode = data_bundle.mode
    field_loss_type = cfg.data.field_loss
    b0_fixed_value = cfg.data.b0_fixed_value

    # D profile initialization
    use_ddi = cfg.d_profile.use_ddi
    ddi_d_min, ddi_d_max = cfg.d_profile.ddi_d_min, cfg.d_profile.ddi_d_max
    pert_scale, pert_freq = cfg.d_profile.pert_scale, cfg.d_profile.pert_freq

    # Regularization
    w_jump = cfg.reg.w_jump
    w_resgrad = cfg.reg.w_resgrad
    w_bc = cfg.reg.w_bc
    wreg_smooth, wreg_scale = cfg.reg.wreg_smooth, cfg.reg.wreg_scale
    smoothness_type = cfg.reg.smoothness_type

    # Training
    pretrain_iters = cfg.train.pretrain_iters
    finetune_iters = cfg.train.finetune_iters
    lr_d_pre, lr_lower_pre = cfg.train.lr_d_pre, cfg.train.lr_lower_pre
    lr_d_fine, lr_lower_fine = cfg.train.lr_d_fine, cfg.train.lr_lower_fine
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
    rff_scale = getattr(cfg.arch, "rff_scale", 1.0)
    rff_width = cfg.arch.rff_width
    use_rff = cfg.arch.use_rff

    # Grid
    n_int = cfg.grid.n_int

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================
    if len(sources) != 1:
        raise NotImplementedError("BiLO currently supports a single source.")

    if bc_type == "neumann" and alpha != 1.0:
        print(f"[WARN] Neumann BCs with alpha={alpha} may be non-identifiable.")

    # =========================================================================
    # DATA PREPARATION
    # =========================================================================
    x_res = data_bundle.x_res.to(device=device, dtype=dtype).view(-1, 1)
    x_field = x_res
    if data_bundle.x_field is not None:
        x_field = data_bundle.x_field.to(device=device, dtype=dtype).view(-1, 1)

    if mode == "field":
        u_true = data_bundle.u_true.to(device=device, dtype=dtype).view(-1, 1)
        ppp = None
    else:
        ppp = PPPData(
            x_particles=data_bundle.ppp.x_particles.to(device=device, dtype=dtype).view(-1, 1),
            m_obs=data_bundle.ppp.m_obs,
        )
        u_true = None

    z_tensor = torch.tensor(sources[0], device=device, dtype=dtype).view(1, 1)
    z_idx = int(torch.argmin(torch.abs(x_res - z_tensor)).item())

    x_int = torch.linspace(domain[0], domain[1], n_int, device=device, dtype=dtype).view(-1, 1)

    # =========================================================================
    # SCALE ESTIMATION
    # =========================================================================
    if use_ddi:
        d_ddi = estimate_ddi_scale(
            mu=mu,
            z=sources[0],
            x_particles=ppp.x_particles if ppp is not None else None,
            u_field=u_true if u_true is not None else None,
            x_grid=x_field.view(-1),
            d_min=ddi_d_min,
            d_max=ddi_d_max,
        )
    else:
        d_ddi = 1.0

    if scalar_fit_iters > 0:
        d_scale = fit_constant_d(
            x=x_res.view(-1),
            alpha=alpha,
            mu=mu,
            sources=sources,
            u_true=u_true if u_true is not None else None,
            ppp=ppp if ppp is not None else None,
            x_field=x_field.view(-1) if u_true is not None else None,
            x_int=x_int.view(-1) if ppp is not None else None,
            d_init=d_ddi,
            max_iters=scalar_fit_iters,
            field_loss=field_loss_type,
            bc_type=bc_type,
            verbose=verbose,
        )
    else:
        d_scale = d_ddi

    if verbose:
        print(f"[BiLO] DDI scale: {d_ddi:.3e}")
        print(f"[BiLO] Scalar fit scale: {d_scale:.3e}")

    d_target = d_scale

    # =========================================================================
    # MODEL INITIALIZATION
    # =========================================================================
    d_init_profile = _init_d_profile(x_res.view(-1), base=d_scale, scale=pert_scale, freq=pert_freq).view(-1, 1)

    d_net = DNet(width=rff_width, use_rff=use_rff, rff_scale=rff_scale, d_min=d_min).to(device=device, dtype=dtype)
    local_op = LocalOperator(width=rff_width, use_rff=use_rff, rff_scale=rff_scale, bc_type=bc_type).to(device=device, dtype=dtype)

    # =========================================================================
    # TRAINING HISTORY
    # =========================================================================
    history = TrainingHistory.for_bilo()

    # =========================================================================
    # PRETRAIN PHASE
    # =========================================================================
    if pretrain_iters > 0:
        d_init_np = d_init_profile.detach().cpu().numpy().reshape(-1)
        u_init_np = physics.fdm_solve_alpha(
            d_init_np, alpha, mu, x_res.view(-1).detach().cpu().numpy(),
            1.0, sources, bc_type=bc_type
        )
        u_init_target = torch.tensor(u_init_np, device=device, dtype=dtype).view(-1, 1)

        opt_d = torch.optim.Adam(d_net.parameters(), lr=lr_d_pre)
        opt_l = torch.optim.Adam(local_op.parameters(), lr=lr_lower_pre)

        for step in range(pretrain_iters):
            opt_d.zero_grad(set_to_none=True)
            d_pred = d_net(x_res)
            anchor_loss = torch.mean((d_pred - d_init_profile) ** 2)
            anchor_loss.backward()
            opt_d.step()

            opt_l.zero_grad(set_to_none=True)
            lower, res_loss, jump_loss, bc_loss, rgrad, jump_rgrad, bc_grad_loss = _calc_physics_loss(
                d_net, local_op, x_res, z_tensor, z_idx, alpha, mu,
                w_jump, w_resgrad, bc_type, domain, w_bc
            )
            d_curr = d_net(x_res).detach()
            u_pred, _ = local_op(x_res, d_curr, z_tensor)
            loss_sup = torch.mean((u_pred - u_init_target) ** 2)
            (lower + loss_sup).backward()

            if verbose and step % log_every == 0:
                with torch.no_grad():
                    mean_d = torch.mean(d_curr).item()
                    pre_total = (anchor_loss + lower + loss_sup).item()
                print(format_bilo_pretrain_progress(
                    step=step,
                    total=pre_total,
                    anchor=anchor_loss.item(),
                    lower=lower.item(),
                    sup=loss_sup.item(),
                    res=res_loss.item(),
                    jump=jump_loss.item(),
                    bc=bc_loss.item(),
                    bc_grad=bc_grad_loss.item(),
                    rgrad=rgrad.item(),
                    jump_rgrad=jump_rgrad.item(),
                    mean_d=mean_d,
                    bc_type=bc_type,
                ))

            opt_l.step()

        # Final pretrain log
        if verbose:
            with torch.enable_grad():
                d_pred = d_net(x_res)
                anchor_loss = torch.mean((d_pred - d_init_profile) ** 2)
                lower, res_loss, jump_loss, bc_loss, rgrad, jump_rgrad, bc_grad_loss = _calc_physics_loss(
                    d_net, local_op, x_res, z_tensor, z_idx, alpha, mu,
                    w_jump, w_resgrad, bc_type, domain, w_bc
                )
                d_curr = d_net(x_res).detach()
                u_pred, _ = local_op(x_res, d_curr, z_tensor)
                loss_sup = torch.mean((u_pred - u_init_target) ** 2)
                mean_d = torch.mean(d_curr).item()
                pre_total = (anchor_loss + lower + loss_sup).item()
            print(format_bilo_pretrain_progress(
                step=pretrain_iters,
                total=pre_total,
                anchor=anchor_loss.item(),
                lower=lower.item(),
                sup=loss_sup.item(),
                res=res_loss.item(),
                jump=jump_loss.item(),
                bc=bc_loss.item(),
                bc_grad=bc_grad_loss.item(),
                rgrad=rgrad.item(),
                jump_rgrad=jump_rgrad.item(),
                mean_d=mean_d,
                bc_type=bc_type,
            ))

    # =========================================================================
    # FINETUNE OPTIMIZER SETUP
    # =========================================================================
    d_params = _trainable_params(d_net)
    local_op_params = _trainable_params(local_op)

    if use_lbfgs:
        optimizer = torch.optim.LBFGS(
            list(d_params) + list(local_op_params),
            lr=lbfgs_lr,
            max_iter=lbfgs_max_iter,
            history_size=10,
            line_search_fn="strong_wolfe",
        )
        scheduler = None
    else:
        optimizer = torch.optim.Adam([
            {"params": d_params, "lr": lr_d_fine},
            {"params": local_op_params, "lr": lr_lower_fine},
        ])
        scheduler = None
        if use_scheduler and finetune_iters > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=finetune_iters, eta_min=min(lr_d_fine, lr_lower_fine) * 0.1
            )

    # Early stopping state
    best_total: Optional[float] = None
    patience = 0

    # =========================================================================
    # MAIN BILEVEL TRAINING LOOP
    # =========================================================================
    for step in range(finetune_iters + 1):
        if not use_lbfgs:
            optimizer.zero_grad(set_to_none=True)

        # Upper level: data loss -> d_net gradients
        b0_star, data_loss, reg_smooth, reg_scale = _calc_data_loss(
            d_net, local_op, x_res, x_int, x_field, z_tensor,
            mode, u_true, ppp, field_loss_type, d_target, smoothness_type,
            b0_fixed_value=b0_fixed_value
        )
        upper_loss = data_loss + wreg_smooth * reg_smooth + wreg_scale * reg_scale

        if not use_lbfgs:
            grads_upper = torch.autograd.grad(upper_loss, d_params, create_graph=False, allow_unused=True)
            for param, grad in zip(d_params, grads_upper):
                if grad is not None:
                    param.grad = grad

        # Lower level: physics loss -> local_op gradients
        lower_loss, res_loss, jump_loss, bc_loss, rgrad, jump_rgrad, bc_grad_loss = _calc_physics_loss(
            d_net, local_op, x_res, z_tensor, z_idx, alpha, mu,
            w_jump, w_resgrad, bc_type, domain, w_bc
        )

        if not use_lbfgs:
            grads_lower = torch.autograd.grad(lower_loss, local_op_params, create_graph=False, allow_unused=True)
            for param, grad in zip(local_op_params, grads_lower):
                if grad is not None:
                    param.grad = grad

        # -----------------------------------------------------------------
        # LOGGING
        # -----------------------------------------------------------------
        total_loss = upper_loss + lower_loss
        if step % log_every == 0:
            with torch.no_grad():
                d_res = d_net(x_res)
                mean_d = torch.mean(d_res).item()
                u_hat_res, _ = local_op(x_res, d_res, z_tensor)
                if mode == "particles":
                    d_int = d_net(x_int)
                    u_hat_int, _ = local_op(x_int, d_int, z_tensor)
                    integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
                else:
                    integral_unit = torch.trapezoid(u_hat_res.view(-1), x_res.view(-1))
                d_snapshot = d_res.detach().cpu().numpy().reshape(-1)

            history.log(
                step=step,
                upper=upper_loss.item(),
                data=data_loss.item(),
                reg_smooth=reg_smooth.item(),
                reg_scale=reg_scale.item(),
                lower=lower_loss.item(),
                res=res_loss.item(),
                jump=jump_loss.item(),
                bc=bc_loss.item(),
                bc_grad=bc_grad_loss.item(),
                rgrad=rgrad.item(),
                jump_rgrad=jump_rgrad.item(),
                b0_star=b0_star.item(),
                mean_d=mean_d,
            )
            history.log_snapshot(step, d_snapshot)

            if verbose:
                loss_name = field_loss_type if mode == "field" else "ppp"
                print(format_bilo_progress(
                    step=step,
                    phase="finetune",
                    total=total_loss.item(),
                    upper=upper_loss.item(),
                    data=data_loss.item(),
                    reg_smooth=reg_smooth.item(),
                    reg_scale=reg_scale.item(),
                    lower=lower_loss.item(),
                    res=res_loss.item(),
                    jump=jump_loss.item(),
                    bc=bc_loss.item(),
                    rgrad=rgrad.item(),
                    jump_rgrad=jump_rgrad.item(),
                    wreg_smooth=wreg_smooth,
                    wreg_scale=wreg_scale,
                    b0_star=b0_star.item(),
                    integral_unit=integral_unit.item(),
                    mean_d=mean_d,
                    loss_name=loss_name,
                    bc_type=bc_type,
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
                def _lbfgs_closure() -> torch.Tensor:
                    optimizer.zero_grad(set_to_none=True)
                    _, data_loss, reg_smooth, reg_scale = _calc_data_loss(
                        d_net, local_op, x_res, x_int, x_field, z_tensor,
                        mode, u_true, ppp, field_loss_type, d_target, smoothness_type,
                        b0_fixed_value=b0_fixed_value
                    )
                    upper_loss = data_loss + wreg_smooth * reg_smooth + wreg_scale * reg_scale
                    grads_upper = torch.autograd.grad(upper_loss, d_params, create_graph=False, allow_unused=True)
                    for param, grad in zip(d_params, grads_upper):
                        if grad is not None:
                            param.grad = grad

                    lower_loss, _, _, _, _, _, _ = _calc_physics_loss(
                        d_net, local_op, x_res, z_tensor, z_idx, alpha, mu,
                        w_jump, w_resgrad, bc_type, domain, w_bc
                    )
                    grads_lower = torch.autograd.grad(lower_loss, local_op_params, create_graph=False, allow_unused=True)
                    for param, grad in zip(local_op_params, grads_lower):
                        if grad is not None:
                            param.grad = grad
                    return upper_loss + lower_loss
                optimizer.step(_lbfgs_closure)
            else:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        else:
            if stop_training and verbose:
                print(f"[BiLO] Early stopping triggered at step {step}.")
            break

    # =========================================================================
    # FINAL RESULT EXTRACTION
    # =========================================================================
    with torch.no_grad():
        d_final = d_net(x_res)
        u_hat_res, _ = local_op(x_res, d_final, z_tensor)
        if mode == "field":
            d_field = d_net(x_field)
            u_hat_field, _ = local_op(x_field, d_field, z_tensor)
            b0_star = varpro.get_b0_field(
                u_hat_field, u_true, field_loss=field_loss_type, b0_fixed_value=b0_fixed_value
            )
        else:
            u_hat_int, _ = local_op(x_int, d_net(x_int), z_tensor)
            integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
            b0_star = varpro.get_b0_ppp(ppp.n_obs, ppp.m_obs, integral_unit, b0_fixed_value=b0_fixed_value)
        u_pred = b0_star * u_hat_res
        d_pred = d_final

    return BiLOResult(
        x_res=x_res.detach().cpu().view(-1),
        d_pred=d_pred.detach().cpu().view(-1),
        u_hat_unit=u_hat_res.detach().cpu().view(-1),
        u_pred=u_pred.detach().cpu().view(-1),
        b0_star=float(b0_star.item()),
        history=history.to_dict(),
        d_net=d_net,
        local_op=local_op,
    )
