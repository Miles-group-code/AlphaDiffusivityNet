"""PINN solver for the 1D alpha-PDE inverse problem.

Defines PINNData/PINNResult, network modules for D and u, and the
single-level training loop that couples data and physics losses.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from DenseNet import DenseNet
from config import Config
from data import PPPData
import physics, varpro
from scale_estimation import estimate_ddi_scale, fit_constant_d
from training_logger import (
    TrainingHistory,
    format_pinn_progress,
    format_pinn_pretrain_progress,
)

# =============================================================================
# D PARAMETERIZATION: Softplus + offset
# =============================================================================
# We use softplus for positivity: D = softplus(raw) + D_min
# Softplus has mild gradient suppression but avoids catastrophic gradient death.
# =============================================================================

D_MIN = 1e-6


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
    x_field: Optional[torch.Tensor] = None
    u_true: Optional[torch.Tensor] = None
    ppp: Optional[PPPData] = None
    d_true: Optional[torch.Tensor] = None


@dataclass
class PINNResult:
    """Outputs and training history for PINN fitting.

    Returned tensors are detached and on CPU for convenience.
    """

    x_res: torch.Tensor
    d_pred: torch.Tensor
    u_hat_unit: torch.Tensor
    u_pred: torch.Tensor
    b0_star: float
    history: Dict[str, List[float]]
    d_net: Optional[nn.Module] = None
    u_net: Optional[nn.Module] = None


class LocalOperator(nn.Module):
    """Local operator network for u(x), conditioned on z."""

    def __init__(
        self,
        u_net: nn.Module,
        bc_type: str = "dirichlet",
    ) -> None:
        super().__init__()
        self.u_net = u_net
        self.bc_type = bc_type.strip().lower()
        if self.bc_type not in {"dirichlet", "neumann"}:
            raise ValueError(f"Unsupported bc_type '{bc_type}'.")

    def forward(self, x: torch.Tensor, z_known: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, 1)
        phi_z = torch.abs(x - z_known)
        if phi_z.is_leaf and not phi_z.requires_grad:
            phi_z.requires_grad_(True)
        # Input: x and distance to source
        net_in = torch.cat([x, phi_z], dim=1)
        u_raw = self.u_net(net_in)
        
        u_pos = F.softplus(u_raw)
        
        if self.bc_type == "dirichlet":
            u = u_pos * x * (1.0 - x)
        else:
            u = u_pos
        return u, phi_z


def _alpha_flux_residual(
    x: torch.Tensor,
    d: torch.Tensor,
    u: torch.Tensor,
    alpha: float,
    mu: float,
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
    u_net: nn.Module,
    z_tensor: torch.Tensor,
    domain: Tuple[float, float],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Penalize boundary derivatives for Neumann BCs: u'(x_min)=u'(x_max)=0."""
    x0 = torch.tensor([[domain[0]]], device=device, dtype=dtype, requires_grad=True)
    x1 = torch.tensor([[domain[1]]], device=device, dtype=dtype, requires_grad=True)

    u0, _ = u_net(x0, z_tensor)
    u1, _ = u_net(x1, z_tensor)

    u0_x = torch.autograd.grad(
        u0, x0, grad_outputs=torch.ones_like(u0), create_graph=True
    )[0]
    u1_x = torch.autograd.grad(
        u1, x1, grad_outputs=torch.ones_like(u1), create_graph=True
    )[0]

    return torch.mean(u0_x ** 2 + u1_x ** 2)


def _compute_physics_losses(
    d_net: nn.Module,
    u_net: nn.Module,
    x_res: torch.Tensor,
    z_tensor: torch.Tensor,
    z_idx: int,
    alpha: float,
    mu: float,
    bc_type: str,
    domain: Tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute PDE residual, jump, and boundary-condition losses for PINN."""
    x_pde = x_res.clone().detach().requires_grad_(True)
    d = d_net(x_pde)
    u_hat, _ = u_net(x_pde, z_tensor)
    residual = _alpha_flux_residual(x_pde, d, u_hat, alpha, mu)
    n = residual.shape[0]
    res_loss = (torch.sum(residual ** 2) - residual[z_idx] ** 2) / (n - 1)

    z_probe = z_tensor.clone().detach().requires_grad_(True)
    d_z = d_net(z_probe)
    u_z, phi_z = u_net(z_probe, z_tensor)
    du_dphi = torch.autograd.grad(
        u_z, phi_z, grad_outputs=torch.ones_like(u_z), create_graph=True
    )[0]
    jump_res = d_z * (2.0 * du_dphi) + 1.0
    jump_loss = torch.mean(jump_res ** 2)

    bc_loss = torch.tensor(0.0, device=x_res.device, dtype=x_res.dtype)
    if bc_type == "neumann":
        bc_loss = _compute_bc_loss_neumann(
            u_net, z_tensor, domain, x_res.device, x_res.dtype
        )

    return res_loss, jump_loss, bc_loss


def fit(data_bundle: PINNData, cfg: Config, verbose: bool = True) -> PINNResult:
    """Fit D with a physics-informed neural network.

    Args:
        data_bundle: PINNData specifying observations and grids.
        cfg: Full configuration (physics/data/grid/train/reg/arch/run).
        verbose: Print progress during training.

    Returns:
        PINNResult with predictions on the solver grid (x_res) and training history.
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

    # Regularization
    w_data = cfg.reg.w_data
    w_phys = cfg.reg.w_phys
    w_jump = cfg.reg.w_jump
    w_bc = cfg.reg.w_bc
    wreg_smooth, wreg_scale = cfg.reg.wreg_smooth, cfg.reg.wreg_scale
    wreg_d_neumann = cfg.reg.wreg_d_neumann
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
    use_rff = cfg.arch.use_rff
    
    d_net_arch = cfg.arch.d_net_arch
    d_net_depth = cfg.arch.d_net_depth
    d_net_width = cfg.arch.d_net_width
    d_net_rff_scale = cfg.arch.d_net_rff_scale
    siren_omega0 = cfg.arch.siren_omega0

    u_net_arch = cfg.arch.u_net_arch
    u_net_depth = cfg.arch.u_net_depth
    u_net_width = cfg.arch.u_net_width

    # Grid
    n_int = cfg.grid.n_int

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================
    if len(sources) != 1:
        raise NotImplementedError("PINN currently supports a single source.")

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
        d_ddi = float(cfg.d_profile.params[0])

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
        print(f"[PINN] DDI scale: {d_ddi:.3e}")
        print(f"[PINN] Scalar fit scale: {d_scale:.3e}")

    d_target = d_scale

    # =========================================================================
    # MODEL INITIALIZATION
    # =========================================================================
    # Determine lambda_transform based on d_transform option
    d_transform = cfg.arch.d_transform
    if d_transform == "fix_end":
        lambda_transform = lambda x, u: d_target + u * x * (1.0 - x)
    elif d_transform == "soft_plus":
        d_min = getattr(cfg.arch, "d_min", D_MIN)
        lambda_transform = lambda x, u: F.softplus(u) + d_min
    elif d_transform == "exp":
        lambda_transform = lambda x, u: torch.exp(u)
    else:
        raise ValueError(f"Unknown d_transform: {d_transform}. Must be one of: 'fix_end', 'soft_plus', 'exp'")

    d_net = DenseNet(
        input_dim=1,
        output_dim=1,
        act='silu',
        width=d_net_width,
        depth=d_net_depth,
        arch=d_net_arch,
        fourier=use_rff,
        sigma=d_net_rff_scale,
        omega_0=siren_omega0,
        lambda_transform=lambda_transform,
    ).to(device=device, dtype=dtype)

    # u_net handles x and phi_z, so input_dim=2
    u_net_inner = DenseNet(
        input_dim=2,
        output_dim=1,
        act='silu',
        width=u_net_width,
        depth=u_net_depth,
        arch=u_net_arch,
        fourier=use_rff,
        sigma=4.0
    )
    
    u_net = LocalOperator(
        u_net=u_net_inner,
        bc_type=bc_type
    ).to(device=device, dtype=dtype)

    # =========================================================================
    # TRAINING HISTORY
    # =========================================================================
    history = TrainingHistory.for_pinn()

    # =========================================================================
    # PRETRAIN PHASE
    # =========================================================================
    if pretrain_iters > 0:
        optim_pre = torch.optim.Adam([
            {"params": d_net.parameters(), "lr": lr_d_pre},
            {"params": u_net.parameters(), "lr": lr_lower_pre},
        ])
        try:
            for step in range(pretrain_iters + 1):
                optim_pre.zero_grad(set_to_none=True)
                res_loss, jump_loss, bc_loss = _compute_physics_losses(
                    d_net, u_net, x_res, z_tensor, z_idx, alpha, mu, bc_type, domain
                )
                phys_loss = w_phys * res_loss + w_jump * jump_loss
                if bc_type == "neumann":
                    phys_loss = phys_loss + w_bc * bc_loss
                d_pred = d_net(x_res)
                anchor_loss = physics.scale_anchor(d_pred, d_target)
                pre_loss = phys_loss + anchor_loss

                if verbose and step % log_every == 0:
                    with torch.no_grad():
                        mean_d = torch.mean(d_pred).item()
                    print(format_pinn_pretrain_progress(
                        step=step,
                        total=pre_loss.item(),
                        phys=phys_loss.item(),
                        anchor=anchor_loss.item(),
                        res=res_loss.item(),
                        jump=jump_loss.item(),
                        bc=bc_loss.item(),
                        mean_d=mean_d,
                        bc_type=bc_type,
                    ))

                if step < pretrain_iters:
                    pre_loss.backward()
                    optim_pre.step()
        except KeyboardInterrupt:
            if verbose:
                print(f"\n[PINN] Pretraining interrupted by user at step {step}. Continuing to finetune...")

    # =========================================================================
    # FINETUNE OPTIMIZER SETUP
    # =========================================================================
    if use_lbfgs:
        optimizer = torch.optim.LBFGS(
            list(d_net.parameters()) + list(u_net.parameters()),
            lr=lbfgs_lr,
            max_iter=lbfgs_max_iter,
            history_size=10,
            line_search_fn="strong_wolfe",
        )
        scheduler = None
    else:
        optimizer = torch.optim.Adam([
            {"params": d_net.parameters(), "lr": lr_d_fine},
            {"params": u_net.parameters(), "lr": lr_lower_fine},
        ])
        scheduler = None
        if use_scheduler and finetune_iters > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=finetune_iters, eta_min=lr_d_fine * 0.1
            )

    # Early stopping state
    best_total: Optional[float] = None
    patience = 0

    # =========================================================================
    # LOSS COMPUTATION
    # =========================================================================
    def _compute_losses() -> Tuple[torch.Tensor, ...]:
        integral_unit = None
        if mode == "field":
            u_hat_field, _ = u_net(x_field, z_tensor)
            b0_star = varpro.get_b0_field(
                u_hat_field, u_true, field_loss=field_loss_type, b0_fixed_value=b0_fixed_value
            )
            data_loss = varpro.field_data_loss(
                u_hat_field, u_true, b0_star, field_loss=field_loss_type
            )
        else:
            u_hat_int, _ = u_net(x_int, z_tensor)
            integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
            b0_star = varpro.get_b0_ppp(
                ppp.n_obs, ppp.m_obs, integral_unit, b0_fixed_value=b0_fixed_value
            )
            u_hat_obs, _ = u_net(ppp.x_particles, z_tensor)
            data_loss = varpro.ppp_nll(u_hat_obs.view(-1), b0_star, ppp.m_obs, integral_unit)

        res_loss, jump_loss, bc_loss = _compute_physics_losses(
            d_net, u_net, x_res, z_tensor, z_idx, alpha, mu, bc_type, domain
        )
        phys_loss = w_phys * res_loss + w_jump * jump_loss
        if bc_type == "neumann":
            phys_loss = phys_loss + w_bc * bc_loss

        x_reg = x_res.clone().detach().requires_grad_(True)
        d_reg = d_net(x_reg)
        if smoothness_type == "tv":
            reg_smooth = physics.tv_smoothness_d(x_reg, d_reg)
        else:
            reg_smooth = physics.h1_smoothness_d(x_reg, d_reg)

        reg_scale = physics.scale_anchor(d_reg, d_target)

        # D Neumann regularization
        x0 = torch.tensor([[domain[0]]], device=x_res.device, dtype=x_res.dtype, requires_grad=True)
        x1 = torch.tensor([[domain[1]]], device=x_res.device, dtype=x_res.dtype, requires_grad=True)
        d0 = d_net(x0)
        d1 = d_net(x1)
        d0_x = torch.autograd.grad(d0, x0, grad_outputs=torch.ones_like(d0), create_graph=True)[0]
        d1_x = torch.autograd.grad(d1, x1, grad_outputs=torch.ones_like(d1), create_graph=True)[0]
        d_neumann = torch.mean(d0_x ** 2 + d1_x ** 2)

        total_loss = (
            w_data * data_loss
            + phys_loss
            + wreg_smooth * reg_smooth
            + wreg_scale * reg_scale
            + wreg_d_neumann * d_neumann
        )
        return (
            total_loss, data_loss, phys_loss, res_loss, jump_loss, bc_loss,
            reg_smooth, reg_scale, b0_star, d_reg, integral_unit, d_neumann
        )

    # =========================================================================
    # MAIN FINETUNE LOOP
    # =========================================================================
    try:
        for step in range(finetune_iters + 1):
            if not use_lbfgs:
                optimizer.zero_grad(set_to_none=True)

            # Forward pass
            (
                total_loss, data_loss, phys_loss, res_loss, jump_loss, bc_loss,
                reg_smooth, reg_scale, b0_star, d_reg, integral_unit, d_neumann
            ) = _compute_losses()

            # -----------------------------------------------------------------
            # LOGGING
            # -----------------------------------------------------------------
            if step % log_every == 0:
                with torch.no_grad():
                    d_vals = d_reg.detach()
                    mean_d = torch.mean(d_vals).item()
                    d_snapshot = d_vals.detach().cpu().numpy().reshape(-1)
                    if mode == "field":
                        u_hat_res, _ = u_net(x_res, z_tensor)
                        integral_unit = torch.trapezoid(u_hat_res.view(-1), x_res.view(-1))

                history.log(
                    step=step,
                    total=total_loss.item(),
                    data=data_loss.item(),
                    phys=phys_loss.item(),
                    res=res_loss.item(),
                    jump=jump_loss.item(),
                    bc=bc_loss.item(),
                    reg_smooth=reg_smooth.item(),
                    reg_scale=reg_scale.item(),
                    b0_star=b0_star.item(),
                    mean_d=mean_d,
                    d_neumann=d_neumann.item(),
                )
                history.log_snapshot(step, d_snapshot)

                if verbose:
                    loss_name = field_loss_type if mode == "field" else "ppp"
                    print(format_pinn_progress(
                        step=step,
                        phase="finetune",
                        total=total_loss.item(),
                        data=data_loss.item(),
                        phys=phys_loss.item(),
                        res=res_loss.item(),
                        jump=jump_loss.item(),
                        bc=bc_loss.item(),
                        reg_smooth=reg_smooth.item(),
                        reg_scale=reg_scale.item(),
                        wreg_smooth=wreg_smooth,
                        wreg_scale=wreg_scale,
                        b0_star=b0_star.item(),
                        integral_unit=integral_unit.item(),
                        mean_d=mean_d,
                        loss_name=loss_name,
                        bc_type=bc_type,
                        d_neumann=d_neumann.item(),
                        wreg_d_neumann=wreg_d_neumann,
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
                        loss_value = _compute_losses()[0]
                        loss_value.backward()
                        return loss_value
                    optimizer.step(_lbfgs_closure)
                else:
                    total_loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
            else:
                if stop_training and verbose:
                    print(f"[PINN] Early stopping triggered at step {step}.")
                break
    except KeyboardInterrupt:
        if verbose:
            print(f"\n[PINN] Training interrupted by user at step {step}. Continuing to post-processing...")

    # =========================================================================
    # FINAL RESULT EXTRACTION
    # =========================================================================
    with torch.no_grad():
        d_final = d_net(x_res)
        u_hat_res, _ = u_net(x_res, z_tensor)
        if mode == "field":
            u_hat_field, _ = u_net(x_field, z_tensor)
            b0_star = varpro.get_b0_field(
                u_hat_field, u_true, field_loss=field_loss_type, b0_fixed_value=b0_fixed_value
            )
        else:
            u_hat_int, _ = u_net(x_int, z_tensor)
            integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
            b0_star = varpro.get_b0_ppp(
                ppp.n_obs, ppp.m_obs, integral_unit, b0_fixed_value=b0_fixed_value
            )
        u_pred = b0_star * u_hat_res
        d_pred = d_final

    return PINNResult(
        x_res=x_res.detach().cpu().view(-1),
        d_pred=d_pred.detach().cpu().view(-1),
        u_hat_unit=u_hat_res.detach().cpu().view(-1),
        u_pred=u_pred.detach().cpu().view(-1),
        b0_star=float(b0_star.item()),
        history=history.to_dict(),
        d_net=d_net,
        u_net=u_net,
    )
