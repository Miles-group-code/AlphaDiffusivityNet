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

from config import Config
from data import PPPData
import physics, varpro
from scale_estimation import estimate_ddi_scale, fit_constant_d

# =============================================================================
# D PARAMETERIZATION: Softplus + offset
# =============================================================================
# We use softplus for positivity: D = softplus(raw) + D_min
#
# HISTORY:
# - v3.0.0: Switched from log(D) to softplus + offset
# - Tried LeakyReLU + clamp(min=0), but clamp killed gradients when raw < 0
# - Reverted to softplus (v3.0.1): smooth gradients everywhere
#
# TRADE-OFF:
# - Softplus has mild gradient suppression: ∂D/∂raw = sigmoid(raw) ≈ D for small D
# - But avoids catastrophic gradient death from clamp operations
# - Combined with DDI initialization and Adam optimizer, works well in practice
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
    x_field: Optional[torch.Tensor] = None  # Field observation grid (field mode)
    u_true: Optional[torch.Tensor] = None
    ppp: Optional[PPPData] = None


@dataclass
class PINNResult:
    """Outputs and training history for PINN fitting.

    Returned tensors are detached and on CPU for convenience.

    Fields:
        x_res: Solver grid (1D).
        d_pred: D(x) on x_res (1D).
        u_hat_unit: Unit-source response u_hat for b0=1 on x_res (1D).
        u_pred: Predicted field b0* * u_hat_unit on x_res (1D).
        b0_star: Projected source amplitude.
        history: Loss curves and diagnostics recorded every log_every steps.
            Typical keys: iter, total, data, phys, res, jump, reg_smooth,
            reg_scale, b0_star, mean_d, d_snap_iters, d_snapshots.
    """

    x_res: torch.Tensor
    d_pred: torch.Tensor
    u_hat_unit: torch.Tensor
    u_pred: torch.Tensor
    b0_star: float
    history: Dict[str, List[float]]


class DNet(nn.Module):
    """RFF-embedded MLP that parameterizes D(x) with softplus + offset.

    Uses softplus parameterization: D = softplus(raw) + D_min
    Softplus is smooth, always positive, and has well-behaved gradients everywhere.
    """

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
        # SiLU (Swish) activation: avoids Tanh's low-frequency bias and
        # saturation issues while maintaining smooth gradients for PDE learning.
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate D(x) at given coordinates.

        Uses softplus for positivity: D = softplus(raw) + D_min
        Softplus has smooth gradients everywhere (no dead regions).
        """
        x = x.view(-1, 1)
        if self.use_rff:
            feat = torch.sin(2.0 * torch.pi * self.rff_scale * self.embed(x))
        else:
            feat = self.embed(x)
        raw = self.net(feat)
        return F.softplus(raw) + self.d_min


class LocalOperator(nn.Module):
    """Local operator network for the unit response u_hat(x)."""

    def __init__(self, width: int = 128, use_rff: bool = True, rff_scale: float = 1.0) -> None:
        super().__init__()
        # NOTE: Unlike BiLO, the PINN u-network does NOT condition on D.
        # This is standard PINN practice: u_net and d_net are trained jointly.
        self.use_rff = use_rff
        self.rff_scale = rff_scale
        self.geom_layer = nn.Linear(2, width)
        if use_rff:
            self.geom_layer.weight.requires_grad = False
            if self.geom_layer.bias is not None:
                self.geom_layer.bias.requires_grad = False
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(3)])
        self.output = nn.Linear(width, 1)
        # SiLU (Swish) avoids Tanh's low-frequency bias and saturation.
        self.activation = F.silu

    def forward(self, x: torch.Tensor, z_known: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate u_hat and return the |x-z| feature for jump constraints."""
        x = x.view(-1, 1)
        phi_z = torch.abs(x - z_known)
        if phi_z.is_leaf and not phi_z.requires_grad:
            phi_z.requires_grad_(True)
        geom_in = torch.cat([x, phi_z], dim=1)
        geom_lin = self.geom_layer(geom_in)
        if self.use_rff:
            h = self.activation(torch.sin(2.0 * torch.pi * self.rff_scale * geom_lin))
        else:
            h = self.activation(geom_lin)
        for layer in self.hidden:
            h = self.activation(layer(h))
        u_raw = self.output(h)
        # Hard enforcement of homogeneous Dirichlet BCs: u(0) = u(1) = 0.
        # This ansatz ensures the network output is always valid at boundaries.
        u = F.softplus(u_raw) * x * (1.0 - x)
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


def _compute_physics_losses(
    d_net: nn.Module,
    u_net: nn.Module,
    x_res: torch.Tensor,
    z_tensor: torch.Tensor,
    z_idx: int,
    alpha: float,
    mu: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PDE residual and jump losses for the PINN objective.

    Returns raw (unweighted) losses. Caller applies weights independently.

    Args:
        z_idx: Index of the source location on the grid (excluded from residual).
    """
    x_pde = x_res.clone().detach().requires_grad_(True)
    d = d_net(x_pde)
    u_hat, _ = u_net(x_pde, z_tensor)
    residual = _alpha_flux_residual(x_pde, d, u_hat, alpha, mu)
    # Exclude the source point from residual loss
    n = residual.shape[0]
    res_loss = (torch.sum(residual ** 2) - residual[z_idx] ** 2) / (n - 1)

    z_probe = z_tensor.clone().detach().requires_grad_(True)
    d_z = d_net(z_probe)
    u_z, phi_z = u_net(z_probe, z_tensor)
    du_dphi = torch.autograd.grad(
        u_z,
        phi_z,
        grad_outputs=torch.ones_like(u_z),
        create_graph=True,
    )[0]
    jump_res = d_z * (2.0 * du_dphi) + 1.0
    jump_loss = torch.mean(jump_res ** 2)

    return res_loss, jump_loss


def fit(data_bundle: PINNData, cfg: Config, verbose: bool = True) -> PINNResult:
    """Fit D with a physics-informed neural network.

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
    # Since z is exactly on the aligned grid, find the single index to exclude from residual.
    z_idx = int(torch.argmin(torch.abs(x_res - z_tensor)).item())

    x_int = torch.linspace(
        cfg.physics.domain[0], cfg.physics.domain[1], cfg.grid.n_int, device=device, dtype=dtype
    ).view(-1, 1)

    if cfg.d_profile.use_ddi:
        d_ddi = estimate_ddi_scale(
            mu=cfg.physics.mu,
            z=cfg.physics.sources[0],
            x_particles=ppp.x_particles if ppp is not None else None,
            u_field=u_true if u_true is not None else None,
            x_grid=x_field.view(-1),
            d_min=cfg.d_profile.ddi_d_min,
            d_max=cfg.d_profile.ddi_d_max,
        )
    else:
        d_ddi = 1.0

    if cfg.train.scalar_fit_iters > 0:
        d_scale = fit_constant_d(
            x=x_res.view(-1),
            alpha=cfg.physics.alpha,
            mu=cfg.physics.mu,
            sources=cfg.physics.sources,
            u_true=u_true if u_true is not None else None,
            ppp=ppp if ppp is not None else None,
            x_field=x_field.view(-1) if u_true is not None else None,
            x_int=x_int.view(-1) if ppp is not None else None,
            d_init=d_ddi,
            max_iters=cfg.train.scalar_fit_iters,
            field_loss=cfg.data.field_loss,
            verbose=verbose,
        )
    else:
        d_scale = d_ddi

    if verbose:
        print(f"[PINN] DDI scale: {d_ddi:.3e}")
        print(f"[PINN] Scalar fit scale: {d_scale:.3e}")
    
    d_target = d_scale

    d_min = getattr(cfg.arch, "d_min", D_MIN)
    rff_scale = getattr(cfg.arch, "rff_scale", 1.0)
    d_net = DNet(
        width=cfg.arch.rff_width,
        use_rff=cfg.arch.use_rff_d,
        rff_scale=rff_scale,
        d_min=d_min,
    ).to(device=device, dtype=dtype)
    u_net = LocalOperator(
        width=cfg.arch.rff_width,
        use_rff=cfg.arch.use_rff_geom,
        rff_scale=rff_scale,
    ).to(device=device, dtype=dtype)

    # =========================================================================
    # TRAINING HISTORY
    # =========================================================================
    # We track metrics at each logged iteration for diagnostics and plotting.
    # Keys:
    #   iter         - iteration number (finetune phase only)
    #   total        - total loss (data + physics + regularization)
    #   data         - data fidelity loss (MSE, RLE, or PPP NLL)
    #   phys         - weighted physics loss (w_phys*res + w_jump*jump)
    #   res          - PDE residual loss (unweighted)
    #   jump         - jump condition loss at source (unweighted)
    #   reg_smooth   - smoothness penalty on D (H1 or TV, unweighted)
    #   reg_scale    - scale anchor penalty (unweighted)
    #   b0_star      - projected source amplitude via VarPro
    #   mean_d       - spatial average of D(x)
    #   d_snap_iters - iterations where D snapshots were saved
    #   d_snapshots  - list of D(x) arrays at those iterations
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

    # =========================================================================
    # PRETRAIN PHASE (optional)
    # =========================================================================
    # The pretrain phase trains the networks on physics constraints only,
    # without data. This helps the u-network learn to satisfy the PDE before
    # we try to fit data, preventing the optimizer from finding shortcuts
    # that fit data but violate physics.
    #
    # Loss = physics_loss + anchor_loss (anchor D to initialization)
    if cfg.train.pretrain_iters > 0:
        optim_pre = torch.optim.Adam(
            [
                {"params": d_net.parameters(), "lr": cfg.train.lr_d_pre},
                {"params": u_net.parameters(), "lr": cfg.train.lr_lower_pre},
            ]
        )
        for step in range(cfg.train.pretrain_iters + 1):
            optim_pre.zero_grad(set_to_none=True)
            res_loss, jump_loss = _compute_physics_losses(
                d_net,
                u_net,
                x_res,
                z_tensor,
                z_idx,
                cfg.physics.alpha,
                cfg.physics.mu,
            )
            # w_phys weights residual, w_jump weights jump (independent)
            phys_loss = cfg.reg.w_phys * res_loss + cfg.reg.w_jump * jump_loss
            d_pred = d_net(x_res)
            anchor_loss = physics.scale_anchor(d_pred, d_target)
            pre_loss = phys_loss + anchor_loss

            if verbose and step % cfg.train.log_every == 0:
                with torch.no_grad():
                    mean_d = torch.mean(d_pred).item()
                print(
                    f"[PINN|pretrain] Iter {step:05d} | Ltot: {pre_loss.item():.3e}\
"
                    f"  Lphys: {phys_loss.item():.3e} | Lanchor: {anchor_loss.item():.3e}\
"
                    f"  Lres: {res_loss.item():.3e} | Ljump: {jump_loss.item():.3e}\
"
                    f"  ⟨D⟩: {mean_d:.3e}"
                )

            if step < cfg.train.pretrain_iters:
                pre_loss.backward()
                optim_pre.step()

    # =========================================================================
    # FINETUNE OPTIMIZER SETUP
    # =========================================================================
    # The finetune phase jointly optimizes both networks to minimize:
    #   total_loss = w_data*data_loss + physics_loss + regularization
    #
    # Adam is the default, but LBFGS can help in ill-conditioned cases.
    # Unlike DTO, PINN typically works well with Adam because the physics
    # loss provides strong gradients (equation error, not output error).
    use_lbfgs = cfg.train.optimizer == "lbfgs"
    optimizer = None
    scheduler = None

    if use_lbfgs:
        optimizer = torch.optim.LBFGS(
            list(d_net.parameters()) + list(u_net.parameters()),
            lr=cfg.train.lbfgs_lr,
            max_iter=cfg.train.lbfgs_max_iter,
            history_size=10,              # How many past gradients to remember
            line_search_fn="strong_wolfe",  # Robust line search for step size
        )
    else:
        optimizer = torch.optim.Adam(
            [
                {"params": d_net.parameters(), "lr": cfg.train.lr_d_fine},
                {"params": u_net.parameters(), "lr": cfg.train.lr_lower_fine},
            ]
        )
        if cfg.train.use_scheduler and cfg.train.finetune_iters > 0:
            # Cosine annealing: smoothly reduce LR from initial to 10% over training
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.train.finetune_iters,
                eta_min=cfg.train.lr_d_fine * 0.1,
            )

    # Early stopping state
    best_total: Optional[float] = None
    patience = 0

    # =========================================================================
    # LOSS COMPUTATION (shared by logging and LBFGS closure)
    # =========================================================================
    # This inner function computes all loss components in one forward pass.
    # PINN loss structure:
    #   total = w_data*data_loss + w_phys*res_loss + w_jump*jump_loss
    #         + wreg_smooth*smoothness + wreg_scale*scale_anchor
    #
    # Returns: (total, data, phys, res, jump, reg_smooth, reg_scale, b0_star, d_reg, integral_unit)
    def _compute_losses() -> Tuple[torch.Tensor, ...]:
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

        res_loss, jump_loss = _compute_physics_losses(
            d_net,
            u_net,
            x_res,
            z_tensor,
            z_idx,
            cfg.physics.alpha,
            cfg.physics.mu,
        )
        # w_phys weights residual, w_jump weights jump (independent)
        phys_loss = cfg.reg.w_phys * res_loss + cfg.reg.w_jump * jump_loss

        x_reg = x_res.clone().detach().requires_grad_(True)
        d_reg = d_net(x_reg)
        if cfg.reg.smoothness_type == "tv":
            reg_smooth = physics.tv_smoothness_d(x_reg, d_reg)
        else:
            reg_smooth = physics.h1_smoothness_d(x_reg, d_reg)

        reg_scale = physics.scale_anchor(d_reg, d_target)

        total_loss = (
            cfg.reg.w_data * data_loss
            + phys_loss
            + cfg.reg.wreg_smooth * reg_smooth
            + cfg.reg.wreg_scale * reg_scale
        )
        return (
            total_loss,
            data_loss,
            phys_loss,
            res_loss,
            jump_loss,
            reg_smooth,
            reg_scale,
            b0_star,
            d_reg,
            integral_unit,
        )

    # =========================================================================
    # MAIN FINETUNE LOOP
    # =========================================================================
    # PINN jointly trains d_net and u_net to minimize the composite loss:
    #   total = w_data*data_loss + physics_loss + regularization
    #
    # Unlike BiLO's bilevel approach, PINN trains both networks simultaneously
    # with gradients flowing through both. This is simpler but can lead to
    # conflicts between fitting data and satisfying physics.
    for step in range(cfg.train.finetune_iters + 1):
        # For Adam: zero gradients before forward pass
        # For LBFGS: gradients are zeroed inside the closure
        if not use_lbfgs:
            optimizer.zero_grad(set_to_none=True)

        # Forward pass: compute all losses
        (
            total_loss,
            data_loss,
            phys_loss,
            res_loss,
            jump_loss,
            reg_smooth,
            reg_scale,
            b0_star,
            d_reg,
            integral_unit,
        ) = _compute_losses()

        # ---------------------------------------------------------------------
        # LOGGING: Record metrics every log_every steps
        # ---------------------------------------------------------------------
        if step % cfg.train.log_every == 0:
            with torch.no_grad():
                d_vals = d_reg.detach()
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
                    f"[PINN|finetune] Iter {step:05d} | Ltot: {total_loss.item():.3e}\
"
                    f"  Ldata({loss_name}): {data_loss.item():.3e} | Lphys: {phys_loss.item():.3e} | "
                    f"RegSmooth: {reg_smooth.item():.3e} (eff: {reg_smooth_eff.item():.3e}) | "
                    f"RegScale: {reg_scale.item():.3e} (eff: {reg_scale_eff.item():.3e})\
"
                    f"  Lres: {res_loss.item():.3e} | Ljump: {jump_loss.item():.3e}\
"
                    f"  b₀*: {b0_star.item():.2f} | ∫û: {integral_unit.item():.3e} | ∫u: {u_int:.3e} | ⟨D⟩: {mean_d:.3e}"
                )

        # ---------------------------------------------------------------------
        # EARLY STOPPING: Stop if loss plateaus after burn-in period
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # OPTIMIZER STEP: Update network parameters
        # ---------------------------------------------------------------------
        if step < cfg.train.finetune_iters and not stop_training:
            if use_lbfgs:
                # LBFGS closure: recomputes loss (may be called multiple times)
                def _lbfgs_closure() -> torch.Tensor:
                    optimizer.zero_grad(set_to_none=True)
                    loss_value = _compute_losses()[0]
                    loss_value.backward()
                    return loss_value
                optimizer.step(_lbfgs_closure)
            else:
                # Standard Adam step
                total_loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        else:
            if stop_training and verbose:
                print(f"[PINN] Early stopping triggered at step {step}.")
            break

    # =========================================================================
    # FINAL RESULT EXTRACTION
    # =========================================================================
    with torch.no_grad():
        d_final = d_net(x_res)
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
        d_pred = d_final

    return PINNResult(
        x_res=x_res.detach().cpu().view(-1),
        d_pred=d_pred.detach().cpu().view(-1),
        u_hat_unit=u_hat_res.detach().cpu().view(-1),
        u_pred=u_pred.detach().cpu().view(-1),
        b0_star=float(b0_star.item()),
        history=history,
    )
