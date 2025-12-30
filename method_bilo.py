"""BiLO (bilevel optimization) solver for the 1D alpha-PDE inverse problem.

Defines BiLOData/BiLOResult, network modules for D and the local operator,
plus bilevel pretrain/finetune training loops.
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
        d_pred: D(x) on x_res (1D).
        u_hat_unit: Unit-source response u_hat for b0=1 on x_res (1D).
        u_pred: Predicted field b0* * u_hat_unit on x_res (1D).
        b0_star: Projected source amplitude.
        history: Loss curves and diagnostics recorded every log_every steps.
            Typical keys: iter, upper, data, reg_smooth, reg_scale, lower, res,
            jump, rgrad, jump_rgrad, b0_star, mean_d, d_snap_iters, d_snapshots.
        d_net: Trained D network (for reuse or inspection).
        local_op: Trained local operator network (for reuse or inspection).
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
        self.use_rff = use_rff
        self.rff_scale = rff_scale
        self.geom_layer = nn.Linear(2, width)
        if use_rff:
            self.geom_layer.weight.requires_grad = False
            if self.geom_layer.bias is not None:
                self.geom_layer.bias.requires_grad = False
        # Condition on D directly.
        # d_embed is intentionally trainable.
        self.d_embed = nn.Linear(1, width, bias=False)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(3)])
        self.output = nn.Linear(width, 1)
        # SiLU (Swish) avoids Tanh's low-frequency bias and saturation.
        self.activation = F.silu

    def forward(
        self, x: torch.Tensor, d: torch.Tensor, z_known: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate u_hat and return the |x-z| feature for jump constraints."""
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
        # Hard enforcement of homogeneous Dirichlet BCs: u(0) = u(1) = 0.
        # This ansatz ensures the network output is always valid at boundaries.
        u = F.softplus(u_raw) * x * (1.0 - x)
        return u, phi_z


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


def _trainable_params(module: nn.Module) -> List[nn.Parameter]:
    """Return list of trainable parameters (excludes frozen RFF layers)."""
    return [p for p in module.parameters() if p.requires_grad]


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute data and regularization terms for the upper-level objective."""
    d_int = d_net(x_int)
    u_hat_int, _ = local_op(x_int, d_int, z_tensor)

    if mode == "field":
        d_field = d_net(x_field)
        u_hat_field, _ = local_op(x_field, d_field, z_tensor)
        b0_star = varpro.project_b0_field(u_hat_field, u_true, field_loss=field_loss)
        data_loss = varpro.field_data_loss(u_hat_field, u_true, b0_star, field_loss=field_loss)
    else:
        integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
        b0_star = varpro.project_b0_ppp(ppp.n_obs, ppp.m_obs, integral_unit)
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute PDE residual and jump penalties for the lower-level objective.

    Args:
        z_idx: Index of the source location on the grid (excluded from residual).
    """
    x_pde = x_res.clone().detach().requires_grad_(True)
    d_pde = d_net(x_pde)
    u_hat_pde, _ = local_op(x_pde, d_pde, z_tensor)
    residual = _alpha_flux_residual(x_pde, d_pde, u_hat_pde, alpha, mu)
    # Exclude the source point from residual loss
    n = residual.shape[0]
    res_loss = (torch.sum(residual ** 2) - residual[z_idx] ** 2) / (n - 1)

    z_probe = z_tensor.clone().detach().requires_grad_(True)
    d_z = d_net(z_probe)
    u_hat_z, phi_z = local_op(z_probe, d_z, z_tensor)
    du_dphi = torch.autograd.grad(
        u_hat_z,
        phi_z,
        grad_outputs=torch.ones_like(u_hat_z),
        create_graph=True,
    )[0]
    jump_res =  d_z * (2.0 * du_dphi) + 1.0
    jump_loss = torch.mean(jump_res ** 2)

    if w_resgrad > 0.0:
        # NOTE: grad_outputs=ones is sufficient here because the residual is
        # evaluated pointwise, so the Jacobian is diagonal in practice.
        grad_jump = torch.autograd.grad(
            jump_res,
            d_z,
            grad_outputs=torch.ones_like(jump_res),
            create_graph=True,
            allow_unused=True,
        )[0]
        # Scale by d_z to convert D-space gradients to log-domain equivalents.
        # Clamp the scaling factor to D_MIN to prevent collapse: without clamping,
        # when D -> 0, the penalty vanishes, creating a positive feedback loop.
        d_scale_jump = d_z.clamp(min=D_MIN)
        jump_rgrad = torch.mean((grad_jump * d_scale_jump) ** 2) if grad_jump is not None else torch.tensor(
            0.0, device=x_res.device, dtype=x_res.dtype
        )
        # Zero out the source point in grad_outputs for resgrad
        grad_outputs = torch.ones_like(residual)
        grad_outputs[z_idx] = 0.0
        grad_res = torch.autograd.grad(
            residual,
            d_pde,
            grad_outputs=grad_outputs,
            create_graph=True,
            allow_unused=True,
        )[0]
        # Scale by d_pde with clamping (same reasoning as jump_rgrad above).
        d_scale_pde = d_pde.clamp(min=D_MIN)
        rgrad = torch.mean((grad_res * d_scale_pde) ** 2) if grad_res is not None else torch.tensor(
            0.0, device=x_res.device, dtype=x_res.dtype
        )
    else:
        rgrad = torch.tensor(0.0, device=x_res.device, dtype=x_res.dtype)
        jump_rgrad = torch.tensor(0.0, device=x_res.device, dtype=x_res.dtype)

    lower_loss = res_loss + w_jump * jump_loss + w_resgrad * (rgrad + jump_rgrad)
    return lower_loss, res_loss, jump_loss, rgrad, jump_rgrad


def fit(data_bundle: BiLOData, cfg: Config, verbose: bool = True) -> BiLOResult:
    """Fit D with bilevel optimization and return reconstructed fields.

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
    
    d_target = d_init_base

    # Create perturbed initialization profile for pretraining target
    d_init_profile = _init_d_profile(
        x_res.view(-1),
        base=d_init_base,
        scale=cfg.d_profile.d_init_pert_scale,
        freq=cfg.d_profile.d_init_pert_freq,
    ).view(-1, 1)

    d_min = getattr(cfg.arch, "d_min", D_MIN)
    rff_scale = getattr(cfg.arch, "rff_scale", 1.0)
    d_net = DNet(
        width=cfg.arch.rff_width,
        use_rff=cfg.arch.use_rff_d,
        rff_scale=rff_scale,
        d_min=d_min,
    ).to(device=device, dtype=dtype)
    local_op = LocalOperator(
        width=cfg.arch.rff_width,
        use_rff=cfg.arch.use_rff_geom,
        rff_scale=rff_scale,
    ).to(device=device, dtype=dtype)

    x_int = torch.linspace(
        domain[0], domain[1], cfg.grid.n_int, device=device, dtype=dtype
    ).view(-1, 1)

    # =========================================================================
    # TRAINING HISTORY
    # =========================================================================
    # BiLO tracks both upper-level (data) and lower-level (physics) losses.
    # Keys:
    #   iter         - iteration number (finetune phase only)
    #   upper        - upper-level loss (data + regularization on D)
    #   data         - data fidelity loss (MSE, RLE, or PPP NLL)
    #   reg_smooth   - smoothness penalty on D (unweighted)
    #   reg_scale    - scale anchor penalty on D (unweighted)
    #   lower        - lower-level loss (physics constraints on local_op)
    #   res          - PDE residual loss (unweighted)
    #   jump         - jump condition loss at source (unweighted)
    #   rgrad        - residual gradient penalty ∂(res)/∂(D) (unweighted)
    #   jump_rgrad   - jump gradient penalty ∂(jump)/∂(D) (unweighted)
    #   b0_star      - projected source amplitude via VarPro
    #   mean_d       - spatial average of D(x)
    #   d_snap_iters - iterations where D snapshots were saved
    #   d_snapshots  - list of D(x) arrays at those iterations
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

    # =========================================================================
    # PRETRAIN PHASE
    # =========================================================================
    # BiLO pretraining has two goals:
    #   1. Train d_net to output the initialization profile (anchoring)
    #   2. Train local_op to satisfy physics for that initial D (supervised)
    #
    # This gives the local_op a "warm start" so it can already solve the PDE
    # before we start updating D based on data. Without this, the local_op
    # would produce garbage u(x) values and the upper-level gradients would
    # be meaningless.
    if cfg.train.pretrain_iters > 0:
        # For initialization, we train d_net to match the perturbed profile d_init_profile
        # And train local_op to match the FDM solution for that perturbed profile.
        d_init_np = d_init_profile.detach().cpu().numpy().reshape(-1)
        u_init_np = physics.fdm_solve_alpha_dirichlet(
            d_init_np,
            cfg.physics.alpha,
            cfg.physics.mu,
            x_res.view(-1).detach().cpu().numpy(),
            1.0,
            cfg.physics.sources,
        )
        u_init_target = torch.tensor(u_init_np, device=device, dtype=dtype).view(-1, 1)
        
        opt_d = torch.optim.Adam(d_net.parameters(), lr=cfg.train.lr_d_pre)
        opt_l = torch.optim.Adam(local_op.parameters(), lr=cfg.train.lr_lower_pre)
        
        for step in range(cfg.train.pretrain_iters):
            opt_d.zero_grad(set_to_none=True)
            # Anchor to d_init_profile (perturbed) during pretraining
            d_pred = d_net(x_res)
            anchor_loss = torch.mean((d_pred - d_init_profile) ** 2)
            
            anchor_loss.backward()
            opt_d.step()

            opt_l.zero_grad(set_to_none=True)
            lower, res_loss, jump_loss, rgrad, jump_rgrad = _calc_physics_loss(
                d_net,
                local_op,
                x_res,
                z_tensor,
                z_idx,
                cfg.physics.alpha,
                cfg.physics.mu,
                cfg.reg.w_jump,
                cfg.reg.w_resgrad,
            )
            d_curr = d_net(x_res).detach()
            u_pred, _ = local_op(x_res, d_curr, z_tensor)
            loss_sup = torch.mean((u_pred - u_init_target) ** 2)
            (lower + loss_sup).backward()

            if verbose and step % cfg.train.log_every == 0:
                with torch.no_grad():
                    mean_d = torch.mean(d_curr).item()
                    pre_total = (anchor_loss + lower + loss_sup).item()
                print(
                    f"[BiLO|pretrain] Iter {step:05d} | Ltot: {pre_total:.3e}\
"
                    f"  Lanchor: {anchor_loss.item():.3e} | Llower: {lower.item():.3e} | Lsup: {loss_sup.item():.3e}\
"
                    f"  Lres: {res_loss.item():.3e} | Ljump: {jump_loss.item():.3e} | "
                    f"Lrgrad: {rgrad.item():.3e} | Ljump_rgrad: {jump_rgrad.item():.3e}\
"
                    f"  ⟨D⟩: {mean_d:.3e}"
                )

            opt_l.step()

    # Final pretrain log
    if cfg.train.pretrain_iters > 0 and verbose:
        with torch.enable_grad():
            d_pred = d_net(x_res)
            anchor_loss = torch.mean((d_pred - d_init_profile) ** 2)
            lower, res_loss, jump_loss, rgrad, jump_rgrad = _calc_physics_loss(
                d_net,
                local_op,
                x_res,
                z_tensor,
                z_idx,
                cfg.physics.alpha,
                cfg.physics.mu,
                cfg.reg.w_jump,
                cfg.reg.w_resgrad,
            )
            d_curr = d_net(x_res).detach()
            u_pred, _ = local_op(x_res, d_curr, z_tensor)
            loss_sup = torch.mean((u_pred - u_init_target) ** 2)
            mean_d = torch.mean(d_curr).item()
            pre_total = (anchor_loss + lower + loss_sup).item()
        print(
            f"[BiLO|pretrain] Iter {cfg.train.pretrain_iters:05d} | Ltot: {pre_total:.3e}\
"
            f"  Lanchor: {anchor_loss.item():.3e} | Llower: {lower.item():.3e} | Lsup: {loss_sup.item():.3e}\
"
            f"  Lres: {res_loss.item():.3e} | Ljump: {jump_loss.item():.3e} | "
            f"Lrgrad: {rgrad.item():.3e} | Ljump_rgrad: {jump_rgrad.item():.3e}\
"
            f"  ⟨D⟩: {mean_d:.3e}"
        )

    # =========================================================================
    # FINETUNE PHASE: Bilevel Optimization
    # =========================================================================
    # BiLO uses bilevel optimization where:
    #   - Upper level: Update d_net to minimize data loss + regularization
    #   - Lower level: Update local_op to satisfy physics for current D
    #
    # The key insight is that these two objectives are decoupled:
    #   - d_net gradients come from data loss (how well does u match observations?)
    #   - local_op gradients come from physics loss (how well does u satisfy PDE?)
    #
    # This avoids the conflicting gradients problem in standard PINN where
    # both networks are updated jointly on a single composite loss.
    d_params = _trainable_params(d_net)
    local_op_params = _trainable_params(local_op)

    # Optimizer setup: Adam (default) or LBFGS
    use_lbfgs = cfg.train.optimizer == "lbfgs"
    optimizer = None
    scheduler = None

    if use_lbfgs:
        optimizer = torch.optim.LBFGS(
            list(d_params) + list(local_op_params),
            lr=cfg.train.lbfgs_lr,
            max_iter=cfg.train.lbfgs_max_iter,
            history_size=10,
            line_search_fn="strong_wolfe",
        )
    else:
        optimizer = torch.optim.Adam([
            {"params": d_params, "lr": cfg.train.lr_d_fine},
            {"params": local_op_params, "lr": cfg.train.lr_lower_fine},
        ])
        if cfg.train.use_scheduler and cfg.train.finetune_iters > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.train.finetune_iters,
                eta_min=min(cfg.train.lr_d_fine, cfg.train.lr_lower_fine) * 0.1,
            )

    # Early stopping state
    best_total: Optional[float] = None
    patience = 0

    # =========================================================================
    # MAIN BILEVEL TRAINING LOOP
    # =========================================================================
    for step in range(cfg.train.finetune_iters + 1):
        if not use_lbfgs:
            optimizer.zero_grad(set_to_none=True)

        # -----------------------------------------------------------------
        # UPPER LEVEL: Compute data loss and gradients for d_net
        # -----------------------------------------------------------------
        # The upper level asks: "Given the current local_op (frozen),
        # how should we update D to better fit the observations?"
        b0_star, data_loss, reg_smooth, reg_scale = _calc_data_loss(
            d_net,
            local_op,
            x_res,
            x_int,
            x_field,
            z_tensor,
            data_bundle.mode,
            u_true,
            ppp,
            cfg.data.field_loss,
            d_target,
            cfg.reg.smoothness_type,
        )
        upper_loss = (
            data_loss + cfg.reg.wreg_smooth * reg_smooth + cfg.reg.wreg_scale * reg_scale
        )

        if not use_lbfgs:
            # Compute upper gradients for d_net only (treat local_op as constant).
            grads_upper = torch.autograd.grad(
                upper_loss,
                d_params,
                create_graph=False,
                allow_unused=True,
            )
            for param, grad in zip(d_params, grads_upper):
                if grad is not None:
                    param.grad = grad

        # -----------------------------------------------------------------
        # LOWER LEVEL: Compute physics loss and gradients for local_op
        # -----------------------------------------------------------------
        # The lower level asks: "Given the current D (frozen), how should
        # we update local_op so that u(x|D) satisfies the PDE?"
        #
        # The physics loss includes:
        #   - res_loss: PDE residual (how well does L[u] - μu = 0?)
        #   - jump_loss: Source jump condition at z
        #   - rgrad: Sensitivity penalty ∂(res)/∂(D) for robustness
        #   - jump_rgrad: Sensitivity penalty ∂(jump)/∂(D)
        lower_loss, res_loss, jump_loss, rgrad, jump_rgrad = _calc_physics_loss(
            d_net,
            local_op,
            x_res,
            z_tensor,
            z_idx,
            cfg.physics.alpha,
            cfg.physics.mu,
            cfg.reg.w_jump,
            cfg.reg.w_resgrad,
        )

        if not use_lbfgs:
            # Compute lower gradients for local_op only (treat d_net as constant).
            grads_lower = torch.autograd.grad(
                lower_loss,
                local_op_params,
                create_graph=False,
                allow_unused=True,
            )
            for param, grad in zip(local_op_params, grads_lower):
                if grad is not None:
                    param.grad = grad

        # -----------------------------------------------------------------
        # LOGGING: Record metrics every log_every steps
        # -----------------------------------------------------------------
        total_loss = upper_loss + lower_loss
        if step % cfg.train.log_every == 0:
            with torch.no_grad():
                d_res = d_net(x_res)
                mean_d = torch.mean(d_res).item()
                u_hat_res, _ = local_op(x_res, d_res, z_tensor)
                if data_bundle.mode == "particles":
                    d_int = d_net(x_int)
                    u_hat_int, _ = local_op(x_int, d_int, z_tensor)
                    integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
                else:
                    integral_unit = torch.trapezoid(u_hat_res.view(-1), x_res.view(-1))
                u_int = (b0_star * integral_unit).item()
                d_snapshot = d_res.detach().cpu().numpy().reshape(-1)
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
                    f"[BiLO|finetune] Iter {step:05d} | Ltot: {total_loss.item():.3e}\
"
                    f"  Upper: {upper_loss.item():.3e} | Ldata({loss_name}): {data_loss.item():.3e} | "
                    f"RegSmooth: {reg_smooth.item():.3e} (eff: {reg_smooth_eff.item():.3e}) | "
                    f"RegScale: {reg_scale.item():.3e} (eff: {reg_scale_eff.item():.3e})\
"
                    f"  Lower: {lower_loss.item():.3e} | Lres: {res_loss.item():.3e} | Ljump: {jump_loss.item():.3e} | "
                    f"Lrgrad: {rgrad.item():.3e} | Ljump_rgrad: {jump_rgrad.item():.3e}\
"
                    f"  b₀*: {b0_star.item():.2f} | ∫û: {integral_unit.item():.3e} | ∫u: {u_int:.3e} | ⟨D⟩: {mean_d:.3e}"
                )

        # -----------------------------------------------------------------
        # EARLY STOPPING: Stop if loss plateaus after burn-in period
        # -----------------------------------------------------------------
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

        # -----------------------------------------------------------------
        # OPTIMIZER STEP: Update both networks simultaneously
        # -----------------------------------------------------------------
        # For Adam: gradients were already computed and assigned above.
        # For LBFGS: we need a closure that recomputes the bilevel gradients.
        #
        # Note: Even though this is "bilevel", we update both networks in
        # one optimizer step. The bilevel structure is in HOW we compute
        # gradients (separate losses), not in alternating updates.
        if step < cfg.train.finetune_iters and not stop_training:
            if use_lbfgs:
                # LBFGS closure must recompute the full bilevel gradient setup
                def _lbfgs_closure() -> torch.Tensor:
                    optimizer.zero_grad(set_to_none=True)
                    # Upper level: data loss -> d_net gradients
                    _, data_loss, reg_smooth, reg_scale = _calc_data_loss(
                        d_net, local_op, x_res, x_int, x_field, z_tensor,
                        data_bundle.mode, u_true, ppp, cfg.data.field_loss,
                        d_target, cfg.reg.smoothness_type,
                    )
                    upper_loss = (
                        data_loss
                        + cfg.reg.wreg_smooth * reg_smooth
                        + cfg.reg.wreg_scale * reg_scale
                    )
                    grads_upper = torch.autograd.grad(
                        upper_loss, d_params, create_graph=False, allow_unused=True,
                    )
                    for param, grad in zip(d_params, grads_upper):
                        if grad is not None:
                            param.grad = grad

                    # Lower level: physics loss -> local_op gradients
                    lower_loss, _, _, _, _ = _calc_physics_loss(
                        d_net, local_op, x_res, z_tensor, z_idx,
                        cfg.physics.alpha, cfg.physics.mu,
                        cfg.reg.w_jump, cfg.reg.w_resgrad,
                    )
                    grads_lower = torch.autograd.grad(
                        lower_loss, local_op_params, create_graph=False, allow_unused=True,
                    )
                    for param, grad in zip(local_op_params, grads_lower):
                        if grad is not None:
                            param.grad = grad
                    return upper_loss + lower_loss
                optimizer.step(_lbfgs_closure)
            else:
                # Standard Adam step (gradients already assigned)
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
        if data_bundle.mode == "field":
            d_field = d_net(x_field)
            u_hat_field, _ = local_op(x_field, d_field, z_tensor)
            b0_star = varpro.project_b0_field(
                u_hat_field, u_true, field_loss=cfg.data.field_loss
            )
        else:
            u_hat_int, _ = local_op(x_int, d_net(x_int), z_tensor)
            integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
            b0_star = varpro.project_b0_ppp(ppp.n_obs, ppp.m_obs, integral_unit)
        u_pred = b0_star * u_hat_res
        d_pred = d_final

    return BiLOResult(
        x_res=x_res.detach().cpu().view(-1),
        d_pred=d_pred.detach().cpu().view(-1),
        u_hat_unit=u_hat_res.detach().cpu().view(-1),
        u_pred=u_pred.detach().cpu().view(-1),
        b0_star=float(b0_star.item()),
        history=history,
        d_net=d_net,
        local_op=local_op,
    )
