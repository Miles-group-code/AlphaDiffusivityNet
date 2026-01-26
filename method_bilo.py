"""BiLO (bilevel optimization) solver for the 1D alpha-PDE inverse problem.

Defines BiLOData/BiLOResult, network modules for D and the local operator,
plus bilevel pretrain/finetune training loops.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from data import PPPData
import physics, varpro
from scale_estimation import estimate_ddi_scale, fit_constant_d
from training_logger import TrainingHistory, format_bilo_pretrain_progress, format_bilo_progress
from DenseNet import DenseNet

try:
    import wandb
except ImportError:
    wandb = None

import matplotlib.pyplot as plt

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
    d_true: Optional[torch.Tensor] = None


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


class LocalOperator(nn.Module):
    """Local operator network for the unit response u_hat(x), conditioned on D."""

    def __init__(
        self,
        u_net: nn.Module,
        bc_type: str = "dirichlet",
    ) -> None:
        super().__init__()
        
        self.bc_type = bc_type.strip().lower()
        if self.bc_type not in {"dirichlet", "neumann"}:
            raise ValueError(f"Unsupported bc_type '{bc_type}'.")
        self.u_net = u_net

    def forward(
        self, x: torch.Tensor, d: torch.Tensor, d_x: torch.Tensor, z_known: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, 1)
        d = d.view(-1, 1)
        d_x = d_x.view(-1, 1)
        phi_z = torch.abs(x - z_known)
        phi_z.requires_grad_(True)
        
        input = torch.cat([x, d, d_x, phi_z], dim=1)
        u = self.u_net(input)
        if self.bc_type == "dirichlet":
            u = u * x * (1.0 - x)
        else:
            u = u
        return u, phi_z


def evaluate_bilo(
    local_op: nn.Module,
    d_net: nn.Module,
    z_tensor: torch.Tensor,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate d, d_x, u, and phi_z at x using the BiLO networks.

    Args:
        local_op: The local operator network (u_hat).
        d_net: The diffusivity network (d).
        z_tensor: Source location tensor.
        x: Input coordinates (must have requires_grad=True or be capable of it).

    Returns:
        d: Diffusivity at x.
        d_x: Gradient of diffusivity at x.
        u: Local operator output at x.
        phi_z: Distance feature |x - z|.
    """
    if not x.requires_grad:
        x.requires_grad_(True)

    d = d_net(x)
    d_x = torch.autograd.grad(d, x, grad_outputs=torch.ones_like(d), create_graph=True)[0]
    u, phi_z = local_op(x, d, d_x, z_tensor)
    return d, d_x, u, phi_z


class DNetVariationWrapper(nn.Module):
    """Temporary wrapper that adds a variation to d_net's output."""
    
    def __init__(self, d_net: nn.Module, variation_func):
        """Initialize wrapper.
        
        Args:
            d_net: Base diffusivity network
            variation_func: Callable(x) -> tensor, adds variation to d_net output
        """
        super().__init__()
        self.d_net = d_net
        self.variation_func = variation_func
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute d_net(x) + variation(x)."""
        d_base = self.d_net(x)
        variation = self.variation_func(x)
        return d_base + variation


def plot_bilo_d_variation(
    solution: "Solution",
    problem: "Problem",
    outdir: str | None = None,
    filename: str = "bilo_d_variation.png",
    show: bool = True,
) -> Optional[plt.Figure]:
    """Visualize BiLO sensitivity to D variations after pretraining.
    
    After pretraining with D_0, shows how u(x) changes with D variations:
    - D_0 + 0.5d (constant shift up)
    - D_0 - 0.5d (constant shift down)
    - D_0 + 0.5d*x (linear increase)
    - D_0 - 0.5d*x (linear decrease)
    
    where d = mean(D_0).
    
    Args:
        solution: BiLO Solution object with d_net and local_op
        problem: Problem object with physics parameters
        outdir: Output directory (if None, don't save)
        show: Whether to show the plot
        
    Returns:
        matplotlib Figure if show=False, None otherwise
    """
    if solution.method != "BILO" or solution.local_op is None:
        return None
    
    # Get device from the networks (they should be on the same device)
    device = next(solution.d_net.parameters()).device
    dtype = next(solution.d_net.parameters()).dtype
    
    x_res = solution.x_res
    x_np = x_res.detach().cpu().numpy().reshape(-1)
    
    # Move x_res to the same device as the networks
    x_res_t = x_res.to(device=device, dtype=dtype).view(-1, 1)
    
    # Prepare z_tensor for local operator (use network device)
    z_tensor = torch.tensor(
        [[problem.source_location]],
        device=device,
        dtype=dtype
    )
    
    # Get D_0 and u_0 using evaluate_bilo
    d_0, d_0_x, u_hat_0, _ = evaluate_bilo(solution.local_op, solution.d_net, z_tensor, x_res_t)
    D_0 = d_0.detach().cpu().numpy().reshape(-1)
    u_0 = solution.b0_star * u_hat_0.view(-1).detach().cpu().numpy()
    
    # Compute mean d
    d_mean = float(np.mean(D_0))
    
    # Create variation functions that take x (tensor) and return variation tensor
    # These will be added to d_net output, so autograd will compute d_x correctly
    variations = {
        "shiftplus": lambda x: 0.5 * d_mean * torch.ones_like(x),
        "shiftminus": lambda x: -0.5 * d_mean * torch.ones_like(x),
        "linplus": lambda x: 0.5 * d_mean * x,
        "linminus": lambda x: -0.5 * d_mean * x,
    }
    
    # Create figure with subplots for each variation
    n_vars = len(variations)
    fig, axes = plt.subplots(n_vars, 2, figsize=(12, 4 * n_vars))
    if n_vars == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (varkey, var_func) in enumerate(variations.items()):
        # Create temporary wrapped d_net with variation
        d_net_var = DNetVariationWrapper(solution.d_net, var_func)
        
        # Use evaluate_bilo to get d, d_x, u - autograd will compute d_x correctly
        d_var, d_var_x, u_hat_var, _ = evaluate_bilo(solution.local_op, d_net_var, z_tensor, x_res_t)
        D_var = d_var.detach().cpu().numpy().reshape(-1)
        u_var = solution.b0_star * u_hat_var.view(-1).detach().cpu().numpy()
        
        # Also compute FDM solution for comparison
        try:
            u_fdm_var = physics.fdm_solve_alpha(
                D_var,
                problem.alpha,
                problem.mu,
                x_np,
                solution.b0_star,
                (problem.source_location,),
                bc_type=problem.bc_type,
            )
        except ValueError as e:
            u_fdm_var = None
        
        # Plot u(x)
        ax_u = axes[idx, 0]
        ax_u.plot(x_np, u_var, label=f"LocalOp ({varkey})", linewidth=2)
        if u_fdm_var is not None:
            ax_u.plot(x_np, u_fdm_var, "--", label="FDM", linewidth=1.5, alpha=0.7)
        ax_u.plot(x_np, u_0, "k:", label="u_0", linewidth=1.5, alpha=0.5)
        
        ax_u.set_xlabel("x")
        ax_u.set_ylabel("u(x)")
        ax_u.set_title(f"u(x) variation: {varkey}")
        ax_u.legend()
        ax_u.grid(True, alpha=0.3)
        
        # Plot D(x)
        ax_d = axes[idx, 1]
        ax_d.plot(x_np, D_0, "k-", label="D_0", linewidth=2)
        ax_d.plot(x_np, D_var, label=f"D_var ({varkey})", linewidth=1.5)
        ax_d.set_xlabel("x")
        ax_d.set_ylabel("D(x)")
        ax_d.set_title(f"D(x) variation: {varkey}")
        ax_d.legend()
        ax_d.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, filename), dpi=150)
        plt.close(fig)
        return None
    elif show:
        plt.show()
        return None
    else:
        return fig

def _resolve_weights_path(path: Optional[str], outdir: str) -> Optional[str]:
    """Resolve relative paths against the run output directory."""
    if path is None:
        return None
    if not str(path).strip():
        raise ValueError("Weight path must be a non-empty string.")
    if os.path.isabs(path):
        return path
    return os.path.join(outdir, path)


def _load_bilo_weights(path: str, d_net: nn.Module, local_op: nn.Module, device: torch.device) -> None:
    """Load BiLO weights from disk into the provided modules."""
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "d_net" not in payload or "local_op" not in payload:
        raise ValueError(f"Invalid BiLO weights file at '{path}'. Expected keys: d_net, local_op.")
    d_net.load_state_dict(payload["d_net"], strict=True)
    local_op.load_state_dict(payload["local_op"], strict=True)


def _save_bilo_weights(path: str, d_net: nn.Module, local_op: nn.Module) -> None:
    """Save BiLO weights for D-net and local operator."""
    outdir = os.path.dirname(path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    torch.save(
        {
            "d_net": d_net.state_dict(),
            "local_op": local_op.state_dict(),
        },
        path,
    )


def _init_d_profile(
    x: torch.Tensor, base: float, scale: float, freq: float
) -> torch.Tensor:
    """Build a sinusoidal D initialization on the grid."""
    if scale >= 1.0:
        raise ValueError("pert_scale must be < 1 to keep D_init positive.")
    return base + scale * torch.sin(2.0 * torch.pi * freq * x)


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
) -> Dict[str, torch.Tensor]:
    """Compute data and regularization terms for the upper-level objective.

    Returns a dict for readability:
        b0_star, data_loss, reg_smooth, reg_scale
    """
    
    if mode == "field":
        d_field, d_field_x, u_hat_field, _ = evaluate_bilo(local_op, d_net, z_tensor, x_field)
        b0_star = varpro.get_b0_field(
            u_hat_field, u_true, field_loss=field_loss, b0_fixed_value=b0_fixed_value
        )
        data_loss = varpro.field_data_loss(u_hat_field, u_true, b0_star, field_loss=field_loss)
    else:
        d_int, d_int_x, u_hat_int, _ = evaluate_bilo(local_op, d_net, z_tensor, x_int)
        integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
        b0_star = varpro.get_b0_ppp(ppp.n_obs, ppp.m_obs, integral_unit, b0_fixed_value=b0_fixed_value)
        d_data, d_data_x, u_hat_data, _ = evaluate_bilo(local_op, d_net, z_tensor, ppp.x_particles)
        data_loss = varpro.ppp_nll(u_hat_data.view(-1), b0_star, ppp.m_obs, integral_unit)

    x_reg = x_res.clone().detach().requires_grad_(True)
    d_reg = d_net(x_reg)
    if smoothness_type == "tv":
        reg_smooth = physics.tv_smoothness_d(x_reg, d_reg)
    else:
        reg_smooth = physics.h1_smoothness_d(x_reg, d_reg)

    reg_scale = physics.scale_anchor(d_reg, d_target)
    return {
        "b0_star": b0_star,
        "data_loss": data_loss,
        "reg_smooth": reg_smooth,
        "reg_scale": reg_scale,
    }


def _calc_physics_loss(
    d_net: nn.Module,
    local_op: nn.Module,
    x_res: torch.Tensor,
    z_tensor: torch.Tensor,
    z_idx: int,
    alpha: float,
    mu: float,
    loss_weights: Dict[str, float],
    bc_type: str,
    domain: Tuple[float, float],
) -> Dict[str, torch.Tensor]:
    """Compute PDE residual, jump, and BC penalties for the lower-level objective.

    Returns a dict for readability:
        lower_loss, res_loss, jump_loss, bc_loss, rgrad, jump_rgrad, bc_grad_loss
    """
    w_jump = loss_weights["w_jump"]
    w_resgrad = loss_weights["w_resgrad"]
    w_bc = loss_weights["w_bc"]

    # x_pde is xres remove z_idx
    x_pde = torch.cat([x_res[:z_idx], x_res[z_idx+1:]]).clone().detach().requires_grad_(True)
    
    d_pde, d_x, u_hat_pde, _ = evaluate_bilo(local_op, d_net, z_tensor, x_pde)
    residual = _alpha_flux_residual(x_pde, d_pde, u_hat_pde, alpha, mu)
    
    n = residual.shape[0]
    res_loss = torch.mean(residual ** 2)

    z_probe = z_tensor.clone().detach().requires_grad_(True)
    d_z, d_z_x, u_hat_z, phi_z = evaluate_bilo(local_op, d_net, z_tensor, z_probe)
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
        grad_jump_d = torch.autograd.grad(jump_res, d_z, grad_outputs=torch.ones_like(jump_res), create_graph=True, allow_unused=True)[0]
        grad_jump_dx = torch.autograd.grad(jump_res, d_z_x, grad_outputs=torch.ones_like(jump_res), create_graph=True, allow_unused=True)[0]
        jump_rgrad = torch.mean(grad_jump_d ** 2 + grad_jump_dx ** 2)

        grad_res_d = torch.autograd.grad(residual, d_pde, grad_outputs=torch.ones_like(residual), create_graph=True, allow_unused=True)[0]
        grad_res_dx = torch.autograd.grad(residual, d_x, grad_outputs=torch.ones_like(residual), create_graph=True, allow_unused=True)[0]
        rgrad = torch.mean(grad_res_d ** 2 + grad_res_dx ** 2)
    else:
        rgrad = torch.tensor(0.0, device=x_res.device, dtype=x_res.dtype)
        jump_rgrad = torch.tensor(0.0, device=x_res.device, dtype=x_res.dtype)

    lower_loss = res_loss + w_jump * jump_loss + w_resgrad * (rgrad + jump_rgrad)
    if bc_type == "neumann":
        lower_loss = lower_loss + w_bc * (bc_loss) + w_resgrad * bc_grad_loss
    return {
        "lower_loss": lower_loss,
        "res_loss": res_loss,
        "jump_loss": jump_loss,
        "bc_loss": bc_loss,
        "rgrad": rgrad,
        "jump_rgrad": jump_rgrad,
        "bc_grad_loss": bc_grad_loss,
    }


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
    d_true = data_bundle.d_true
    if d_true is not None:
        d_true = d_true.to(device=device, dtype=dtype).view(-1, 1)

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
    lower_tol = getattr(cfg.train, "lower_tol", None)

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
        # Fall back to the ground-truth profile mean when DDI is disabled.
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
        print(f"[BiLO] DDI scale: {d_ddi:.3e}")
        print(f"[BiLO] Scalar fit scale: {d_scale:.3e}")

    d_target = d_scale

    # =========================================================================
    # MODEL INITIALIZATION
    # =========================================================================
    d_init_profile = _init_d_profile(x_res.view(-1), base=d_scale, scale=pert_scale, freq=pert_freq).view(-1, 1)

    fix_endpoint = getattr(cfg.arch, "fix_endpoint", False)
    if fix_endpoint:
        lambda_transform = lambda x, u: d_target + u * x * (1.0 - x)
    else:
        lambda_transform = lambda x, u: F.softplus(u) + D_MIN

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
        # lambda_transform=lambda x, u: torch.exp(u),
        lambda_transform=lambda_transform,
    ).to(device=device, dtype=dtype)
    


    u_net = DenseNet(
        input_dim=4,
        output_dim=1,
        act='silu',
        width=u_net_width,
        depth=u_net_depth,
        arch=u_net_arch,
        fourier=use_rff,
        sigma=4.0)

    local_op = LocalOperator(u_net, bc_type=bc_type).to(device=device, dtype=dtype)

    load_path = _resolve_weights_path(getattr(cfg.train, "bilo_load_path", None), cfg.run.outdir)
    if load_path is not None:
        if verbose:
            print(f"[BiLO] Loading weights from: {load_path}")
        _load_bilo_weights(load_path, d_net, local_op, device)

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

        try:
            for step in range(pretrain_iters):
                opt_d.zero_grad(set_to_none=True)
                d_pred = d_net(x_res)
                anchor_loss = torch.mean((d_pred - d_init_profile) ** 2)
                anchor_loss.backward()
                opt_d.step()

                opt_l.zero_grad(set_to_none=True)
                phys = _calc_physics_loss(
                    d_net, local_op, x_res, z_tensor, z_idx, alpha, mu,
                    {"w_jump": w_jump, "w_resgrad": w_resgrad, "w_bc": w_bc},
                    bc_type, domain
                )
                lower = phys["lower_loss"]
                d_curr, d_curr_x, u_pred, _ = evaluate_bilo(local_op, d_net, z_tensor, x_res)
                loss_sup = torch.mean((u_pred - u_init_target) ** 2)
                (lower + loss_sup).backward()

                if verbose and step % log_every == 0:
                    with torch.no_grad():
                        mean_d = torch.mean(d_curr).item()
                        pre_total = (anchor_loss + lower + loss_sup).item()
                    print(format_bilo_pretrain_progress(
                        step=step,
                        metrics={
                            "total": pre_total,
                            "anchor": anchor_loss.item(),
                            "lower": lower.item(),
                            "sup": loss_sup.item(),
                            "res": phys["res_loss"].item(),
                            "jump": phys["jump_loss"].item(),
                            "bc": phys["bc_loss"].item(),
                            "bc_grad": phys["bc_grad_loss"].item(),
                            "rgrad": phys["rgrad"].item(),
                            "jump_rgrad": phys["jump_rgrad"].item(),
                            "mean_d": mean_d,
                        },
                        bc_type=bc_type,
                    ))

                opt_l.step()
        except KeyboardInterrupt:
            if verbose:
                print(f"\n[BiLO] Pretraining interrupted by user at step {step}. Continuing to finetune...")

        # Final pretrain log
        if verbose:
            with torch.enable_grad():
                d_pred = d_net(x_res)
                anchor_loss = torch.mean((d_pred - d_init_profile) ** 2)
                phys = _calc_physics_loss(
                    d_net, local_op, x_res, z_tensor, z_idx, alpha, mu,
                    {"w_jump": w_jump, "w_resgrad": w_resgrad, "w_bc": w_bc},
                    bc_type, domain
                )
                lower = phys["lower_loss"]
                d_curr, d_curr_x, u_pred, _ = evaluate_bilo(local_op, d_net, z_tensor, x_res)
                loss_sup = torch.mean((u_pred - u_init_target) ** 2)
                mean_d = torch.mean(d_curr).item()
                pre_total = (anchor_loss + lower + loss_sup).item()
            print(format_bilo_pretrain_progress(
                step=pretrain_iters,
                metrics={
                    "total": pre_total,
                    "anchor": anchor_loss.item(),
                    "lower": lower.item(),
                    "sup": loss_sup.item(),
                    "res": phys["res_loss"].item(),
                    "jump": phys["jump_loss"].item(),
                    "bc": phys["bc_loss"].item(),
                    "bc_grad": phys["bc_grad_loss"].item(),
                    "rgrad": phys["rgrad"].item(),
                    "jump_rgrad": phys["jump_rgrad"].item(),
                    "mean_d": mean_d,
                },
                bc_type=bc_type,
            ))

        # ---------------------------------------------------------------------
        # VISUALIZATION (after pretraining)
        # ---------------------------------------------------------------------
        try:
            solution = SimpleNamespace(
                method="BILO", d_net=d_net, local_op=local_op, x_res=x_res, b0_star=1.0
            )
            problem = SimpleNamespace(
                source_location=float(sources[0]), alpha=float(alpha), mu=float(mu), bc_type=str(bc_type)
            )

            if cfg.wandb.enabled and wandb is not None and wandb.run is not None:
                fig_var = plot_bilo_d_variation(
                    solution, problem, outdir=None, filename="bilo_d_variation_pretrain.png", show=False
                )
                if fig_var is not None:
                    wandb.log({"bilo_d_variation_pretrain": wandb.Image(fig_var)}, step=0, commit=True)
                    if verbose:
                        print("  ✓ Logged bilo_d_variation_pretrain to wandb")
                    plt.close(fig_var)
            else:
                plot_bilo_d_variation(
                    solution, problem, outdir=cfg.run.outdir, filename="bilo_d_variation_pretrain.png", show=False
                )
                if verbose:
                    print(f"  ✓ Saved bilo_d_variation_pretrain to: {os.path.join(cfg.run.outdir, 'bilo_d_variation_pretrain.png')}")
        except Exception as e:
            if verbose:
                import traceback
                traceback.print_exc()
                print(f"Warning: Failed to generate BiLO pretrain D variation plot: {e}")

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
        d_optimizer = torch.optim.Adam(d_params, lr=lr_d_fine)
        local_op_optimizer = torch.optim.Adam(local_op_params, lr=lr_lower_fine)
        scheduler = None
        d_scheduler = None
        local_op_scheduler = None
        if use_scheduler and finetune_iters > 0:
            d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=finetune_iters, eta_min=lr_d_fine * 0.1)
            local_op_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(local_op_optimizer, T_max=finetune_iters, eta_min=lr_lower_fine * 0.1)

    # Early stopping state
    best_total: Optional[float] = None
    patience = 0

    # =========================================================================
    # MAIN BILEVEL TRAINING LOOP
    # =========================================================================
    try:
        for step in range(finetune_iters + 1):
            if not use_lbfgs:
                d_optimizer.zero_grad(set_to_none=True)
                local_op_optimizer.zero_grad(set_to_none=True)

            # Upper level: data loss -> d_net gradients
            data_terms = _calc_data_loss(
                d_net, local_op, x_res, x_int, x_field, z_tensor,
                mode, u_true, ppp, field_loss_type, d_target, smoothness_type,
                b0_fixed_value=b0_fixed_value
            )
            b0_star = data_terms["b0_star"]
            data_loss = data_terms["data_loss"]
            reg_smooth = data_terms["reg_smooth"]
            reg_scale = data_terms["reg_scale"]
            upper_loss = data_loss + wreg_smooth * reg_smooth + wreg_scale * reg_scale

            # Lower level: physics loss -> local_op gradients
            phys = _calc_physics_loss(
                d_net, local_op, x_res, z_tensor, z_idx, alpha, mu,
                {"w_jump": w_jump, "w_resgrad": w_resgrad, "w_bc": w_bc},
                bc_type, domain
            )
            lower_loss = phys["lower_loss"]
            res_loss = phys["res_loss"]
            jump_loss = phys["jump_loss"]
            bc_loss = phys["bc_loss"]
            rgrad = phys["rgrad"]
            jump_rgrad = phys["jump_rgrad"]
            bc_grad_loss = phys["bc_grad_loss"]

            update_both = lower_tol is None or lower_loss.item() <= lower_tol
            if not use_lbfgs:
                grads_lower = torch.autograd.grad(lower_loss, local_op_params, create_graph=False, allow_unused=True)
                for param, grad in zip(local_op_params, grads_lower):
                    if grad is not None:
                        param.grad = grad
                if update_both:
                    grads_upper = torch.autograd.grad(upper_loss, d_params, create_graph=False, allow_unused=True)
                    for param, grad in zip(d_params, grads_upper):
                        if grad is not None:
                            param.grad = grad

            # -----------------------------------------------------------------
            # LOGGING
            # -----------------------------------------------------------------
            total_loss = upper_loss + lower_loss
            if step % log_every == 0:
                
                d_res, d_res_x, u_hat_res, _ = evaluate_bilo(local_op, d_net, z_tensor, x_res)
                mean_d = torch.mean(d_res).item()
                if mode == "particles":
                    d_int, d_int_x, u_hat_int, _ = evaluate_bilo(local_op, d_net, z_tensor, x_int)
                    integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
                else:
                    integral_unit = torch.trapezoid(u_hat_res.view(-1), x_res.view(-1))
                d_snapshot = d_res.detach().cpu().numpy().reshape(-1)

                # Validation Metrics
                metrics_extra = {}
                
                # 1. D Error (if ground truth available)
                if d_true is not None:
                    d_err = torch.abs(d_res - d_true)
                    d_err_l2 = torch.sqrt(torch.mean(d_err ** 2)).item()
                    d_err_linf = torch.max(d_err).item()
                    metrics_extra["d_err_l2"] = d_err_l2
                    metrics_extra["d_err_linf"] = d_err_linf
                
                # 2. Solver Consistency (BiLO vs FDM with predicted D)
                # We compute FDM solution using the current d_pred and b0_star
                # BiLO prediction: u_pred = b0_star * u_hat_res
                b0_val = b0_star.item()
                u_bilo = b0_val * u_hat_res.view(-1)
                
                try:
                    u_fdm_np = physics.fdm_solve_alpha(
                        d=d_snapshot,
                        alpha=alpha,
                        mu=mu,
                        x=x_res.detach().cpu().numpy().reshape(-1),
                        b0=b0_val,
                        sources=sources,
                        bc_type=bc_type
                    )
                    u_fdm = torch.from_numpy(u_fdm_np).to(device=device, dtype=dtype)
                    
                    if field_loss_type == "rle":
                        u_fdm_err = torch.abs(u_bilo - u_fdm) / (u_fdm + 1e-6)
                    else:
                        u_fdm_err = torch.abs(u_bilo - u_fdm)

                    u_fdm_err = torch.mean(u_fdm_err ** 2).item()
                    
                    metrics_extra["u_fdm_err"] = u_fdm_err
                except Exception as e:
                    # Fallback if FDM fails (e.g. numerical instability)
                    pass

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
                    **metrics_extra
                )
                history.log_snapshot(step, d_snapshot)

                if verbose:
                    loss_name = field_loss_type if mode == "field" else "ppp"
                    
                    # Merge extra metrics into the display dict
                    display_metrics = {
                        "total": total_loss.item(),
                        "upper": upper_loss.item(),
                        "data": data_loss.item(),
                        "reg_smooth": reg_smooth.item(),
                        "reg_scale": reg_scale.item(),
                        "lower": lower_loss.item(),
                        "res": res_loss.item(),
                        "jump": jump_loss.item(),
                        "bc": bc_loss.item(),
                        "rgrad": rgrad.item(),
                        "jump_rgrad": jump_rgrad.item(),
                        "b0_star": b0_star.item(),
                        "integral_unit": integral_unit.item(),
                        "mean_d": mean_d,
                    }
                    display_metrics.update(metrics_extra)
                    
                    print(format_bilo_progress(
                        step=step,
                        phase="finetune",
                        metrics=display_metrics,
                        weights={
                            "wreg_smooth": wreg_smooth,
                            "wreg_scale": wreg_scale,
                        },
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
                        data_terms = _calc_data_loss(
                            d_net, local_op, x_res, x_int, x_field, z_tensor,
                            mode, u_true, ppp, field_loss_type, d_target, smoothness_type,
                            b0_fixed_value=b0_fixed_value
                        )
                        data_loss = data_terms["data_loss"]
                        reg_smooth = data_terms["reg_smooth"]
                        reg_scale = data_terms["reg_scale"]
                        upper_loss = data_loss + wreg_smooth * reg_smooth + wreg_scale * reg_scale
                        grads_upper = torch.autograd.grad(upper_loss, d_params, create_graph=False, allow_unused=True)
                        for param, grad in zip(d_params, grads_upper):
                            if grad is not None:
                                param.grad = grad

                        phys = _calc_physics_loss(
                            d_net, local_op, x_res, z_tensor, z_idx, alpha, mu,
                            {"w_jump": w_jump, "w_resgrad": w_resgrad, "w_bc": w_bc},
                            bc_type, domain
                        )
                        lower_loss = phys["lower_loss"]
                        update_both = lower_tol is None or lower_loss.item() <= lower_tol
                        grads_lower = torch.autograd.grad(lower_loss, local_op_params, create_graph=False, allow_unused=True)
                        for param, grad in zip(local_op_params, grads_lower):
                            if grad is not None:
                                param.grad = grad
                        if update_both:
                            grads_upper = torch.autograd.grad(upper_loss, d_params, create_graph=False, allow_unused=True)
                            for param, grad in zip(d_params, grads_upper):
                                if grad is not None:
                                    param.grad = grad
                        return upper_loss + lower_loss
                    optimizer.step(_lbfgs_closure)
                else:
                    if update_both:
                        d_optimizer.step()
                    local_op_optimizer.step()
                    if d_scheduler is not None:
                        d_scheduler.step()
                        local_op_scheduler.step()
            else:
                if stop_training and verbose:
                    print(f"[BiLO] Early stopping triggered at step {step}.")
                break
    except KeyboardInterrupt:
        if verbose:
            print(f"\n[BiLO] Training interrupted by user at step {step}. Continuing to post-processing...")

    # =========================================================================
    # FINAL RESULT EXTRACTION
    # =========================================================================
    
    d_final, d_final_x, u_hat_res, _ = evaluate_bilo(local_op, d_net, z_tensor, x_res)
    if mode == "field":
        d_field, d_field_x, u_hat_field, _ = evaluate_bilo(local_op, d_net, z_tensor, x_field)
        b0_star = varpro.get_b0_field(
            u_hat_field, u_true, field_loss=field_loss_type, b0_fixed_value=b0_fixed_value
        )
    else:
        raise NotImplementedError("PPP mode not implemented yet")
        u_hat_int, _ = local_op(x_int, d_net(x_int), z_tensor)
        integral_unit = torch.trapezoid(u_hat_int.view(-1), x_int.view(-1))
        b0_star = varpro.get_b0_ppp(ppp.n_obs, ppp.m_obs, integral_unit, b0_fixed_value=b0_fixed_value)
    u_pred = b0_star * u_hat_res
    d_pred = d_final

    save_path = _resolve_weights_path(getattr(cfg.train, "bilo_save_path", None), cfg.run.outdir)
    if save_path is not None:
        _save_bilo_weights(save_path, d_net, local_op)
        if verbose:
            print(f"[BiLO] Saved weights to: {save_path}")

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
