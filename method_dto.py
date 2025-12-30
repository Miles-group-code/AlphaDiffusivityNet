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
from data import PPPData, estimate_ddi_scale
import physics, varpro


# =============================================================================
# D PARAMETERIZATION: DIRECT + PROJECTION
# =============================================================================
# We parameterize D(x) directly rather than through softplus or exp(logD).
#
# RATIONALE: Both softplus and logD parameterizations have Jacobians proportional
# to D itself, which causes catastrophic gradient suppression for small D values:
#
#   - logD: D = exp(θ) → ∂D/∂θ = D
#   - softplus: D = softplus(θ) → ∂D/∂θ = sigmoid(θ) ≈ D (for small D)
#
# When D = 0.01, the gradient w.r.t. θ is suppressed by 100x. Combined with the
# already-flat Output Error landscape (see dto_issues.md), this makes optimization
# nearly impossible—LBFGS reports param_change → 0 within ~20 steps.
#
# SOLUTION: Use direct parameterization with projection:
#   - Store θ = D - D_min as the parameter
#   - Forward: D = θ + D_min (∂D/∂θ = 1, no suppression!)
#   - After step: clamp θ ≥ 0 to enforce D ≥ D_min
#
# ALTERNATIVE: If projection causes issues, squared parameterization is smoother:
#   D = θ² + D_min → ∂D/∂θ = 2θ = 2√(D - D_min)
# This has √D suppression (milder than D), but direct + projection is preferred.
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
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble tridiagonal coefficients for the alpha-PDE discretization.

    h is now a vector of step sizes: h[i] = x[i+1] - x[i].

    NOTE: We use the harmonic mean of D^alpha at cell interfaces rather than
    (arithmetic_mean(D))^alpha. This preserves flux continuity across interfaces
    when D is discontinuous (e.g., step profiles). For smooth D, both approaches
    give second-order accuracy, but the harmonic mean is more physically correct
    for heterogeneous media.

    The flux form of the PDE is: J = D^alpha * grad(D^(1-alpha) * u)
    At interface i+1/2, we need a_half = D^alpha evaluated at the interface.
    Harmonic mean ensures: 1/a_half = 0.5 * (1/D_i^alpha + 1/D_{i+1}^alpha)
    """
    g = d ** (1.0 - alpha)

    # Harmonic mean of D^alpha at cell interfaces for flux continuity
    # harmonic_mean(a, b) = 2*a*b / (a + b)
    d_alpha_left = d[:-1] ** alpha
    d_alpha_right = d[1:] ** alpha
    a_half = 2.0 * d_alpha_left * d_alpha_right / (d_alpha_left + d_alpha_right + eps) / h

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

        # Direct parameterization: θ = D - D_min
        # Forward pass: D = θ + D_min (Jacobian = 1, no gradient suppression)
        # Constraint: θ ≥ 0 enforced by projection after each optimizer step
        self.theta_int = nn.Parameter(d_init[1:-1] - d_min)

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
        """Build full D array from interior parameters.

        Uses direct parameterization: D = θ + D_min (Jacobian = 1).
        Boundary values are mirrored from adjacent interior nodes.
        """
        d = torch.empty(self.n, device=self.x_res.device, dtype=self.x_res.dtype)
        # Direct parameterization: no softplus, no gradient suppression
        d[1:-1] = self.theta_int + self.d_min
        d[0] = d[1]
        d[-1] = d[-2]
        return d

    def solve_unit_u_hat(self, d_full: torch.Tensor) -> torch.Tensor:
        """Solve for the unit-source response u_hat given D.

        Uses a cached unit-source RHS vector to avoid per-iteration rebuilds.
        """
        lower, diag, upper = _build_tridiag_alpha(d_full, self.alpha, self.mu, self.h)
        u_int = _thomas_solve(lower, diag, upper, self.rhs_unit)

        # Build full solution with boundary conditions (u=0 at boundaries)
        u = torch.zeros(self.n, device=self.x_res.device, dtype=d_full.dtype)
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
        # x_int and its interpolation are no longer needed for the integral
        # because we will integrate directly on x_res.
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
    ).to(device=device, dtype=dtype)

    # =========================================================================
    # OPTIMIZER SETUP
    # =========================================================================
    # DTO supports two optimizers:
    #   - Adam: First-order, robust to noisy gradients, good default
    #   - LBFGS: Quasi-Newton, uses curvature info, better for ill-conditioned
    #            problems like DTO where the loss landscape has flat valleys
    #
    # LBFGS is recommended for DTO because:
    #   1. Few parameters (~n_res grid values) fit in LBFGS memory
    #   2. The output-error loss landscape is ill-conditioned (see dto_issues.md)
    #   3. LBFGS can take large steps along flat valleys where Adam stalls
    use_lbfgs = cfg.train.optimizer == "lbfgs"
    optimizer = None
    scheduler = None

    if use_lbfgs:
        optimizer = torch.optim.LBFGS(
            [model.theta_int],
            lr=cfg.train.lbfgs_lr,
            max_iter=cfg.train.lbfgs_max_iter,
            history_size=10,              # How many past gradients to remember
            line_search_fn="strong_wolfe", # Use strong_wolfe to handle curvature
        )
    else:
        optimizer = torch.optim.Adam([model.theta_int], lr=cfg.train.lr_d_fine)
        if cfg.train.use_scheduler and cfg.train.finetune_iters > 0:
            # Cosine annealing: smoothly reduce LR from initial to 10% over training
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.train.finetune_iters,
                eta_min=cfg.train.lr_d_fine * 0.1,
            )

    # =========================================================================
    # TRAINING HISTORY
    # =========================================================================
    # We track metrics at each logged iteration for diagnostics and plotting.
    # Keys:
    #   iter         - iteration number
    #   total        - total loss (data + weighted regularization)
    #   data         - data fidelity loss (MSE, RLE, or PPP NLL depending on mode)
    #   reg_smooth   - smoothness penalty on D (H1 or TV, unweighted)
    #   reg_scale    - scale anchor penalty (deviation from DDI estimate, unweighted)
    #   b0_star      - projected source amplitude via VarPro
    #   mean_d       - spatial average of D(x) for monitoring scale drift
    #   d_snap_iters - iterations where D snapshots were saved
    #   d_snapshots  - list of D(x) arrays at those iterations (for evolution plots)
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

    # Early stopping state: track best loss and patience counter
    best_total: Optional[float] = None
    patience = 0

    # =========================================================================
    # LOSS COMPUTATION (shared by logging and LBFGS closure)
    # =========================================================================
    # This inner function computes all loss components in one forward pass.
    # It's defined here so LBFGS can call it inside its closure (LBFGS may
    # evaluate the loss multiple times per step during line search).
    #
    # Returns: (d_full, u_hat_unit, b0_star, data_loss, reg_smooth, reg_scale, total_loss)
    #   - d_full: Full D(x) array on solver grid (with boundary mirroring)
    #   - u_hat_unit: Unit-source solution (b0=1) from Thomas algorithm
    #   - b0_star: Optimal amplitude from VarPro projection
    #   - data_loss: Data fidelity (MSE/RLE for field, NLL for particles)
    #   - reg_smooth: Smoothness penalty (H1 or TV on log D)
    #   - reg_scale: Scale anchor penalty (deviation from d_target)
    #   - total_loss: Weighted sum for optimization
    def _compute_losses() -> Tuple[torch.Tensor, ...]:
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
            # Integrate directly on x_res to avoid aliasing errors from grid mismatch
            integral_unit = torch.trapezoid(u_hat_unit.view(-1), x_res.view(-1))
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

        return d_full, u_hat_unit, b0_star, data_loss, reg_smooth, reg_scale, total_loss

    # =========================================================================
    # MAIN TRAINING LOOP
    # =========================================================================
    # DTO uses single-level optimization: we directly optimize the grid values
    # of D to minimize data loss + regularization. The forward pass solves the
    # PDE via Thomas algorithm, projects b0 via VarPro, and computes losses.
    for step in range(cfg.train.finetune_iters + 1):
        # For Adam: zero gradients before forward pass
        # For LBFGS: gradients are zeroed inside the closure
        if not use_lbfgs:
            optimizer.zero_grad(set_to_none=True)

        # Forward pass: compute D, solve PDE, project b0, compute losses
        d_full, u_hat_unit, b0_star, data_loss, reg_smooth, reg_scale, total_loss = (
            _compute_losses()
        )

        # ---------------------------------------------------------------------
        # LOGGING: Record metrics every log_every steps
        # ---------------------------------------------------------------------
        if step % cfg.train.log_every == 0:
            with torch.no_grad():
                mean_d = torch.mean(d_full).item()
                integral_unit = torch.trapezoid(u_hat_unit.view(-1), x_res.view(-1))
                u_int = (b0_star * integral_unit).item()
                d_snapshot = d_full.detach().cpu().numpy()

            # Append to history for later analysis/plotting
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

        # ---------------------------------------------------------------------
        # EARLY STOPPING: Stop if loss plateaus after burn-in period
        # ---------------------------------------------------------------------
        # We wait until early_burnin steps before checking, then track whether
        # the loss improves by at least early_tol (relative). If no improvement
        # for early_patience consecutive checks, we stop training.
        stop_training = False
        if step >= cfg.train.early_burnin:
            total_val = total_loss.item()
            if best_total is None:
                best_total = total_val
                patience = 0
            else:
                # Compute relative improvement
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
        # OPTIMIZER STEP: Update parameters
        # ---------------------------------------------------------------------
        if step < cfg.train.finetune_iters and not stop_training:
            if use_lbfgs:
                # LBFGS requires a closure that recomputes the loss (it may call
                # this multiple times during line search). The closure must:
                # 1. Zero gradients, 2. Compute loss, 3. Call backward, 4. Return loss
                closure_calls = 0
                def _lbfgs_closure() -> torch.Tensor:
                    nonlocal closure_calls
                    closure_calls += 1
                    optimizer.zero_grad(set_to_none=True)
                    loss_value = _compute_losses()[-1]
                    loss_value.backward()
                    return loss_value.detach()

                # Track whether LBFGS actually updated parameters
                theta_before = model.theta_int.data.clone()
                optimizer.step(_lbfgs_closure)

                # PROJECT: Enforce θ ≥ 0 to maintain D ≥ D_min
                # This is the key fix for gradient suppression—direct parameterization
                # with projection preserves full gradient magnitude (∂D/∂θ = 1).
                with torch.no_grad():
                    model.theta_int.data.clamp_(min=0)

                theta_after = model.theta_int.data
                param_change = (theta_after - theta_before).abs().max().item()
                grad_norm = model.theta_int.grad.norm().item() if model.theta_int.grad is not None else 0.0
                if step % cfg.train.log_every == 0 and verbose:
                    print(f"  [LBFGS DEBUG] grad_norm: {grad_norm:.3e} | param_change: {param_change:.3e} | closure_calls: {closure_calls}")
            else:
                # Standard Adam step: backward then step
                total_loss.backward()
                optimizer.step()

                # PROJECT: Enforce θ ≥ 0 to maintain D ≥ D_min
                with torch.no_grad():
                    model.theta_int.data.clamp_(min=0)

                if scheduler is not None:
                    scheduler.step()
        else:
            if stop_training and verbose:
                print(f"[DTO] Early stopping triggered at step {step}.")
            break

    # =========================================================================
    # FINAL RESULT EXTRACTION
    # =========================================================================
    # After training, compute final predictions with no gradients needed.
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
            # Consistent integration on x_res
            integral_unit = torch.trapezoid(u_hat_unit.view(-1), x_res.view(-1))
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
