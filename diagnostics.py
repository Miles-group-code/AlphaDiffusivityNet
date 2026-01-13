"""Plotting and metric utilities for solver outputs.

Includes metric computation, training-history plots, and method-comparison
visualizations used by example scripts and notebooks.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
import torch

import physics
import varpro

if TYPE_CHECKING:
    from interface import Problem, Solution


def _compute_metrics_core(
    x: np.ndarray,
    d_true: np.ndarray,
    d_pred: np.ndarray,
    *,
    u_true: Optional[np.ndarray] = None,
    u_pred: Optional[np.ndarray] = None,
    b0_star: Optional[float] = None,
    b_true: Optional[float] = None,
) -> Dict[str, float]:
    eps = 1e-12
    denom = float(np.trapz(np.ones_like(x), x))

    d_err = float(np.trapz((d_pred - d_true) ** 2, x))
    d_true_err = float(np.trapz(d_true ** 2, x))

    logd_pred = np.log(np.clip(d_pred, eps, None))
    logd_true = np.log(np.clip(d_true, eps, None))
    logd_err = float(np.trapz((logd_pred - logd_true) ** 2, x))
    logd_true_err = float(np.trapz(logd_true ** 2, x))

    d_true_int = float(np.trapz(d_true, x))
    d_pred_int = float(np.trapz(d_pred, x))
    d_true_norm = d_true / max(d_true_int, eps)
    d_pred_norm = d_pred / max(d_pred_int, eps)
    d_shape_err = float(np.trapz((d_pred_norm - d_true_norm) ** 2, x))
    d_shape_true_err = float(np.trapz(d_true_norm ** 2, x))

    # Pearson Correlation
    d_flat = d_pred.flatten()
    t_flat = d_true.flatten()
    if np.std(d_flat) > eps and np.std(t_flat) > eps:
        corr_matrix = np.corrcoef(d_flat, t_flat)
        d_corr = float(corr_matrix[0, 1])
    else:
        d_corr = 0.0

    metrics: Dict[str, float] = {
        "D_error_L2": np.sqrt(d_err),
        "D_error_rel": np.sqrt(d_err / max(d_true_err, eps)),
        "D_shape_L2": np.sqrt(d_shape_err),
        "D_shape_rel": np.sqrt(d_shape_err / max(d_shape_true_err, eps)),
        "D_correlation": d_corr,
        "logD_error_L2": np.sqrt(logd_err),
        "logD_error_rel": np.sqrt(logd_err / max(logd_true_err, eps)),
        "mean_D": float(np.trapz(d_pred, x)) / max(denom, eps),
    }

    if u_pred is not None:
        metrics["integral_u"] = float(np.trapz(u_pred, x))

    if u_true is not None and u_pred is not None:
        u_err = float(np.trapz((u_pred - u_true) ** 2, x))
        u_true_err = float(np.trapz(u_true ** 2, x))
        metrics["u_error_L2"] = np.sqrt(u_err)
        metrics["u_error_rel"] = np.sqrt(u_err / max(u_true_err, eps))

        u_weight = np.clip(u_true, 0.0, None)
        d_err_u = float(np.trapz((d_pred - d_true) ** 2 * u_weight, x))
        d_true_u = float(np.trapz(d_true ** 2 * u_weight, x))
        metrics["D_error_u_L2"] = np.sqrt(d_err_u)
        metrics["D_error_u_rel"] = np.sqrt(d_err_u / max(d_true_u, eps))

    if b0_star is not None and b_true is not None:
        metrics["b0_star_err"] = abs(b0_star - b_true)
        metrics["b0_star_rel_err"] = abs(b0_star - b_true) / max(abs(b_true), eps)

    return metrics


def compute_solution_metrics(
    x: np.ndarray,
    d_true: np.ndarray,
    d_pred: np.ndarray,
    *,
    u_true: Optional[np.ndarray] = None,
    u_pred: Optional[np.ndarray] = None,
    b0_star: Optional[float] = None,
    b_true: Optional[float] = None,
) -> Dict[str, float]:
    """Compute metrics with Solution-style key names."""
    base = _compute_metrics_core(
        x,
        d_true,
        d_pred,
        u_true=u_true,
        u_pred=u_pred,
        b0_star=b0_star,
        b_true=b_true,
    )
    metrics = {
        "d_l2_error": base["D_error_L2"],
        "d_rel_error": base["D_error_rel"],
        "d_max_error": float(np.max(np.abs(d_pred - d_true))),
        "d_shape_l2_error": base["D_shape_L2"],
        "d_shape_rel_error": base["D_shape_rel"],
        "d_correlation": base["D_correlation"],
        "logd_l2_error": base["logD_error_L2"],
        "logd_rel_error": base["logD_error_rel"],
        "mean_d": base["mean_D"],
    }
    if "D_error_u_L2" in base:
        metrics["d_u_l2_error"] = base["D_error_u_L2"]
        metrics["d_u_rel_error"] = base["D_error_u_rel"]
    if "u_error_L2" in base:
        metrics["u_l2_error"] = base["u_error_L2"]
        metrics["u_rel_error"] = base["u_error_rel"]
    if "b0_star_err" in base:
        metrics["b0_abs_error"] = base["b0_star_err"]
        metrics["b0_rel_error"] = base["b0_star_rel_error"] if "b0_star_rel_error" in base else base.get("b0_star_rel_err", 0.0)
    if "integral_u" in base:
        metrics["integral_u"] = base["integral_u"]
    return metrics


def compute_loss_comparison(
    solution: "Solution",
    problem: "Problem",
    field_loss: str = "rle",
    verbose: bool = True,
) -> Dict[str, float]:
    """Compare data loss for D_pred vs D_true.

    Computes the data loss (field MSE/RLE or PPP NLL) using:
    1. The fitted D_pred and b0_star from the solution
    2. The ground-truth D_true with optimal b0 projection

    This helps assess how much of the remaining loss is due to:
    - Model error (D_pred != D_true)
    - Irreducible noise (even D_true has nonzero loss for particles)

    Args:
        solution: Fitted Solution object with D_pred, b0_star, etc.
        problem: Problem with D_true and observation data.
        field_loss: Loss type for field mode ("mse" or "rle").
        verbose: Print comparison summary.

    Returns:
        Dictionary with loss_pred, loss_true, and loss_ratio.
    """
    if problem.d_true is None:
        raise ValueError("Problem has no d_true for comparison.")

    # Get grids
    x_res = solution.x_res.detach().cpu().numpy().reshape(-1)
    x_obs = problem.x_grid.detach().cpu().numpy().reshape(-1)

    # Get D arrays
    d_pred = solution._to_numpy(solution.d_pred).reshape(-1)
    d_true = np.asarray(problem.d_true).reshape(-1)

    # Interpolate D_true to solver grid if needed
    if d_true.shape != d_pred.shape or not np.allclose(x_obs, x_res):
        d_true_res = np.interp(x_res, x_obs, d_true)
    else:
        d_true_res = d_true

    bc_type = getattr(problem, "bc_type", "dirichlet")

    # Solve FDM with D_pred (unit source)
    u_hat_pred = None
    try:
        u_hat_pred = physics.fdm_solve_alpha(
            d_pred,
            problem.alpha,
            problem.mu,
            x_res,
            1.0,
            (problem.source_location,),
            bc_type=bc_type,
        )
    except ValueError as e:
        if "D must be strictly positive" in str(e):
            if verbose:
                print(f"Warning: FDM solve with D_pred failed ({e}). Skipping FDM comparison.")
        else:
            raise

    # Solve FDM with D_true (unit source)
    u_hat_true = physics.fdm_solve_alpha(
        d_true_res,
        problem.alpha,
        problem.mu,
        x_res,
        1.0,
        (problem.source_location,),
        bc_type=bc_type,
    )

    # Convert to torch for VarPro
    x_res_t = torch.tensor(x_res, dtype=torch.float64)
    u_hat_pred_t = torch.tensor(u_hat_pred, dtype=torch.float64) if u_hat_pred is not None else None
    u_hat_true_t = torch.tensor(u_hat_true, dtype=torch.float64)
    if problem.mode == "field":
        # Field mode: use observation field
        u_obs = problem.u_field.detach().cpu().view(-1)
        x_field = problem.x_grid.detach().cpu().view(-1)

        # Interpolate u_hat to observation grid if needed
        if x_field.shape != x_res_t.shape or not torch.allclose(x_field, x_res_t):
            u_hat_pred_obs = varpro.interpolate_1d(u_hat_pred_t, x_res_t, x_field)
            u_hat_true_obs = varpro.interpolate_1d(u_hat_true_t, x_res_t, x_field)
        else:
            u_hat_pred_obs = u_hat_pred_t
            u_hat_true_obs = u_hat_true_t

        # Project b0 and compute loss for D_pred
        b0_pred = varpro.project_b0_field(u_hat_pred_obs, u_obs, field_loss=field_loss)
        loss_pred = varpro.field_data_loss(u_hat_pred_obs, u_obs, b0_pred, field_loss=field_loss)

        # Project b0 and compute loss for D_true
        b0_true = varpro.project_b0_field(u_hat_true_obs, u_obs, field_loss=field_loss)
        loss_true = varpro.field_data_loss(u_hat_true_obs, u_obs, b0_true, field_loss=field_loss)

        loss_pred_val = float(loss_pred.item())
        loss_true_val = float(loss_true.item())
        b0_pred_val = float(b0_pred.item())
        b0_true_val = float(b0_true.item())

    else:
        # Particles mode: use PPP data
        ppp = problem.particles
        x_particles = ppp.x_particles.detach().cpu().view(-1).double()

        # Compute integral on solver grid
        integral_pred = torch.trapezoid(u_hat_pred_t, x_res_t)
        integral_true = torch.trapezoid(u_hat_true_t, x_res_t)

        # Interpolate to particle locations
        u_at_pts_pred = varpro.interpolate_1d(u_hat_pred_t, x_res_t, x_particles)
        u_at_pts_true = varpro.interpolate_1d(u_hat_true_t, x_res_t, x_particles)

        # Project b0 and compute PPP NLL for D_pred
        b0_pred = varpro.project_b0_ppp(ppp.n_obs, ppp.m_obs, integral_pred)
        loss_pred = varpro.ppp_nll(u_at_pts_pred, b0_pred, ppp.m_obs, integral_pred)

        # Project b0 and compute PPP NLL for D_true
        b0_true = varpro.project_b0_ppp(ppp.n_obs, ppp.m_obs, integral_true)
        loss_true = varpro.ppp_nll(u_at_pts_true, b0_true, ppp.m_obs, integral_true)

        loss_pred_val = float(loss_pred.item())
        loss_true_val = float(loss_true.item())
        b0_pred_val = float(b0_pred.item())
        b0_true_val = float(b0_true.item())

    # Compute ratio (how much worse is D_pred vs D_true)
    eps = 1e-12
    if abs(loss_true_val) > eps:
        loss_ratio = loss_pred_val / loss_true_val
    else:
        loss_ratio = float("inf") if loss_pred_val > eps else 1.0

    result = {
        "loss_pred": loss_pred_val,
        "loss_true": loss_true_val,
        "loss_ratio": loss_ratio,
        "b0_pred": b0_pred_val,
        "b0_true": b0_true_val,
    }

    if verbose:
        mode_str = f"field ({field_loss})" if problem.mode == "field" else "PPP NLL"
        print(f"\n[Loss Comparison] Mode: {mode_str}")
        print(f"  D_pred loss: {loss_pred_val:.6e}  (b0* = {b0_pred_val:.2f})")
        print(f"  D_true loss: {loss_true_val:.6e}  (b0* = {b0_true_val:.2f})")
        print(f"  Ratio (pred/true): {loss_ratio:.3f}x")
        if problem.b_true is not None:
            print(f"  (True b0 = {problem.b_true:.2f})")

    return result


def compute_field_metrics(
    x: np.ndarray,
    d_true: np.ndarray,
    u_true: np.ndarray,
    d_pred: np.ndarray,
    u_pred: np.ndarray,
    b0_star: float,
    b_true: float,
) -> Dict[str, float]:
    """Compute L2/relative errors for D and u plus summary stats.

    Relative errors use sqrt(∫|true-est|^2 / ∫|true|^2).
    """
    return _compute_metrics_core(
        x,
        d_true,
        d_pred,
        u_true=u_true,
        u_pred=u_pred,
        b0_star=b0_star,
        b_true=b_true,
    )


def plot_field_comparison(
    x: np.ndarray,
    d_true: np.ndarray,
    u_true: np.ndarray,
    results: Dict[str, object],
    metrics: Dict[str, Dict[str, float]],
    outdir: str | None,
    filename: str = "comparison.png",
) -> None:
    """Plot D(x) and u(x) curves for field data."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax_d = axes[0]
    ax_d.plot(x, d_true, "k-", label="D true", linewidth=2)
    for name, res in results.items():
        rel = metrics[name]["D_error_rel"]
        ax_d.plot(x, res.d_pred, label=f"D {name} (rel {rel:.2e})")
    ax_d.set_ylabel("D(x)")
    ax_d.legend()
    ax_d.grid(True, alpha=0.2)

    ax_u = axes[1]
    ax_u.plot(x, u_true, "k-", label="u true", linewidth=2)
    for name, res in results.items():
        rel = metrics[name]["u_error_rel"]
        ax_u.plot(x, res.u_pred, label=f"u {name} (rel {rel:.2e})")
    ax_u.set_ylabel("u(x)")
    ax_u.legend()
    ax_u.grid(True, alpha=0.2)

    ax_err = axes[2]
    for name, res in results.items():
        ax_err.plot(x, res.u_pred - u_true, label=f"u error {name}")
    ax_err.axhline(0.0, color="k", linewidth=0.8)
    ax_err.set_xlabel("x")
    ax_err.set_ylabel("u_pred - u_true")
    ax_err.legend()
    ax_err.grid(True, alpha=0.2)

    plt.tight_layout()

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, filename), dpi=150)
    else:
        plt.show()


def plot_solution_comparison(
    x: np.ndarray,
    d_true: np.ndarray,
    u_true: np.ndarray,
    results: Dict[str, object],
    metrics: Dict[str, Dict[str, float]],
    outdir: str | None,
    mode: str,
    x_particles: Optional[np.ndarray] = None,
    filename: str = "comparison.png",
) -> None:
    """Plot either field or particle comparisons based on the mode."""
    mode = mode.lower()
    if mode == "field":
        plot_field_comparison(x, d_true, u_true, results, metrics, outdir, filename=filename)
        return
    if mode != "particles":
        raise ValueError(f"Unsupported mode '{mode}' for plot_solution_comparison.")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_d = axes[0]
    ax_d.plot(x, d_true, "k-", label="D true", linewidth=2)
    for name, res in results.items():
        rel = metrics[name]["D_error_rel"]
        ax_d.plot(x, res.d_pred, label=f"D {name} (rel {rel:.2e})")
    ax_d.set_ylabel("D(x)")
    ax_d.legend()
    ax_d.grid(True, alpha=0.2)

    ax_u = axes[1]
    if x_particles is not None and x_particles.size > 0:
        ax_u.hist(x_particles, bins=50, density=True, color="gray", alpha=0.3, label="samples")
    int_true = max(float(np.trapz(u_true, x)), 1e-12)
    ax_u.plot(x, u_true / int_true, "k--", label="true PDF")
    for name, res in results.items():
        int_pred = max(float(np.trapz(np.asarray(res.u_pred), x)), 1e-12)
        ax_u.plot(x, np.asarray(res.u_pred) / int_pred, label=f"pred PDF {name}")
    ax_u.set_xlabel("x")
    ax_u.set_ylabel("pdf")
    ax_u.legend()
    ax_u.grid(True, alpha=0.2)

    plt.tight_layout()
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, filename), dpi=150)
    else:
        plt.show()


def plot_training_history(
    name: str,
    hist: Dict[str, list],
    b_true: float,
    outdir: str | None,
    filename: str | None = None,
    mean_d_true: float | None = None,
    weights: Dict[str, float] | None = None,
    log_threshold: float = 5.0,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Plot training loss curves and summary statistics.

    Loss components are plotted as change from initial value (val - val[0]),
    which puts all components on commensurate scales. This is especially useful
    for particle mode where NLL can be large negative (-50) but changes by small
    amounts that are meaningful relative to other loss terms.

    Args:
        name: Method name.
        hist: Dictionary of history lists.
        b_true: True source strength.
        outdir: Output directory.
        filename: Output filename.
        mean_d_true: True mean diffusivity for reference.
        weights: Optional dictionary mapping history keys to their scalar weights.
                 If provided, plots effective (weighted) loss as solid lines and
                 raw (unweighted) loss as dashed lines.
        log_threshold: Log-scale b0/mean-D axes if values exceed this multiple of truth.
    """
    iters = np.array(hist.get("iter", []))
    if iters.size == 0:
        return

    # Detect BILO mode: if "upper" and "lower" are present, split the loss panel.
    is_bilo = "upper" in hist and "lower" in hist

    # Define component groups
    upper_components = ["data", "reg_smooth", "reg_scale"]
    lower_components = ["phys", "res", "jump", "bc", "rgrad", "jump_rgrad"]
    all_components = upper_components + lower_components

    def _finite(values: list[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return arr[np.isfinite(arr)]

    def _subtract_initial(vals: np.ndarray) -> tuple[np.ndarray, float]:
        """Subtract initial value, return (shifted_vals, initial_value)."""
        if len(vals) == 0:
            return vals, 0.0
        v0 = vals[0]
        if not np.isfinite(v0):
            v0 = 0.0
        return vals - v0, v0

    def _compute_symlog_params(all_delta_vals: list) -> tuple[float, float, float]:
        """Compute symlog linthresh and axis limits from delta values."""
        if not all_delta_vals:
            return 1.0, -1, 1

        all_vals = np.concatenate([v for v in all_delta_vals if len(v) > 0])
        if len(all_vals) == 0:
            return 1.0, -1, 1

        abs_vals = np.abs(all_vals[all_vals != 0])

        # linthresh: use a small percentile of absolute values for smooth transition
        if abs_vals.size > 0:
            linthresh = max(np.percentile(abs_vals, 10), 1e-10)
        else:
            linthresh = 1e-6

        # Axis limits: tight to data with small padding
        vmin, vmax = np.min(all_vals), np.max(all_vals)
        if vmin == vmax:
            vmin, vmax = vmin - 1, vmax + 1
        pad = 0.1 * max(abs(vmax), abs(vmin), linthresh)
        return linthresh, vmin - pad, vmax + pad

    def _plot_group_delta(ax, keys, aggregate_key=None, aggregate_label="Total", aggregate_color="k"):
        """Plot loss components as change from initial value using symlog scale."""
        # First pass: compute all delta values and initial values for legend
        delta_vals_list = []
        initial_vals = {}

        if aggregate_key and aggregate_key in hist:
            vals = np.array(hist[aggregate_key])
            delta, v0 = _subtract_initial(vals)
            delta_vals_list.append(delta)
            initial_vals[aggregate_key] = v0

        for key in keys:
            if key in hist and len(hist[key]) > 0:
                val = np.array(hist[key])
                if weights and key in weights:
                    w = weights[key]
                    if w != 0.0:
                        eff_val = val * w
                        delta, v0 = _subtract_initial(eff_val)
                        delta_vals_list.append(delta)
                        initial_vals[f"{key}_eff"] = v0
                else:
                    delta, v0 = _subtract_initial(val)
                    delta_vals_list.append(delta)
                    initial_vals[key] = v0

        linthresh, ymin, ymax = _compute_symlog_params(delta_vals_list)

        # Second pass: plot
        if aggregate_key and aggregate_key in hist:
            vals = np.array(hist[aggregate_key])
            delta, v0 = _subtract_initial(vals)
            label = f"{aggregate_label} (init: {v0:.2e})"
            ax.plot(iters, delta, color=aggregate_color, linewidth=2, label=label)

        for key in keys:
            if key in hist and len(hist[key]) > 0:
                val = np.array(hist[key])
                if weights and key in weights:
                    w = weights[key]
                    color = None
                    if w != 0.0:
                        eff_val = val * w
                        delta, v0 = _subtract_initial(eff_val)
                        label = f"{key} (init: {v0:.2e})"
                        (line,) = ax.plot(iters, delta, alpha=0.7, label=label)
                        color = line.get_color()
                    if w != 1.0:
                        delta_raw, _ = _subtract_initial(val)
                        ax.plot(iters, delta_raw, color=color, linestyle="--", alpha=0.4)
                else:
                    delta, v0 = _subtract_initial(val)
                    label = f"{key} (init: {v0:.2e})"
                    ax.plot(iters, delta, alpha=0.6, label=label)

        ax.axhline(0, color="k", linestyle=":", alpha=0.3, linewidth=0.5)
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel("Δ Loss (from initial)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.2)

    # ==================== Create Figure ====================
    if is_bilo:
        fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
        ax_upper, ax_lower, ax_b, ax_mean = axes
    else:
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        ax_upper, ax_b, ax_mean = axes
        ax_lower = None

    if is_bilo:
        _plot_group_delta(ax_upper, upper_components, aggregate_key="upper",
                         aggregate_label="Upper Total", aggregate_color="b")
        ax_upper.set_title(f"Training History: {name} (Upper Level)")

        _plot_group_delta(ax_lower, lower_components, aggregate_key="lower",
                         aggregate_label="Lower Total", aggregate_color="r")
        ax_lower.set_title("Lower Level")
    else:
        agg_key = "total" if "total" in hist else None
        _plot_group_delta(ax_upper, all_components, aggregate_key=agg_key,
                         aggregate_label="Total", aggregate_color="k")
        ax_upper.set_title(f"Training History: {name}")

    # b0 and Mean D panels
    if "b0_star" in hist:
        b_vals = np.array(hist["b0_star"])
        ax_b.plot(iters, b_vals, label="b0*")
        ax_b.axhline(b_true, color="k", linestyle="--", linewidth=1.0, label="b_true")
        if (b_true > 0.0 and b_vals.size > 0 and np.all(b_vals > 0.0)
                and np.nanmax(b_vals) > log_threshold * b_true):
            ax_b.set_yscale("log")
    ax_b.set_ylabel("Source Strength b0")
    ax_b.legend()
    ax_b.grid(True, alpha=0.2)

    if "mean_d" in hist:
        mean_vals = np.array(hist["mean_d"])
        ax_mean.plot(iters, mean_vals, label="⟨D⟩")
        if (mean_d_true is not None and mean_d_true > 0.0 and mean_vals.size > 0
                and np.all(mean_vals > 0.0) and np.nanmax(mean_vals) > log_threshold * mean_d_true):
            ax_mean.set_yscale("log")
    if mean_d_true is not None:
        ax_mean.axhline(mean_d_true, color="k", linestyle="--", linewidth=1.0, label="⟨D⟩ true")
    ax_mean.set_xlabel("Iteration")
    ax_mean.set_ylabel("⟨D⟩")
    ax_mean.legend()
    ax_mean.grid(True, alpha=0.2)

    plt.tight_layout()

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        if filename is None:
            filename = f"{name.lower()}_history.png"
        fig.savefig(os.path.join(outdir, filename), dpi=150)
        plt.close(fig)
        return None
    elif show:
        plt.show()
        return None
    else:
        return fig


def plot_d_evolution(
    name: str,
    hist: Dict[str, list],
    x: np.ndarray,
    outdir: str | None,
    filename: str | None = None,
    mean_d_true: float | None = None,
    log_threshold: float = 5.0,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Visualize D(x) snapshots across training iterations."""
    snaps = hist.get("d_snapshots", [])
    iters = hist.get("d_snap_iters", [])
    if not snaps:
        return
    d_stack = np.stack(snaps, axis=0)
    log_scale = False
    if mean_d_true is not None and mean_d_true > 0.0:
        mean_vals = np.array(hist.get("mean_d", []))
        if (
            mean_vals.size > 0
            and np.all(mean_vals > 0.0)
            and np.nanmax(mean_vals) > log_threshold * mean_d_true
        ):
            log_scale = True

    norm = None
    if log_scale:
        positive = d_stack[d_stack > 0.0]
        if positive.size > 0:
            from matplotlib.colors import LogNorm

            vmin = max(float(np.min(positive)), 1e-12)
            vmax = float(np.max(positive))
            norm = LogNorm(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(f"D(x) evolution: {name}")
    im = ax.imshow(
        d_stack,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], iters[0], iters[-1]],
        norm=norm,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("iter")
    plt.colorbar(im, ax=ax, label="D(x)")
    plt.tight_layout()
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        if filename is None:
            filename = f"{name.lower()}_d_evolution.png"
        fig.savefig(os.path.join(outdir, filename), dpi=150)
        plt.close(fig)
        return None
    elif show:
        plt.show()
        return None
    else:
        return fig


def plot_d_evolution_color(
    name: str,
    hist: Dict[str, list],
    x: np.ndarray,
    outdir: str | None = None,
    filename: str | None = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Visualize D(x) evolution across training iterations with color gradient.
    
    Plots D(x) curves for each snapshot, colored from blue (initial) to yellow (final).
    
    Args:
        name: Method name for title
        hist: Training history with d_snapshots and d_snap_iters
        x: Spatial grid (1D numpy array)
        outdir: Output directory (if None, don't save)
        filename: Output filename (if None, auto-generate)
        show: Whether to show the plot
        
    Returns:
        matplotlib Figure if show=False, None otherwise
    """
    snaps = hist.get("d_snapshots", [])
    iters = hist.get("d_snap_iters", [])
    
    if not snaps:
        return None
    
    # Debug: print snapshot info
    if len(snaps) == 1:
        print(f"Warning: Only 1 snapshot found in history (iter {iters[0] if iters else 'unknown'})")
    else:
        print(f"Plotting {len(snaps)} snapshots from iterations: {iters[:5]}{'...' if len(iters) > 5 else ''}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"D(x) evolution: {name}")
    
    # Sample colors evenly from viridis colormap
    n_snaps = len(snaps)
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n_snaps - 1, 1)) for i in range(n_snaps)]
    
    # Plot D(x) for each snapshot
    for i, (d_snap, iter_val) in enumerate(zip(snaps, iters)):
        # Ensure d_snap is 1D and matches x length
        d_snap = np.asarray(d_snap).reshape(-1)
        if len(d_snap) != len(x):
            # Try to interpolate if lengths don't match
            if len(d_snap) > 0:
                x_snap = np.linspace(x[0], x[-1], len(d_snap))
                d_snap = np.interp(x, x_snap, d_snap)
            else:
                continue  # Skip empty snapshots
        
        # Plot with color gradient
        label = f"iter {iter_val}" if i == 0 or i == n_snaps - 1 else None
        ax.plot(x, d_snap, color=colors[i], alpha=0.7, linewidth=1.5, label=label)
    
    ax.set_xlabel("x")
    ax.set_ylabel("D(x)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        if filename is None:
            filename = f"{name.lower()}_d_evolution.png"
        fig.savefig(os.path.join(outdir, filename), dpi=150)
        plt.close(fig)
        return None
    elif show:
        plt.show()
        return None
    else:
        return fig


def plot_bilo_d_variation(
    solution: "Solution",
    problem: "Problem",
    outdir: str | None = None,
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
    
    # Get D_0 after pretraining (evaluate d_net at x_res)
    with torch.no_grad():
        D_0 = solution.d_net(x_res_t).view(-1).detach().cpu().numpy()
    
    # Compute mean d
    d_mean = float(np.mean(D_0))
    
    # Create variations
    variations = {
        "shiftplus": lambda x: D_0 + 0.5 * d_mean * np.ones_like(x),
        "shiftminus": lambda x: D_0 - 0.5 * d_mean * np.ones_like(x),
        "linplus": lambda x: D_0 + 0.5 * d_mean * x,
        "linminus": lambda x: D_0 - 0.5 * d_mean * x,
    }
    
    # Prepare z_tensor for local operator (use network device)
    z_tensor = torch.tensor(
        [[problem.source_location]],
        device=device,
        dtype=dtype
    )
    
    # Create figure with subplots for each variation
    n_vars = len(variations)
    fig, axes = plt.subplots(n_vars, 2, figsize=(12, 4 * n_vars))
    if n_vars == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (varkey, varfun) in enumerate(variations.items()):
        D_var = varfun(x_np)
        D_var_t = torch.tensor(D_var, device=device, dtype=dtype).view(-1, 1)
        
        # Compute u(x, D_var) using local operator
        with torch.no_grad():
            u_hat_var, _ = solution.local_op(x_res_t, D_var_t, z_tensor)
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
        
        # Plot u_0 for reference
        with torch.no_grad():
            D_0_t = torch.tensor(D_0, device=device, dtype=dtype).view(-1, 1)
            u_hat_0, _ = solution.local_op(x_res_t, D_0_t, z_tensor)
            u_0 = solution.b0_star * u_hat_0.view(-1).detach().cpu().numpy()
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
        filename = "bilo_d_variation.png"
        fig.savefig(os.path.join(outdir, filename), dpi=150)
        plt.close(fig)
        return None
    elif show:
        plt.show()
        return None
    else:
        return fig


def plot_particle_comparison(
    x: np.ndarray,
    u_true: np.ndarray,
    u_pred: np.ndarray,
    x_particles: Optional[np.ndarray],
    b0_star: float,
    outdir: str | None,
    filename: str = "particle_comparison.png",
    bins: int = 50,
) -> None:
    """Plot particle histogram against predicted/true densities."""
    fig, ax = plt.subplots(figsize=(10, 4))
    if x_particles is not None and x_particles.size > 0:
        ax.hist(x_particles, bins=bins, density=True, color="gray", alpha=0.3, label="samples")

    int_true = max(float(np.trapz(u_true, x)), 1e-12)
    int_pred = max(float(np.trapz(u_pred, x)), 1e-12)
    ax.plot(x, u_true / int_true, "k--", label="true PDF")
    ax.plot(x, u_pred / int_pred, "b-", label=f"pred PDF (b={b0_star:.2f})")
    ax.set_xlabel("x")
    ax.set_ylabel("pdf")
    ax.legend()
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, filename), dpi=150)
    else:
        plt.show()


def plot_bilo_neighborhood_check(
    d_net: torch.nn.Module,
    local_op: torch.nn.Module,
    x_res: torch.Tensor,
    z: float,
    alpha: float,
    mu: float,
    bc_type: str = "dirichlet",
    delta_d: float = 0.05,
    outdir: str | None = None,
    filename: str = "bilo_neighborhood_check.png",
) -> None:
    """Compare local operator outputs under small D perturbations."""
    device = x_res.device
    dtype = x_res.dtype
    z_tensor = torch.tensor(z, device=device, dtype=dtype).view(1, 1)
    x_plot = x_res.view(-1, 1)

    with torch.no_grad():
        d_base = d_net(x_plot)
        d_perturbed = d_base + delta_d
        u_hat_base, _ = local_op(x_plot, d_base, z_tensor)
        u_hat_perturbed, _ = local_op(x_plot, d_perturbed, z_tensor)

    x_np = x_plot.detach().cpu().numpy().reshape(-1)
    d_base_np = d_base.detach().cpu().numpy().reshape(-1)
    d_perturbed_np = d_perturbed.detach().cpu().numpy().reshape(-1)

    u_fdm_base = physics.fdm_solve_alpha(
        d_base_np, alpha, mu, x_np, 1.0, [float(z)], bc_type=bc_type
    )
    u_fdm_perturbed = physics.fdm_solve_alpha(
        d_perturbed_np, alpha, mu, x_np, 1.0, [float(z)], bc_type=bc_type
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(x_np, d_base_np, "k-", label="D")
    axes[0].plot(x_np, d_perturbed_np, "r-", label="D + ΔD")
    axes[0].set_title("D perturbation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(x_np, u_hat_base.detach().cpu().numpy().flatten(), label="LocalOp unit (D)")
    axes[1].plot(x_np, u_hat_perturbed.detach().cpu().numpy().flatten(), label="LocalOp unit (D+ΔD)")
    axes[1].plot(x_np, u_fdm_base, "k--", label="FDM unit (D)")
    axes[1].plot(x_np, u_fdm_perturbed, "r--", label="FDM unit (D+ΔD)")
    axes[1].set_title(f"Neighborhood check (ΔD={delta_d:+.2f})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, filename), dpi=150)
    else:
        plt.show()
