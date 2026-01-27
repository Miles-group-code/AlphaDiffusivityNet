"""Training logger utilities for method files.

This module provides helpers for logging training progress and history tracking,
reducing boilerplate in the method files (method_dto.py, method_pinn.py, method_bilo.py).

The goal is to corner off logging/history logic so the core algorithm in each
method file is easier to read and understand.

Usage:
    from training_logger import TrainingHistory, format_dto_progress, format_pinn_progress, format_bilo_progress

    # Initialize history
    history = TrainingHistory.for_dto()  # or for_pinn(), for_bilo()

    # In training loop:
    if step % log_every == 0:
        history.log(
            step=step,
            total=total_loss.item(),
            data=data_loss.item(),
            ...
        )
        if verbose:
            print(format_dto_progress(step, losses, scalars, bc_type))

    # After training:
    return DTOResult(..., history=history.to_dict())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None


# =============================================================================
# TRAINING HISTORY
# =============================================================================

class TrainingHistory:
    """Accumulator for training metrics with method-specific factory methods.

    This class handles the boilerplate of initializing history dicts and
    appending metrics at each logged step. Each method has different tracked
    metrics, so we provide factory methods for each.

    Example:
        history = TrainingHistory.for_dto()
        history.log(step=0, total=1.5, data=1.0, reg_smooth=0.3, ...)
        history.log_snapshot(step=0, d_snapshot=d_array)
        result_dict = history.to_dict()
    """

    def __init__(self, keys: List[str]):
        """Initialize with the list of metric keys to track.

        Args:
            keys: List of metric names (e.g., ["iter", "total", "data", ...])
        """
        self._data: Dict[str, List[Any]] = {k: [] for k in keys}
        # Always include snapshot tracking
        if "d_snap_iters" not in self._data:
            self._data["d_snap_iters"] = []
        if "d_snapshots" not in self._data:
            self._data["d_snapshots"] = []

    @classmethod
    def for_dto(cls) -> "TrainingHistory":
        """Create history tracker for DTO method.

        Tracked metrics:
            iter, total, data, reg_smooth, reg_scale, b0_star, mean_d
        """
        return cls([
            "iter", "total", "data", "reg_smooth", "reg_scale",
            "b0_star", "mean_d", "d_snap_iters", "d_snapshots"
        ])

    @classmethod
    def for_pinn(cls) -> "TrainingHistory":
        """Create history tracker for PINN method.

        Tracked metrics:
            iter, total, data, phys, res, jump, bc, reg_smooth, reg_scale,
            b0_star, mean_d
        """
        return cls([
            "iter", "total", "data", "phys", "res", "jump", "bc",
            "reg_smooth", "reg_scale", "b0_star", "mean_d",
            "d_snap_iters", "d_snapshots"
        ])

    @classmethod
    def for_bilo(cls) -> "TrainingHistory":
        """Create history tracker for BiLO method.

        Tracked metrics:
            iter, upper, data, reg_smooth, reg_scale, lower, res, jump, bc,
            bc_grad, rgrad, jump_rgrad, b0_star, mean_d,
            d_err_l2, d_err_linf, u_fdm_err
        """
        return cls([
            "iter", "upper", "data", "reg_smooth", "reg_scale",
            "lower", "res", "jump", "bc", "bc_grad", "rgrad", "jump_rgrad",
            "b0_star", "mean_d", "d_snap_iters", "d_snapshots",
            "d_err_l2", "d_err_linf", "u_fdm_err"
        ])

    def log(self, **kwargs) -> None:
        """Log metrics for the current step.

        Args:
            **kwargs: Metric name-value pairs. Only metrics that exist in
                     the history will be recorded (others are silently ignored).

        Example:
            history.log(step=100, total=1.5, data=1.0, mean_d=0.05)
        """
        # Handle 'step' -> 'iter' mapping for convenience
        if "step" in kwargs:
            kwargs["iter"] = kwargs.pop("step")

        for k, v in kwargs.items():
            if k in self._data:
                self._data[k].append(v)
        
        # Log to wandb if active
        if wandb is not None and wandb.run is not None:
            # We filter only what is in _data to be consistent, or just log everything provided?
            # The docstring says "others are silently ignored" for self._data. 
            # For wandb, we probably want to log what's being tracked.
            log_dict = {k: v for k, v in kwargs.items() if k in self._data}
            if "iter" in log_dict:
                step = log_dict["iter"]
                wandb.log(log_dict, step=step)

    def log_snapshot(self, step: int, d_snapshot: np.ndarray) -> None:
        """Log a D(x) snapshot for evolution visualization.

        Args:
            step: Iteration number
            d_snapshot: D(x) values as numpy array
        """
        self._data["d_snap_iters"].append(step)
        self._data["d_snapshots"].append(d_snapshot)

    def to_dict(self) -> Dict[str, List[Any]]:
        """Return the accumulated history as a plain dict.

        Returns:
            Dict mapping metric names to lists of values.
        """
        return self._data


# =============================================================================
# PROGRESS FORMATTERS
# =============================================================================
# Each method has its own format because they track different loss components.
# The formatters produce multi-line strings for console output.

def format_dto_progress(
    step: int,
    total: float,
    data: float,
    reg_smooth: float,
    reg_scale: float,
    wreg_smooth: float,
    wreg_scale: float,
    b0_star: float,
    integral_unit: float,
    mean_d: float,
    loss_name: str = "mse",
) -> str:
    """Format DTO training progress for console output.

    Args:
        step: Current iteration
        total: Total loss value
        data: Data loss (MSE, RLE, or PPP NLL)
        reg_smooth: Smoothness regularization (unweighted)
        reg_scale: Scale anchor regularization (unweighted)
        wreg_smooth: Smoothness weight (for effective value display)
        wreg_scale: Scale anchor weight (for effective value display)
        b0_star: Projected source amplitude
        integral_unit: Integral of unit-response
        mean_d: Mean D value
        loss_name: Loss type label ("mse", "rle", or "ppp")

    Returns:
        Formatted multi-line string for printing.
    """
    reg_smooth_eff = wreg_smooth * reg_smooth
    reg_scale_eff = wreg_scale * reg_scale
    u_int = b0_star * integral_unit

    return (
        f"[DTO] Iter {step:05d} | Ltot: {total:.3e}\n"
        f"  Ldata({loss_name}): {data:.3e} | "
        f"RegSmooth: {reg_smooth:.3e} (eff: {reg_smooth_eff:.3e}) | "
        f"RegScale: {reg_scale:.3e} (eff: {reg_scale_eff:.3e})\n"
        f"  b0*: {b0_star:.2f} | int_u_hat: {integral_unit:.3e} | int_u: {u_int:.3e} | mean_D: {mean_d:.3e}"
    )


def format_pinn_progress(
    step: int,
    phase: str,
    total: float,
    data: float,
    phys: float,
    res: float,
    jump: float,
    bc: float,
    reg_smooth: float,
    reg_scale: float,
    wreg_smooth: float,
    wreg_scale: float,
    b0_star: float,
    integral_unit: float,
    mean_d: float,
    loss_name: str = "mse",
    bc_type: str = "dirichlet",
) -> str:
    """Format PINN training progress for console output.

    Args:
        step: Current iteration
        phase: Training phase ("pretrain" or "finetune")
        total: Total loss value
        data: Data loss
        phys: Weighted physics loss
        res: PDE residual loss (unweighted)
        jump: Jump condition loss (unweighted)
        bc: Boundary condition loss (unweighted, Neumann only)
        reg_smooth: Smoothness regularization (unweighted)
        reg_scale: Scale anchor regularization (unweighted)
        wreg_smooth: Smoothness weight
        wreg_scale: Scale anchor weight
        b0_star: Projected source amplitude
        integral_unit: Integral of unit-response
        mean_d: Mean D value
        loss_name: Loss type label
        bc_type: Boundary condition type (for conditional display)

    Returns:
        Formatted multi-line string for printing.
    """
    reg_smooth_eff = wreg_smooth * reg_smooth
    reg_scale_eff = wreg_scale * reg_scale
    u_int = b0_star * integral_unit
    bc_str = f" | Lbc: {bc:.3e}" if bc_type == "neumann" else ""

    return (
        f"[PINN|{phase}] Iter {step:05d} | Ltot: {total:.3e}\n"
        f"  Ldata({loss_name}): {data:.3e} | Lphys: {phys:.3e} | "
        f"RegSmooth: {reg_smooth:.3e} (eff: {reg_smooth_eff:.3e}) | "
        f"RegScale: {reg_scale:.3e} (eff: {reg_scale_eff:.3e})\n"
        f"  Lres: {res:.3e} | Ljump: {jump:.3e}{bc_str}\n"
        f"  b0*: {b0_star:.2f} | int_u_hat: {integral_unit:.3e} | int_u: {u_int:.3e} | mean_D: {mean_d:.3e}"
    )


def format_pinn_pretrain_progress(
    step: int,
    total: float,
    phys: float,
    anchor: float,
    res: float,
    jump: float,
    bc: float,
    mean_d: float,
    bc_type: str = "dirichlet",
) -> str:
    """Format PINN pretrain progress (simpler, no data loss).

    Args:
        step: Current iteration
        total: Total pretrain loss
        phys: Physics loss (weighted)
        anchor: Scale anchor loss
        res: PDE residual loss (unweighted)
        jump: Jump condition loss (unweighted)
        bc: Boundary condition loss (unweighted)
        mean_d: Mean D value
        bc_type: Boundary condition type

    Returns:
        Formatted multi-line string for printing.
    """
    bc_str = f" | Lbc: {bc:.3e}" if bc_type == "neumann" else ""

    return (
        f"[PINN|pretrain] Iter {step:05d} | Ltot: {total:.3e}\n"
        f"  Lphys: {phys:.3e} | Lanchor: {anchor:.3e}\n"
        f"  Lres: {res:.3e} | Ljump: {jump:.3e}{bc_str}\n"
        f"  mean_D: {mean_d:.3e}"
    )


def format_bilo_progress(
    step: int,
    phase: str,
    metrics: Dict[str, float],
    weights: Dict[str, float],
    loss_name: str = "mse",
    bc_type: str = "dirichlet",
) -> str:
    """Format BiLO training progress for console output.

    Args:
        step: Current iteration
        phase: Training phase ("pretrain" or "finetune")
        metrics: Dict containing the scalar metrics to display:
            total, upper, data, reg_smooth, reg_scale, lower, res, jump, bc,
            rgrad, jump_rgrad, b0_star, integral_unit, mean_d
        weights: Dict containing weights to display effective reg terms:
            wreg_smooth, wreg_scale
        loss_name: Loss type label
        bc_type: Boundary condition type

    Returns:
        Formatted multi-line string for printing.
    """
    total = float(metrics["total"])
    upper = float(metrics["upper"])
    data = float(metrics["data"])
    reg_smooth = float(metrics["reg_smooth"])
    reg_scale = float(metrics["reg_scale"])
    lower = float(metrics["lower"])
    res = float(metrics["res"])
    jump = float(metrics["jump"])
    bc = float(metrics["bc"])
    rgrad = float(metrics["rgrad"])
    jump_rgrad = float(metrics["jump_rgrad"])
    b0_star = float(metrics["b0_star"])
    integral_unit = float(metrics["integral_unit"])
    mean_d = float(metrics["mean_d"])

    wreg_smooth = float(weights.get("wreg_smooth", 0.0))
    wreg_scale = float(weights.get("wreg_scale", 0.0))
    reg_smooth_eff = wreg_smooth * reg_smooth
    reg_scale_eff = wreg_scale * reg_scale
    u_int = b0_star * integral_unit
    bc_str = f" | Lbc: {bc:.3e}" if bc_type == "neumann" else ""

    # Append validation metrics if they exist
    val_str = ""
    if "d_err_l2" in metrics:
        val_str += f"\n  D_err: L2 {float(metrics['d_err_l2']):.3e} | Linf {float(metrics['d_err_linf']):.3e}"
    if "u_fdm_err" in metrics:
        val_str += f" | U_consistency: L2 {float(metrics['u_fdm_err']):.3e}"

    # Append dynamic rgrad metrics
    rgrad_str = ""
    for k in sorted(metrics.keys()):
        if k.startswith("rgrad_d"):
            rgrad_str += f" | {k}: {metrics[k]:.3e}"
        elif k.startswith("jump_rgrad_d"):
            rgrad_str += f" | {k}: {metrics[k]:.3e}"
        elif k.startswith("bc_grad_d"):
            rgrad_str += f" | {k}: {metrics[k]:.3e}"
    
    if rgrad_str:
        val_str += "\n  " + rgrad_str.strip(" |")

    return (
        f"[BiLO|{phase}] Iter {step:05d} | Ltot: {total:.3e}\n"
        f"  Upper: {upper:.3e} | Ldata({loss_name}): {data:.3e} | "
        f"RegSmooth: {reg_smooth:.3e} (eff: {reg_smooth_eff:.3e}) | "
        f"RegScale: {reg_scale:.3e} (eff: {reg_scale_eff:.3e})\n"
        f"  Lower: {lower:.3e} | Lres: {res:.3e} | Ljump: {jump:.3e}{bc_str} | "
        f"Lrgrad: {rgrad:.3e} | Ljump_rgrad: {jump_rgrad:.3e}\n"
        f"  b0*: {b0_star:.2f} | int_u_hat: {integral_unit:.3e} | int_u: {u_int:.3e} | mean_D: {mean_d:.3e}"
        f"{val_str}"
    )


def format_bilo_pretrain_progress(
    step: int,
    metrics: Dict[str, float],
    bc_type: str = "dirichlet",
) -> str:
    """Format BiLO pretrain progress.

    Args:
        step: Current iteration
        metrics: Dict containing the scalar metrics to display:
            total, anchor, lower, sup, res, jump, bc, bc_grad, rgrad, jump_rgrad, mean_d
        bc_type: Boundary condition type

    Returns:
        Formatted multi-line string for printing.
    """
    total = float(metrics["total"])
    anchor = float(metrics["anchor"])
    lower = float(metrics["lower"])
    sup = float(metrics["sup"])
    res = float(metrics["res"])
    jump = float(metrics["jump"])
    bc = float(metrics["bc"])
    bc_grad = float(metrics.get("bc_grad", 0.0))
    rgrad = float(metrics["rgrad"])
    jump_rgrad = float(metrics["jump_rgrad"])
    mean_d = float(metrics["mean_d"])

    bc_str = f" | Lbc: {bc:.3e}" if bc_type == "neumann" else ""
    bc_grad_str = f" | Lbc_grad: {bc_grad:.3e}" if bc_type == "neumann" else ""

    # Append dynamic rgrad metrics
    rgrad_str = ""
    for k in sorted(metrics.keys()):
        if k.startswith("rgrad_d"):
            rgrad_str += f" | {k}: {metrics[k]:.3e}"
        elif k.startswith("jump_rgrad_d"):
            rgrad_str += f" | {k}: {metrics[k]:.3e}"
        elif k.startswith("bc_grad_d"):
            rgrad_str += f" | {k}: {metrics[k]:.3e}"
    
    extra_str = ""
    if rgrad_str:
        extra_str = "\n  " + rgrad_str.strip(" |")

    return (
        f"[BiLO|pretrain] Iter {step:05d} | Ltot: {total:.3e}\n"
        f"  Lanchor: {anchor:.3e} | Llower: {lower:.3e} | Lsup: {sup:.3e}\n"
        f"  Lres: {res:.3e} | Ljump: {jump:.3e}{bc_str}{bc_grad_str} | "
        f"Lrgrad: {rgrad:.3e} | Ljump_rgrad: {jump_rgrad:.3e}\n"
        f"  mean_D: {mean_d:.3e}"
        f"{extra_str}"
    )
