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
            rgrad, jump_rgrad, b0_star, mean_d
        """
        return cls([
            "iter", "upper", "data", "reg_smooth", "reg_scale",
            "lower", "res", "jump", "bc", "rgrad", "jump_rgrad",
            "b0_star", "mean_d", "d_snap_iters", "d_snapshots"
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
    total: float,
    upper: float,
    data: float,
    reg_smooth: float,
    reg_scale: float,
    lower: float,
    res: float,
    jump: float,
    bc: float,
    rgrad: float,
    jump_rgrad: float,
    wreg_smooth: float,
    wreg_scale: float,
    b0_star: float,
    integral_unit: float,
    mean_d: float,
    loss_name: str = "mse",
    bc_type: str = "dirichlet",
) -> str:
    """Format BiLO training progress for console output.

    Args:
        step: Current iteration
        phase: Training phase ("pretrain" or "finetune")
        total: Total loss (upper + lower)
        upper: Upper-level loss (data + regularization)
        data: Data loss
        reg_smooth: Smoothness regularization (unweighted)
        reg_scale: Scale anchor regularization (unweighted)
        lower: Lower-level loss (physics)
        res: PDE residual loss (unweighted)
        jump: Jump condition loss (unweighted)
        bc: Boundary condition loss (unweighted)
        rgrad: Residual gradient penalty (unweighted)
        jump_rgrad: Jump gradient penalty (unweighted)
        wreg_smooth: Smoothness weight
        wreg_scale: Scale anchor weight
        b0_star: Projected source amplitude
        integral_unit: Integral of unit-response
        mean_d: Mean D value
        loss_name: Loss type label
        bc_type: Boundary condition type

    Returns:
        Formatted multi-line string for printing.
    """
    reg_smooth_eff = wreg_smooth * reg_smooth
    reg_scale_eff = wreg_scale * reg_scale
    u_int = b0_star * integral_unit
    bc_str = f" | Lbc: {bc:.3e}" if bc_type == "neumann" else ""

    return (
        f"[BiLO|{phase}] Iter {step:05d} | Ltot: {total:.3e}\n"
        f"  Upper: {upper:.3e} | Ldata({loss_name}): {data:.3e} | "
        f"RegSmooth: {reg_smooth:.3e} (eff: {reg_smooth_eff:.3e}) | "
        f"RegScale: {reg_scale:.3e} (eff: {reg_scale_eff:.3e})\n"
        f"  Lower: {lower:.3e} | Lres: {res:.3e} | Ljump: {jump:.3e}{bc_str} | "
        f"Lrgrad: {rgrad:.3e} | Ljump_rgrad: {jump_rgrad:.3e}\n"
        f"  b0*: {b0_star:.2f} | int_u_hat: {integral_unit:.3e} | int_u: {u_int:.3e} | mean_D: {mean_d:.3e}"
    )


def format_bilo_pretrain_progress(
    step: int,
    total: float,
    anchor: float,
    lower: float,
    sup: float,
    res: float,
    jump: float,
    bc: float,
    rgrad: float,
    jump_rgrad: float,
    mean_d: float,
    bc_type: str = "dirichlet",
) -> str:
    """Format BiLO pretrain progress.

    Args:
        step: Current iteration
        total: Total pretrain loss (anchor + lower + sup)
        anchor: Anchor loss to initialization
        lower: Lower-level physics loss
        sup: Supervised loss to FDM solution
        res: PDE residual loss (unweighted)
        jump: Jump condition loss (unweighted)
        bc: Boundary condition loss (unweighted)
        rgrad: Residual gradient penalty (unweighted)
        jump_rgrad: Jump gradient penalty (unweighted)
        mean_d: Mean D value
        bc_type: Boundary condition type

    Returns:
        Formatted multi-line string for printing.
    """
    bc_str = f" | Lbc: {bc:.3e}" if bc_type == "neumann" else ""

    return (
        f"[BiLO|pretrain] Iter {step:05d} | Ltot: {total:.3e}\n"
        f"  Lanchor: {anchor:.3e} | Llower: {lower:.3e} | Lsup: {sup:.3e}\n"
        f"  Lres: {res:.3e} | Ljump: {jump:.3e} | "
        f"Lrgrad: {rgrad:.3e} | Ljump_rgrad: {jump_rgrad:.3e}{bc_str}\n"
        f"  mean_D: {mean_d:.3e}"
    )
