"""Notebook-friendly interface for inverse diffusion experiments.

Key surface:
- Problem: data + physics specification (includes Problem.synthetic()).
- Solution: method outputs plus plotting/metrics helpers.
- solve(): run DTO/PINN/BiLO with config overrides.
- DEFAULT_SETTINGS/get_default_settings/show_settings(): inspect and reuse defaults.
- compare_methods(): run multiple solvers on a shared problem.

Also includes D-profile builders and helper utilities for synthetic setups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional
import warnings

import numpy as np
import torch

from config import Config
from data import PPPData
import data as data_utils
import diagnostics, method_bilo, method_dto, method_pinn, physics


DEFAULT_SETTINGS = {
    # Training
    "max_iters": 10000,
    "pretrain_iters": 1000,
    "lr_d": 1e-4,
    "lr_lower": 1e-4,
    "optimizer": "adam",
    "lbfgs_lr": 1.0,
    "lbfgs_max_iter": 20,
    "use_scheduler": True,

    # Loss weights
    "w_data": 1.0,
    "w_phys": 1.0,
    "w_jump": 1.0,
    "w_resgrad": 0.01,
    "w_bc": 1.0,

    # Regularization
    "wreg_smooth": 1e-7,  # Smoothness on D
    "wreg_scale": 0.1,    # Scale anchor on D
    "smoothness_type": "h1",

    # Data
    "field_loss": "rle",

    # Source amplitude (b0) estimation
    "b0_fixed_value": None,  # If set, use fixed b0 instead of VarPro

    # Initialization
    "use_ddi": True,
    "scalar_fit_iters": 500,
    "pert_scale": 0.1,
    "pert_freq": 2.0,

    # Architecture
    "use_rff": True,
    "rff_scale": 1.0,
    "n_res": 201,

    # Early stopping
    "early_burnin": 2500,
    "early_patience": 500,
    "early_tol": 0.01,
    "log_every": 200,
}


def get_default_settings() -> Dict[str, Any]:
    """Return a copy of the default settings dict.

    Use this as a starting point, then override specific values:

        settings = get_default_settings()
        settings["lr_d"] = 1e-3
        settings["max_iters"] = 5000
        result = solve(problem, method="dto", **settings)

    Or use dict merge syntax:

        my_settings = {**get_default_settings(), "lr_d": 1e-3, "max_iters": 5000}
    """
    return dict(DEFAULT_SETTINGS)


def show_settings(settings: Optional[Dict[str, Any]] = None) -> None:
    """Print settings with parameter descriptions.

    Args:
        settings: Dict to display. If None, shows DEFAULT_SETTINGS.
    """
    if settings is None:
        settings = DEFAULT_SETTINGS
        print("DEFAULT_SETTINGS:")
    else:
        print("Current settings:")

    descriptions = {
        "max_iters": "Finetune iterations",
        "pretrain_iters": "Physics warmup (pinn/bilo)",
        "lr_d": "Learning rate for D(x) (both phases)",
        "lr_d_pre": "LR for D(x) pretrain (overrides lr_d)",
        "lr_d_fine": "LR for D(x) finetune (overrides lr_d)",
        "lr_lower": "Learning rate for u-net (both phases)",
        "lr_lower_pre": "LR for u-net pretrain (overrides lr_lower)",
        "lr_lower_fine": "LR for u-net finetune (overrides lr_lower)",
        "optimizer": "'adam' or 'lbfgs'",
        "lbfgs_lr": "LBFGS step size",
        "lbfgs_max_iter": "LBFGS inner iterations per step",
        "use_scheduler": "Cosine LR decay",
        "w_data": "Data loss weight (pinn only)",
        "w_phys": "Physics loss weight (pinn only)",
        "wreg_smooth": "Smoothness penalty on D",
        "wreg_scale": "Scale anchor on D",
        "smoothness_type": "'h1' or 'tv'",
        "w_jump": "Jump condition (pinn/bilo)",
        "w_resgrad": "Residual gradient (bilo only)",
        "w_bc": "Boundary condition loss (neumann only)",
        "field_loss": "'mse' or 'rle'",
        "b0_fixed_value": "Fixed b0 (None=VarPro)",
        "use_ddi": "DDI seed for scalar fit",
        "scalar_fit_iters": "Scalar-fit iterations (constant D)",
        "pert_scale": "Init perturbation amplitude",
        "pert_freq": "Init perturbation frequency",
        "use_rff": "Fourier features (pinn/bilo)",
        "rff_scale": "RFF freq multiplier (higher=sharper)",
        "n_res": "Solver grid points",
        "early_burnin": "Warmup before early stop",
        "early_patience": "Patience iterations",
        "early_tol": "Min improvement threshold",
        "log_every": "Log every N iters",
    }

    for key, value in settings.items():
        desc = descriptions.get(key, "")
        print(f"  {key}: {value!r:<12}  # {desc}")


def _finite_difference_derivative(
    d_callable: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Numerical derivative for profiles without a closed-form slope."""
    return (d_callable(x + eps) - d_callable(x - eps)) / (2.0 * eps)


def _parse_profile_params(
    d_profile: str,
    d_profile_params: Optional[tuple[float, float, float] | list[float]],
) -> tuple[float, float, float]:
    """Return (mean, amplitude, frequency) for parameterized D(x) profiles."""
    if d_profile_params is None:
        raise ValueError(
            f"d_profile_params=(mean, amplitude, frequency) is required for d_profile='{d_profile}'."
        )
    if len(d_profile_params) != 3:
        raise ValueError(
            "d_profile_params must be a 3-tuple/list: (mean, amplitude, frequency)."
        )
    mean, amplitude, frequency = d_profile_params
    return float(mean), float(amplitude), float(frequency)


def _build_d_profile(
    d_profile: Literal["sinusoidal", "steps", "custom"],
    d_profile_params: Optional[tuple[float, float, float] | list[float]],
    d_func: Optional[Callable[[np.ndarray], np.ndarray]],
) -> tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:
    """Construct D(x) and D'(x) callables for synthetic problem generation."""
    profile = d_profile.lower()
    if profile == "custom":
        if d_func is None:
            raise ValueError("d_func is required when d_profile='custom'.")
        if d_profile_params is not None:
            raise ValueError("d_profile_params must be None when d_profile='custom'.")

        def d_callable(x: np.ndarray) -> np.ndarray:
            return np.asarray(d_func(x))

        def dprime_callable(x: np.ndarray) -> np.ndarray:
            return _finite_difference_derivative(d_callable, x)

        return d_callable, dprime_callable

    mean, amplitude, frequency = _parse_profile_params(profile, d_profile_params)
    if profile == "sinusoidal":
        def d_callable(x: np.ndarray) -> np.ndarray:
            return mean + amplitude * np.sin(2.0 * np.pi * frequency * x)

        def dprime_callable(x: np.ndarray) -> np.ndarray:
            return (
                amplitude
                * (2.0 * np.pi * frequency)
                * np.cos(2.0 * np.pi * frequency * x)
            )

        return d_callable, dprime_callable

    if profile == "cos":
        def d_callable(x: np.ndarray) -> np.ndarray:
            return mean + amplitude * np.cos(2.0 * np.pi * frequency * x)

        def dprime_callable(x: np.ndarray) -> np.ndarray:
            return (
                -amplitude
                * (2.0 * np.pi * frequency)
                * np.sin(2.0 * np.pi * frequency * x)
            )

        return d_callable, dprime_callable

    if profile == "steps":
        # Random phase shift to avoid jumps landing exactly on grid points (e.g. source at 0.5)
        phase_shift = np.random.rand()

        def d_callable(x: np.ndarray) -> np.ndarray:
            phase = np.sin(2.0 * np.pi * frequency * (x + phase_shift))
            step = np.where(phase >= 0.0, 1.0, -1.0)
            return mean + amplitude * step

        def dprime_callable(x: np.ndarray) -> np.ndarray:
            return _finite_difference_derivative(d_callable, x)

        return d_callable, dprime_callable

    raise ValueError(f"Unsupported d_profile '{d_profile}'.")


@dataclass
class Problem:
    """Full specification of an inverse diffusion problem.

    Fields:
        x_grid: Observation grid (field mode) or PPP sampling grid.
        mode: "field" or "particles".
        u_field: Observed u(x) on x_grid (field mode).
        particles: PPPData with particle positions (particles mode).
        alpha/mu/source_location/b_true/bc_type: Physics parameters.
        d_true/u_true: Optional ground-truth arrays for synthetic problems.
        Solver grid is configured in solve() via config.grid.n_res.
    """

    x_grid: torch.Tensor
    mode: Literal["field", "particles"]
    u_field: Optional[torch.Tensor] = None
    particles: Optional[PPPData] = None
    alpha: float = 0.0
    mu: float = 5.0
    source_location: float = 0.5
    b_true: float = 100.0
    bc_type: str = "dirichlet"
    d_true: Optional[np.ndarray] = None
    u_true: Optional[np.ndarray] = None

    @classmethod
    def synthetic(
        cls,
        alpha: float,
        mode: Literal["field", "particles"],
        d_profile: Literal["sinusoidal", "steps", "custom"],
        d_profile_params: Optional[tuple[float, float, float] | list[float]],
        mu: float,
        b_true: float,
        m_obs: Optional[int] = None,
        d_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        source_location: float = 0.5,
        n_obs: int = 201,
        domain: tuple[float, float] = (0.0, 1.0),
        bc_type: str = "dirichlet",
        use_pde_sampling: bool = True,
        sde_tmax: float = 100.0,
        sde_dt: float = 1e-3,
        seed: int = 42,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
        verbose: bool = True,
    ) -> "Problem":
        """Generate a synthetic problem for testing and experimentation.

        Args:
            alpha: Diffusion convention (0=Ito, 0.5=Stratonovich, 1=Fickian).
            mode: "field" for dense observations or "particles" for PPP snapshots.
            d_profile: "sinusoidal", "steps", or "custom".
            d_profile_params: (mean, amplitude, frequency) for sinusoidal/steps.
                Steps uses a square-wave pattern with values mean ± amplitude.
                Note: "steps" includes a random phase shift (use `seed` to control).
                Use None when d_profile="custom".
            mu: Death rate in the PDE.
            b_true: True source strength for synthetic data.
            m_obs: Number of PPP snapshots. Required for particles.
            d_func: Custom D(x) profile for d_profile="custom".
            source_location: Location of the point source.
            n_obs: Number of observation grid points (field) or PPP sampling grid.
            domain: Spatial domain bounds.
            bc_type: Boundary condition type ("dirichlet" or "neumann").
            use_pde_sampling: Sample particles from PDE field instead of SDE (particles only).
            sde_tmax: SDE simulation horizon.
            sde_dt: SDE simulation time step.
            seed: Random seed.
            device: Torch device string.
            dtype: Torch dtype.
            verbose: Print summary statistics for debugging.
        """
        if mode == "particles" and m_obs is None:
            raise ValueError("m_obs is required when mode='particles'.")
        # Default to 0 for field mode if not provided, just to be safe with integers later
        m_obs_safe = m_obs if m_obs is not None else 0

        torch.manual_seed(seed)
        np.random.seed(seed)

        if n_obs < 3:
            raise ValueError("n_obs must be >= 3.")
        bc = bc_type.strip().lower()
        if bc not in {"dirichlet", "neumann"}:
            raise ValueError(f"Unsupported bc_type '{bc_type}'.")

        x_grid = torch.linspace(domain[0], domain[1], n_obs, device=device, dtype=dtype)
        x_field_np = x_grid.detach().cpu().numpy()

        d_callable, dprime_callable = _build_d_profile(
            d_profile,
            d_profile_params,
            d_func,
        )

        d_true = np.asarray(d_callable(x_field_np))
        u_true = physics.fdm_solve_alpha(
            d_true,
            alpha,
            mu,
            x_field_np,
            b_true,
            (source_location,),
            bc_type=bc,
        )

        u_obs = u_true if mode == "field" or use_pde_sampling else None

        if mode == "field":
            if u_obs is None:
                raise RuntimeError("Field observations require u_obs to be computed.")
            u_field = torch.tensor(u_obs, device=device, dtype=dtype).view(-1, 1)
            particles = None
        else:
            u_field = None
            rng = np.random.default_rng(seed)
            if use_pde_sampling:
                if u_obs is None:
                    raise RuntimeError("PDE sampling requires u_obs to be computed.")
                u_tensor = torch.tensor(u_obs, device=device, dtype=dtype).view(-1, 1)
                particles = data_utils.sample_ppp_from_field(
                    x_grid, u_tensor, m_obs_safe, rng, device=device, dtype=dtype
                )
            else:
                if verbose:
                    print(f"[Problem] Simulating particles (tmax={sde_tmax}, dt={sde_dt})...")
                particles = data_utils.simulate_particles_alpha(
                    d_callable,
                    dprime_callable,
                    z=source_location,
                    birth_rate=b_true,
                    death_rate=mu,
                    alpha=alpha,
                    tmax=sde_tmax,
                    dt=sde_dt,
                    rng=rng,
                    m_obs=m_obs_safe,
                    device=device,
                    dtype=dtype,
                    bc_type=bc,
                )

        if verbose:
            mean_d_true = float(np.mean(d_true))
            if mode == "field":
                u_integral = float(np.trapz(u_obs, x_field_np)) if u_obs is not None else 0.0
                print(f"[Problem] Field: ∫u {u_integral:.3e} | ⟨D⟩_true: {mean_d_true:.3e}")
            else:
                n_obs = particles.n_obs if particles is not None else 0
                m_obs = particles.m_obs if particles is not None else m_obs_safe
                avg_obs = n_obs / max(m_obs, 1)
                print(
                    f"[Problem] Particles: n_obs {n_obs} | m_obs {m_obs} | ⟨n⟩/obs {avg_obs:.2f} | ⟨D⟩_true: {mean_d_true:.3e}"
                )

        return cls(
            x_grid=x_grid,
            mode=mode,
            u_field=u_field,
            particles=particles,
            alpha=alpha,
            mu=mu,
            source_location=source_location,
            b_true=b_true,
            bc_type=bc,
            d_true=d_true,
            u_true=u_true,
        )

    @classmethod
    def from_observations(
        cls,
        x_grid: torch.Tensor,
        observations: torch.Tensor | PPPData,
        alpha: float,
        mu: float,
        source_location: float,
        b_true: Optional[float] = None,
        bc_type: str = "dirichlet",
    ) -> "Problem":
        """Create a problem from external observations.

        Args:
            x_grid: Observation grid as a torch tensor.
            observations: Field tensor for "field" mode or PPPData for "particles".
            alpha: Diffusion convention (0=Ito, 0.5=Stratonovich, 1=Fickian).
            mu: Death rate in the PDE.
            source_location: Location of the point source.
            b_true: Optional true source strength for reference.
            bc_type: Boundary condition type ("dirichlet" or "neumann").
        """
        bc = bc_type.strip().lower()
        if bc not in {"dirichlet", "neumann"}:
            raise ValueError(f"Unsupported bc_type '{bc_type}'.")
        if isinstance(observations, PPPData):
            return cls(
                x_grid=x_grid,
                mode="particles",
                particles=observations,
                alpha=alpha,
                mu=mu,
                source_location=source_location,
                b_true=b_true if b_true is not None else 100.0,
                bc_type=bc,
            )
        u_field = observations.view(-1, 1) if observations.ndim == 1 else observations
        return cls(
            x_grid=x_grid,
            mode="field",
            u_field=u_field,
            alpha=alpha,
            mu=mu,
            source_location=source_location,
            b_true=b_true if b_true is not None else 100.0,
            bc_type=bc,
        )


@dataclass
class Solution:
    """Result of solving an inverse diffusion problem.

    Fields:
        method: Solver name ("DTO", "PINN", "BILO").
        x_res: Solver grid used for physics enforcement (1D).
        d_pred: Predicted D(x) on the solver grid (1D).
        u_pred: Predicted field on the solver grid (1D).
        b0_star: Projected source amplitude.
        history: Loss curves and diagnostics from training (keys vary by method).
        weights: Loss weights used (useful for plotting history).
        d_net: Optional trained D network (BiLO/PINN).
        local_op: Optional trained local operator network (BiLO/PINN).
        _raw_result: Method-specific result dataclass with full outputs
            (DTOResult/PINNResult/BiLOResult).
    """

    method: str
    x_res: torch.Tensor
    d_pred: torch.Tensor
    u_pred: torch.Tensor
    b0_star: float
    history: Dict[str, List[float]]
    weights: Dict[str, float] = field(default_factory=dict)
    d_net: Optional[Any] = None
    local_op: Optional[Any] = None
    _raw_result: Any = None

    def plot(
        self,
        problem: Optional[Problem] = None,
        show: bool = True,
        figsize: tuple[int, int] = (10, 4),
        ax: Optional[tuple[Any, Any]] = None,
    ):
        """Visualize the solution and (optionally) ground truth.

        Pass ax=(ax0, ax1) to draw into existing matplotlib axes.
        """
        import matplotlib.pyplot as plt

        x_res_np = self._get_x_array()
        x_obs_np = x_res_np
        if problem is not None:
            x_obs_np = problem.x_grid.detach().cpu().view(-1).numpy()
        d_np = self._to_numpy(self.d_pred).reshape(-1)
        u_np = self._to_numpy(self.u_pred).reshape(-1)
        u_fdm = None
        if problem is not None and self.method in {"PINN", "BILO"}:
            try:
                u_fdm = physics.fdm_solve_alpha(
                    d_np,
                    problem.alpha,
                    problem.mu,
                    x_res_np,
                    self.b0_star,
                    (problem.source_location,),
                    bc_type=problem.bc_type,
                )
            except ValueError as e:
                print(f"Warning: FDM solve failed ({e}). Skipping FDM plot.")
                u_fdm = None
                pass
                
                
        plot_particles = (
            problem is not None
            and problem.mode == "particles"
            and problem.particles is not None
        )

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
        else:
            axes = ax
            if len(axes) != 2:
                raise ValueError("ax must be a 2-tuple of matplotlib axes.")
            fig = axes[0].figure

        axes[0].plot(x_res_np, d_np, "b-", linewidth=2, label=self.method)
        if problem is not None and problem.d_true is not None:
            axes[0].plot(x_obs_np, problem.d_true, "k--", linewidth=1.5, label="true")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("D(x)")
        axes[0].set_title("Diffusivity")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if plot_particles:
            x_particles = problem.particles.x_particles.detach().cpu().numpy().reshape(-1)
            if x_particles.size > 0:
                m_obs = max(int(problem.particles.m_obs), 1)
                avg_count = float(x_particles.size) / m_obs
                hist, edges = np.histogram(
                    x_particles,
                    bins=50,
                    range=(x_res_np[0], x_res_np[-1]),
                    density=True,
                )
                bin_width = float(edges[1] - edges[0])
                centers = edges[:-1] + 0.5 * bin_width
                axes[1].bar(
                    centers,
                    hist * avg_count,
                    width=bin_width,
                    color="gray",
                    alpha=0.3,
                    label="samples (avg/obs)",
                    align="center",
                )

            axes[1].plot(x_res_np, u_np, "b-", linewidth=2, label=self.method)
            if u_fdm is not None:
                axes[1].plot(x_res_np, u_fdm, "r--", linewidth=1.5, label="FDM (D_pred)")
            if problem is not None and problem.u_true is not None:
                axes[1].plot(x_obs_np, problem.u_true, "k--", linewidth=1.5, label="true")
        else:
            axes[1].plot(x_res_np, u_np, "b-", linewidth=2, label=self.method)
            if u_fdm is not None:
                axes[1].plot(x_res_np, u_fdm, "r--", linewidth=1.5, label="FDM (D_pred)")
            if problem is not None and problem.u_true is not None:
                axes[1].plot(x_obs_np, problem.u_true, "k--", linewidth=1.5, label="true")
        axes[1].set_ylabel("u(x)")
        axes[1].set_title(f"Field (b0*={self.b0_star:.2f})")
        axes[1].set_xlabel("x")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def metrics(self, problem: Problem) -> Dict[str, float]:
        """Compute error metrics against ground truth."""
        if problem.d_true is None:
            raise ValueError("Problem has no d_true for comparison.")

        x_res = self.x_res.detach().cpu().numpy().reshape(-1)
        x_obs = problem.x_grid.detach().cpu().numpy().reshape(-1)
        d_pred = self._to_numpy(self.d_pred).reshape(-1)
        d_true = np.asarray(problem.d_true).reshape(-1)
        same_grid = d_pred.shape == d_true.shape and np.allclose(x_res, x_obs)
        d_pred_eval = d_pred if same_grid else np.interp(x_obs, x_res, d_pred)

        u_true = None
        u_pred_eval = None
        if problem.u_true is not None:
            u_pred = self._to_numpy(self.u_pred).reshape(-1)
            u_true = np.asarray(problem.u_true).reshape(-1)
            u_pred_eval = u_pred if same_grid else np.interp(x_obs, x_res, u_pred)

        return diagnostics.compute_solution_metrics(
            x_obs,
            d_true,
            d_pred_eval,
            u_true=u_true,
            u_pred=u_pred_eval,
            b0_star=self.b0_star,
            b_true=problem.b_true,
        )

    @staticmethod
    def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _get_x_array(self, problem: Optional[Problem] = None) -> np.ndarray:
        return self.x_res.detach().cpu().numpy().reshape(-1)


def solve(
    problem: Problem,
    method: Literal["dto", "pinn", "bilo"] = "bilo",
    *,
    max_iters: Optional[int] = None,
    pretrain_iters: Optional[int] = None,
    n_res: Optional[int] = None,
    lr_d: Optional[float] = None,
    lr_d_pre: Optional[float] = None,
    lr_d_fine: Optional[float] = None,
    lr_lower: Optional[float] = None,
    lr_lower_pre: Optional[float] = None,
    lr_lower_fine: Optional[float] = None,
    optimizer: Optional[Literal["adam", "lbfgs"]] = None,
    lbfgs_lr: Optional[float] = None,
    lbfgs_max_iter: Optional[int] = None,
    w_data: Optional[float] = None,
    w_phys: Optional[float] = None,
    wreg_smooth: Optional[float] = None,
    wreg_scale: Optional[float] = None,
    w_jump: Optional[float] = None,
    w_resgrad: Optional[float] = None,
    w_bc: Optional[float] = None,
    smoothness_type: Optional[Literal["h1", "tv"]] = None,
    field_loss: Optional[Literal["mse", "rle"]] = None,
    b0_fixed_value: Optional[float] = None,
    use_ddi: Optional[bool] = None,
    scalar_fit_iters: Optional[int] = None,
    pert_scale: Optional[float] = None,
    pert_freq: Optional[float] = None,
    use_scheduler: Optional[bool] = None,
    use_rff: Optional[bool] = None,
    rff_scale: Optional[float] = None,
    early_burnin: Optional[int] = None,
    early_patience: Optional[int] = None,
    early_tol: Optional[float] = None,
    log_every: Optional[int] = None,
    config: Optional[Config] = None,
    verbose: bool = True,
) -> Solution:
    """Solve an inverse diffusion problem.

    Args:
        problem: Problem definition with observations and physics parameters.
        method: "dto", "pinn", or "bilo".
        max_iters: Main training iterations (finetune phase).
        pretrain_iters: Warmup iterations for physics/anchor steps.
        n_res: Number of solver grid points for D/u (overrides config.grid.n_res).
        lr_d: Learning rate for D(x) parameters (sets both pretrain + finetune).
        lr_d_pre: Learning rate for D(x) during pretrain (overrides lr_d for pretrain).
        lr_d_fine: Learning rate for D(x) during finetune (overrides lr_d for finetune).
        lr_lower: Learning rate for u(x) / physics network (sets both pretrain + finetune).
        lr_lower_pre: LR for physics network during pretrain (overrides lr_lower).
        lr_lower_fine: LR for physics network during finetune (overrides lr_lower).
        optimizer: "adam" or "lbfgs" (finetune optimizer).
        lbfgs_lr: LBFGS step size (default 1.0).
        lbfgs_max_iter: LBFGS inner iterations per step (default 20).
        w_data: Data loss weight (PINN only).
        w_phys: Physics loss weight (PINN only).
        wreg_smooth: Smoothness penalty on D(x).
        wreg_scale: Scale anchor weight on D(x).
        w_jump: Jump condition weight (PINN/BiLO).
        w_resgrad: Residual gradient penalty (BiLO).
        w_bc: Boundary condition penalty (Neumann only).
        smoothness_type: Smoothness penalty selector ("h1" or "tv").
        field_loss: "mse" or "rle" for field observations.
        b0_fixed_value: If set to a positive value, use this fixed source amplitude
            instead of VarPro projection. Useful when b0 is known a priori.
        use_ddi: Enable DDI seed for scalar fit.
        scalar_fit_iters: Iterations for constant-D scalar fit.
        pert_scale: Amplitude of initial D(x) wiggles.
        pert_freq: Frequency of initial D(x) wiggles.
        use_scheduler: Enable cosine LR scheduling.
        use_rff: Enable random Fourier features in networks.
        rff_scale: RFF frequency multiplier (higher values allow sharper features).
        early_burnin: Iterations before early stopping activates.
        early_patience: Iterations without improvement before stop.
        early_tol: Minimum relative improvement to reset patience.
        log_every: Record history every N iterations.
        config: Optional Config object for advanced overrides.
        verbose: Print progress during training.

    Note:
        Arguments set to None default to `DEFAULT_SETTINGS` (which mirrors Config
        defaults). Use `get_default_settings()` or `show_settings()` to inspect
        and copy the current defaults.
        Advanced settings not listed here (e.g. `rff_width`, `n_int`, `ddi_bounds`)
        can be modified by passing a custom `Config` object.
        The solver grid is built from `config.grid.n_res` over the problem domain.

    Returns:
        Solution with the main predictions and training history. Use
        `solution._raw_result` to access method-specific outputs like d_pred,
        u_hat_unit, or the trained networks.
    """
    if problem.mode not in {"field", "particles"}:
        raise ValueError(f"Unsupported mode '{problem.mode}'.")
    if problem.mode == "field" and problem.u_field is None:
        raise ValueError("Field mode requires u_field observations.")
    if problem.mode == "particles" and problem.particles is None:
        raise ValueError("Particle mode requires PPPData observations.")

    if config is None:
        config = Config()

    if n_res is not None:
        config.grid.n_res = n_res

    x_grid_flat = problem.x_grid.view(-1)
    domain = (float(x_grid_flat[0].item()), float(x_grid_flat[-1].item()))

    method_lower = method.lower()
    if method_lower in ("pinn", "bilo"):
        # PINN/BiLO: use aligned grid so z is exactly a grid point
        x_res = physics.build_aligned_grid(
            domain,
            int(config.grid.n_res),
            problem.source_location,
            problem.x_grid.device,
            problem.x_grid.dtype,
        )
    else:
        # DTO: use uniform grid (hat delta handles arbitrary z)
        x_res = torch.linspace(
            domain[0],
            domain[1],
            int(config.grid.n_res),
            device=problem.x_grid.device,
            dtype=problem.x_grid.dtype,
        )

    config.physics.alpha = problem.alpha
    config.physics.mu = problem.mu
    config.physics.sources = (problem.source_location,)
    config.physics.b_true = problem.b_true
    config.physics.domain = domain
    config.physics.bc_type = problem.bc_type
    config.data.mode = problem.mode
    config.run.device = str(x_res.device)
    config.run.dtype = x_res.dtype

    if max_iters is not None:
        config.train.finetune_iters = max_iters
    if pretrain_iters is not None:
        config.train.pretrain_iters = pretrain_iters
    # Learning rates: general param sets both, specific params override individually
    if lr_d is not None:
        config.train.lr_d_fine = lr_d
        config.train.lr_d_pre = lr_d
    if lr_d_pre is not None:
        config.train.lr_d_pre = lr_d_pre
    if lr_d_fine is not None:
        config.train.lr_d_fine = lr_d_fine
    if lr_lower is not None:
        config.train.lr_lower_fine = lr_lower
        config.train.lr_lower_pre = lr_lower
    if lr_lower_pre is not None:
        config.train.lr_lower_pre = lr_lower_pre
    if lr_lower_fine is not None:
        config.train.lr_lower_fine = lr_lower_fine
    if optimizer is not None:
        config.train.optimizer = optimizer
    if lbfgs_lr is not None:
        config.train.lbfgs_lr = lbfgs_lr
    if lbfgs_max_iter is not None:
        config.train.lbfgs_max_iter = lbfgs_max_iter
    if w_data is not None:
        config.reg.w_data = w_data
    if w_phys is not None:
        config.reg.w_phys = w_phys
    if wreg_smooth is not None:
        config.reg.wreg_smooth = wreg_smooth
    if wreg_scale is not None:
        config.reg.wreg_scale = wreg_scale
    if w_jump is not None:
        config.reg.w_jump = w_jump
    if w_resgrad is not None:
        config.reg.w_resgrad = w_resgrad
    if w_bc is not None:
        config.reg.w_bc = w_bc
    if smoothness_type is not None:
        config.reg.smoothness_type = smoothness_type
    if field_loss is not None:
        config.data.field_loss = field_loss
    if b0_fixed_value is not None:
        config.data.b0_fixed_value = b0_fixed_value
    if use_ddi is not None:
        config.d_profile.use_ddi = use_ddi
    if scalar_fit_iters is not None:
        config.train.scalar_fit_iters = scalar_fit_iters
    if pert_scale is not None:
        config.d_profile.pert_scale = pert_scale
    if pert_freq is not None:
        config.d_profile.pert_freq = pert_freq
    if use_scheduler is not None:
        config.train.use_scheduler = use_scheduler
    if use_rff is not None:
        config.arch.use_rff = use_rff
    if rff_scale is not None:
        config.arch.rff_scale = rff_scale
    if early_burnin is not None:
        config.train.early_burnin = early_burnin
    if early_patience is not None:
        config.train.early_patience = early_patience
    if early_tol is not None:
        config.train.early_tol = early_tol
    if log_every is not None:
        config.train.log_every = log_every

    config.validate()

    def _warn_irrelevant(name: str, value: Any, message: str) -> None:
        if value is None:
            return
        default_value = DEFAULT_SETTINGS.get(name, None)
        if default_value is not None and value == default_value:
            return
        warnings.warn(message, stacklevel=2)

    if method_lower == "dto":
        _warn_irrelevant("pretrain_iters", pretrain_iters, "'pretrain_iters' is ignored for DTO.")
        _warn_irrelevant("lr_lower", lr_lower, "'lr_lower' is ignored for DTO.")
        _warn_irrelevant("w_data", w_data, "'w_data' is ignored for DTO.")
        _warn_irrelevant("w_phys", w_phys, "'w_phys' is ignored for DTO.")
        _warn_irrelevant("w_jump", w_jump, "'w_jump' is ignored for DTO.")
        _warn_irrelevant("w_resgrad", w_resgrad, "'w_resgrad' is ignored for DTO.")
        _warn_irrelevant("w_bc", w_bc, "'w_bc' is ignored for DTO.")
        _warn_irrelevant("use_rff", use_rff, "'use_rff' is ignored for DTO.")
        _warn_irrelevant("rff_scale", rff_scale, "'rff_scale' is ignored for DTO.")
    elif method_lower == "pinn":
        _warn_irrelevant("w_resgrad", w_resgrad, "'w_resgrad' is ignored for PINN.")
    elif method_lower == "bilo":
        _warn_irrelevant("w_data", w_data, "'w_data' is ignored for BiLO.")
        _warn_irrelevant("w_phys", w_phys, "'w_phys' is ignored for BiLO.")

    if verbose:
        print(f"[Solve] Method: {method.upper()} | α: {problem.alpha} | Mode: {problem.mode}")

    if problem.mode == "field":
        if problem.u_field is None:
            raise ValueError("Field mode requires u_field observations.")
        x_grid_flat = problem.x_grid.detach().cpu().view(-1)
        if problem.u_field.numel() != x_grid_flat.numel():
            raise ValueError("Field mode requires u_field to align with x_grid.")

    if method_lower == "dto":
        data_bundle = method_dto.DTOData(
            mode=problem.mode,
            x_res=x_res,
            x_field=problem.x_grid,
            u_true=problem.u_field,
            ppp=problem.particles,
        )
        result = method_dto.fit(data_bundle, config, verbose=verbose)
        d_net = None
        local_op = None
    elif method_lower == "pinn":
        data_bundle = method_pinn.PINNData(
            mode=problem.mode,
            x_res=x_res,
            x_field=problem.x_grid,
            u_true=problem.u_field,
            ppp=problem.particles,
        )
        result = method_pinn.fit(data_bundle, config, verbose=verbose)
        d_net = None
        local_op = None
    elif method_lower == "bilo":
        d_true_res = None
        if problem.d_true is not None:
             x_obs_np = problem.x_grid.detach().cpu().numpy().reshape(-1)
             x_res_np = x_res.detach().cpu().numpy().reshape(-1)
             d_true_obs = problem.d_true.reshape(-1)
             if x_obs_np.shape == x_res_np.shape and np.allclose(x_obs_np, x_res_np):
                 d_true_res_np = d_true_obs
             else:
                 d_true_res_np = np.interp(x_res_np, x_obs_np, d_true_obs)
             d_true_res = torch.from_numpy(d_true_res_np).to(device=x_res.device, dtype=x_res.dtype)

        data_bundle = method_bilo.BiLOData(
            mode=problem.mode,
            x_res=x_res,
            x_field=problem.x_grid,
            u_true=problem.u_field,
            ppp=problem.particles,
            d_true=d_true_res,
        )
        result = method_bilo.fit(data_bundle, config, verbose=verbose)
        d_net = getattr(result, "d_net", None)
        local_op = getattr(result, "local_op", None)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from dto, pinn, bilo.")

    if verbose:
        print(f"[Solve] Done | b₀*: {result.b0_star:.4f}")

    # Capture weights for plotting effective losses
    # For PINN: phys = w_phys*res + w_jump*jump (already weighted), so phys weight is 1.0
    # For BiLO: lower = res + w_jump*jump + w_resgrad*rgrad (res has implicit weight 1.0)
    weights = {
        "data": config.reg.w_data,
        "phys": 1.0,  # phys is already the weighted sum in PINN
        "res": config.reg.w_phys,  # PINN: w_phys weights res; BiLO: implicit 1.0 (close enough)
        "reg_smooth": config.reg.wreg_smooth,
        "reg_scale": config.reg.wreg_scale,
        "jump": config.reg.w_jump,
        "rgrad": config.reg.w_resgrad,
        "jump_rgrad": config.reg.w_resgrad,
        "bc": config.reg.w_bc,
    }

    return Solution(
        method=method.upper(),
        x_res=result.x_res,
        d_pred=result.d_pred,
        u_pred=result.u_pred,
        b0_star=float(result.b0_star),
        history=result.history,
        weights=weights,
        d_net=d_net,
        local_op=local_op,
        _raw_result=result,
    )


def compare_methods(
    problem: Problem,
    methods: Optional[List[str]] = None,
    **solve_kwargs,
) -> Dict[str, Solution]:
    """Run multiple methods on the same problem for comparison.

    Args:
        problem: Problem definition to solve.
        methods: List of method names to run (default: dto/pinn/bilo).
        **solve_kwargs: Keyword arguments forwarded to solve().

    Returns:
        Dict mapping method names (uppercased) to Solution objects.
    """
    if methods is None:
        methods = ["dto", "pinn", "bilo"]

    results: Dict[str, Solution] = {}
    for method in methods:
        results[method.upper()] = solve(problem, method=method, **solve_kwargs)
    return results
