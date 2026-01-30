"""Configuration dataclasses and helpers for experiments and training.

Key structures:
- PhysicsConfig, DProfileConfig, DataConfig, GridConfig, TrainConfig, RegConfig,
  ArchConfig, RunConfig: validated config sections with defaults.
- Config: top-level container with validate(), to_dict(), and from_dict().

Also includes small normalization helpers for common string fields.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Literal, Tuple

import torch


def _normalize_field_loss(value: str) -> str:
    """Normalize the field loss selector string."""
    return value.strip().lower()


def _normalize_data_mode(value: str) -> str:
    """Normalize the data mode selector string."""
    return value.strip().lower()


@dataclass
class PhysicsConfig:
    """Physical parameters for the alpha-PDE."""

    alpha: float = 0.0
    mu: float = 5.0  # Nondimensional default is mu=1; override for testing.
    domain: Tuple[float, float] = (0.0, 1.0)  # Assumed nondimensionalized to [0, 1]
    sources: Tuple[float, ...] = (0.5,)
    b_true: float = 100.0
    bc_type: str = "dirichlet"  # "dirichlet" (u=0) or "neumann" (zero flux)

    def validate(self) -> None:
        """Validate physics parameters for consistency."""
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if len(self.domain) != 2 or self.domain[0] >= self.domain[1]:
            raise ValueError(f"domain must be (min, max) with min < max, got {self.domain}")
        if any(z <= self.domain[0] or z >= self.domain[1] for z in self.sources):
            raise ValueError(f"source locations must lie strictly within domain, got {self.sources}")
        self.bc_type = self.bc_type.strip().lower()
        if self.bc_type not in {"dirichlet", "neumann"}:
            raise ValueError(
                f"unsupported bc_type '{self.bc_type}' (use 'dirichlet' or 'neumann')"
            )


@dataclass
class DProfileConfig:
    """Initialization and data-driven scaling settings for D(x).

    The pert_scale and pert_freq parameters control the initial "wiggles" in D(x):
        D_init(x) = d_scale * (1 + pert_scale * sin(2π * pert_freq * x))

    These wiggles serve different purposes for different methods:
    - BiLO/PINN: Needed during pretrain so the LocalOperator learns ∇D sensitivity.
      Without wiggles, the operator may become "blind" to D variations.
    - DTO: Does NOT need wiggles since there's no neural operator to train. Consider
      setting pert_scale=0 for DTO to start from a constant initialization.

    NOTE: d_init_base was removed in favor of automatic scale estimation via
    scale_estimation.fit_constant_d(). The scalar fit finds the optimal constant D
    directly from the data, eliminating the need for manual scale tuning.
    """

    pert_scale: float = 0.1  # Relative amplitude of initial wiggles
    pert_freq: float = 2.0   # Number of oscillations across the domain
    use_ddi: bool = True     # Use DDI as starting point for scalar fit
    ddi_d_min: float = 1e-4  # Lower clamp for DDI estimate
    ddi_d_max: float = 10.0  # Upper clamp for DDI estimate

    # Synthetic problem generation
    profile_type: Literal["sinusoidal", "steps"] = "sinusoidal"
    params: Tuple[float, float, float] = (0.1, 0.04, 4.0) # mean, amplitude, frequency

    def validate(self) -> None:
        """Validate diffusion profile settings."""
        if self.pert_scale >= 1.0:
            raise ValueError("pert_scale must be < 1 to keep D_init positive.")
        if self.ddi_d_min > self.ddi_d_max:
            raise ValueError("ddi_d_min must be <= ddi_d_max.")
        if self.profile_type not in {"sinusoidal", "steps"}:
            raise ValueError("profile_type must be 'sinusoidal' or 'steps'.")
        if len(self.params) != 3:
            raise ValueError("params must be a 3-tuple (mean, amplitude, frequency).")


@dataclass
class DataConfig:
    """Data-generation configuration for field or particle modes.

    Attributes:
        mode: Observation type - "field" for dense u(x) measurements or "particles"
            for Poisson Point Process snapshots.
        field_loss: Loss function for field data - "mse" or "rle" (relative log error).
        m_obs: Number of particle snapshots (particles mode only).
        b0_fixed_value: If set to a positive value, use this fixed source amplitude
            instead of estimating via VarPro projection. If None (default), b0 is
            estimated via Variable Projection at each iteration. This is useful when
            the source amplitude is known a priori (e.g., from experimental
            calibration), eliminating the amplitude-diffusivity ambiguity.
    """

    mode: str = "field"  # "field" or "particles"
    field_loss: str = "rle"  # "mse" or "rle"
    m_obs: int = 250
    b0_fixed_value: float | None = None  # If set, use fixed b0 instead of VarPro

    def validate(self) -> None:
        """Normalize and validate data settings."""
        self.mode = _normalize_data_mode(self.mode)
        self.field_loss = _normalize_field_loss(self.field_loss)
        if self.mode not in {"field", "particles"}:
            raise ValueError(f"mode must be 'field' or 'particles', got {self.mode}")
        if self.field_loss not in {"mse", "rle"}:
            raise ValueError(f"field_loss must be 'mse' or 'rle', got {self.field_loss}")
        # Validate b0_fixed_value if provided
        if self.b0_fixed_value is not None and self.b0_fixed_value <= 0:
            raise ValueError(f"b0_fixed_value must be positive, got {self.b0_fixed_value}")


@dataclass
class GridConfig:
    """Spatial grid configuration."""

    n_res: int = 201
    n_int: int = 1001

    def validate(self) -> None:
        """Validate grid sizes."""
        return


@dataclass
class TrainConfig:
    """Optimizer and training loop parameters."""

    bilo_load_path: str | None = None
    bilo_save_path: str | None = None
    lower_tol: float | None = None
    scalar_fit_iters: int = 500
    pretrain_iters: int = 1000
    finetune_iters: int = 10000
    lr_d_pre: float = 1e-4
    lr_lower_pre: float = 1e-4
    lr_d_fine: float = 1e-4
    lr_lower_fine: float = 1e-4
    optimizer: Literal["adam", "lbfgs"] = "adam"
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 20
    use_scheduler: bool = True
    early_burnin: int = 2500
    early_patience: int = 500
    early_tol: float = 1e-2  # Relative improvement threshold for early stopping.
    log_every: int = 200

    def validate(self) -> None:
        """Validate training loop settings."""
        if self.bilo_load_path is not None and not str(self.bilo_load_path).strip():
            raise ValueError("bilo_load_path must be a non-empty string if provided.")
        if self.bilo_save_path is not None and not str(self.bilo_save_path).strip():
            raise ValueError("bilo_save_path must be a non-empty string if provided.")
        if self.lower_tol is not None and self.lower_tol < 0.0:
            raise ValueError("lower_tol must be non-negative if provided.")
        if self.scalar_fit_iters < 0:
            raise ValueError("scalar_fit_iters must be >= 0.")
        if self.optimizer not in {"adam", "lbfgs"}:
            raise ValueError("optimizer must be 'adam' or 'lbfgs'.")
        if self.lbfgs_max_iter <= 0:
            raise ValueError("lbfgs_max_iter must be > 0.")
        return


@dataclass
class RegConfig:
    """Weights for data/physics objectives.

    Regularization notes:
    - wreg_smooth: Penalizes fluctuations in log(D) to ensure scale invariance.
    - wreg_scale: Anchors log(D) to the data-driven initialization estimate.
    - w_bc: Boundary-condition penalty for Neumann BCs (ignored for Dirichlet).
    """

    w_data: float = 1.0
    w_phys: float = 1.0
    w_jump: float = 1.0
    w_bc: float = 1.0
    wreg_d_neumann: float = 0.0
    w_resgrad: float = 0.01
    wreg_smooth: float = 1e-7
    wreg_scale: float = 0.1
    lower_data: float | None = None
    smoothness_type: Literal["h1", "tv"] = "h1"

    def validate(self) -> None:
        """Validate regularization weights."""
        if self.smoothness_type not in {"h1", "tv"}:
            raise ValueError("smoothness_type must be 'h1' or 'tv'.")
        if self.w_bc < 0.0:
            raise ValueError("w_bc must be non-negative.")


@dataclass
class ArchConfig:
    """Neural architecture options for RFF embeddings."""

    use_rff: bool = True  # Enable Random Fourier Features for all networks
    d_min: float = 1e-6
    rff_seed: int = 0
    
    # Architecture selection
    d_net_arch: Literal["mlp", "pirate", "mmlp", "siren", "fourier", "grid"] = "mlp"
    d_net_depth: int = 2
    d_net_width: int = 128
    d_net_rff_scale: float = 1.0  # Frequency multiplier for RFF (higher = sharper features)
    siren_omega0: float = 30.0
    fix_endpoint: bool = False
    bilo_order: int = 1  # Highest order of derivative of D to use (0=D only, 1=D,D', 2=D,D',D'')

    u_net_arch: Literal["mlp", "pirate", "mmlp", "siren", "fourier", "grid"] = "mlp"
    u_net_depth: int = 3
    u_net_width: int = 128

    def validate(self) -> None:
        """Validate architecture settings."""
        return


@dataclass
class RunConfig:
    """Runtime settings such as device, dtype, and output directory."""

    seed: int = 42
    device: str = "cpu"
    dtype: str = "float64"
    outdir: str = "runs"
    name: str | None = None
    group: str | None = None

    def validate(self) -> None:
        """Validate run configuration."""
        if not self.device:
            raise ValueError("device must be a non-empty string.")
        if not self.dtype:
            raise ValueError("dtype must be a non-empty string.")

    @property
    def torch_device(self) -> torch.device:
        """Return a torch.device from the configured device string."""
        return torch.device(self.device)

    @property
    def torch_dtype(self) -> torch.dtype:
        """Map the configured dtype string to a torch dtype."""
        if isinstance(self.dtype, torch.dtype):
            return self.dtype
        dtype_str = str(self.dtype).lower()
        mapping = {
            "float32": torch.float32,
            "float": torch.float32,
            "float64": torch.float64,
            "double": torch.float64,
            "float16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype_str not in mapping:
            raise ValueError(f"Unsupported dtype '{self.dtype}'.")
        return mapping[dtype_str]


@dataclass
class SolverConfig:
    """Method-specific solver configuration."""
    method: Literal["dto", "pinn", "bilo"] = "dto"

    def validate(self) -> None:
        if self.method not in {"dto", "pinn", "bilo"}:
            raise ValueError(f"Unknown method '{self.method}'")


@dataclass
class WandBConfig:
    """WandB logging configuration."""
    enabled: bool = False
    project: str = "AlphaDiffusivityNet"
    entity: str | None = None
    group: str | None = None
    name: str | None = None
    tags: Tuple[str, ...] = ()

    def validate(self) -> None:
        pass


@dataclass
class Config:
    """Top-level configuration object with nested sections."""

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    d_profile: DProfileConfig = field(default_factory=DProfileConfig)
    data: DataConfig = field(default_factory=DataConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    reg: RegConfig = field(default_factory=RegConfig)
    arch: ArchConfig = field(default_factory=ArchConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def __post_init__(self) -> None:
        """Validate nested configs after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate all nested configuration sections."""
        self.physics.validate()
        self.d_profile.validate()
        self.data.validate()
        self.grid.validate()
        self.train.validate()
        self.reg.validate()
        self.arch.validate()
        self.solver.validate()
        self.wandb.validate()
        self.run.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the configuration to a JSON-friendly dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Load a Config from a nested dict of config sections."""
        if not isinstance(data, dict):
            raise TypeError("Config.from_dict expects a dict.")

        nested_keys = {"physics", "d_profile", "data", "grid", "train", "reg", "arch", "solver", "wandb", "run"}
        if any(k in data for k in nested_keys):
            return cls.from_nested_dict(data)
        raise ValueError("Expected nested config dict; legacy flat configs are no longer supported.")

    @classmethod
    def from_nested_dict(cls, data: Dict[str, Any]) -> "Config":
        """Build a Config from nested configuration sections."""
        d_profile_data = dict(data.get("d_profile", {}))
        d_profile_aliases = {
            "d_init_pert_scale": "pert_scale",
            "d_init_pert_freq": "pert_freq",
        }
        for old_key, new_key in d_profile_aliases.items():
            if old_key in d_profile_data and new_key not in d_profile_data:
                d_profile_data[new_key] = d_profile_data.pop(old_key)
        d_profile_data.pop("d_init_base", None)

        reg_data = dict(data.get("reg", {}))
        reg_aliases = {
            "w_jump": "w_jump",
            "wreg_jump": "w_jump",
            "w_rgrad": "w_resgrad",
            "wreg_rgrad": "w_resgrad",
            "w_resgrad": "w_resgrad",
            "w_reg_smooth": "wreg_smooth",
            "w_reg_mean": "wreg_scale",
            "wreg_mean": "wreg_scale",
            "w_reg_scale": "wreg_scale",
            "wreg_scale": "wreg_scale",
        }
        for old_key, new_key in reg_aliases.items():
            if old_key in reg_data and new_key not in reg_data:
                reg_data[new_key] = reg_data.pop(old_key)
        return cls(
            physics=PhysicsConfig(**data.get("physics", {})),
            d_profile=DProfileConfig(**d_profile_data),
            data=DataConfig(**data.get("data", {})),
            grid=GridConfig(**data.get("grid", {})),
            train=TrainConfig(**data.get("train", {})),
            reg=RegConfig(**reg_data),
            arch=ArchConfig(**data.get("arch", {})),
            solver=SolverConfig(**data.get("solver", {})),
            wandb=WandBConfig(**data.get("wandb", {})),
            run=RunConfig(**data.get("run", {})),
        )



def load_config(path: str) -> Config:
    """Load a config JSON file into a Config object."""
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Config.from_dict(data)


def save_config(cfg: Config, path: str) -> None:
    """Write a Config object to JSON."""
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, sort_keys=True)
