# Alpha-Parameterized Diffusion Inference: Modular Implementation

This codebase implements three inference methods for inferring spatially-varying diffusivity `D(x)` from steady-state or particle data, with support for varying stochastic conventions via an alpha parameter.

## Scientific Context

We solve the inverse problem for the steady-state birth-diffusion-death equation:

```
L_α[u](x) - μ u(x) = -b₀ Σ δ(x - zₖ)
u(0) = u(1) = 0  (Dirichlet BCs)
```

where the **alpha-parameterized operator** is:

```
L_α[u] ≡ ∇ · [D(x)^α ∇(D(x)^(1-α) u(x))]
```

**Alpha convention:** Controls the stochastic interpretation:
- `α = 0` (Itô): Pure diffusion SDE, `(D u)''` operator
- `α = 0.5` (Stratonovich): Intermediate
- `α = 1` (Fickian): Classical diffusion, `(D u')'` operator

**Goal:** Infer `D(x)` and `b₀` from either:
- Dense field observations of `u(x)`
- Particle snapshots (Poisson point process)

**Domain convention:** The code assumes a nondimensionalized spatial domain `[0, 1]`
for boundary enforcement and particle simulation. We typically nondimensionalize
space by `L` and time by `1/μ`, so `μ = 1` in nondimensional units (the code still
allows other `μ` values for testing).

## Architecture Overview

### Module Structure

```
combined_code/
├── config.py             # Configuration system (dataclasses + validation)
├── physics.py            # Alpha-parameterized PDE: residuals, FDM solver, regularization
├── varpro.py             # Variable projection for b₀ (field + PPP)
├── data.py               # Synthetic data generation (DDI, PPP sampling, SDE particles)
├── interface.py          # Notebook-friendly API: Problem, Solution, solve()
│
├── method_dto.py         # Discretize-then-optimize (tridiagonal solve + VarPro)
├── method_pinn.py        # Physics-informed neural network + VarPro
├── method_bilo.py        # Bilevel local operator (neural surrogate) + VarPro
│
├── diagnostics.py        # Metrics, plotting, comparison tools
└── examples/             # Example notebook-style scripts
```

### Key Design Principles

1. **Single physics implementation** - All methods use the same `physics.py` definitions to avoid accidental inconsistencies
2. **VarPro everywhere** - Amplitude `b₀` is always projected via Variable Projection (never optimized directly)
3. **Method self-containment** - Each method file is readable standalone (**INTENTIONAL CODE DUPLICATION** for `_init_logd_profile` and network classes to ensure isolated benchmarking)
4. **Notebook-first interface** - `interface.py` exposes a minimal API for interactive experiments

## Core Modules

### `config.py` - Configuration System

Nested dataclass structure with validation:

```python
from config import Config

cfg = Config()
cfg.physics.alpha = 0.5          # Stratonovich convention
cfg.physics.mu = 5.0              # Death rate
cfg.physics.sources = (0.5,)      # Source location
cfg.data.mode = "field"           # or "particles"
cfg.data.field_loss = "rle"       # or "mse"
cfg.train.finetune_iters = 10000
```

**Config loading:** `Config.from_dict(...)` expects nested config dicts (legacy flat configs
are not supported).

For most notebook usage, prefer the dict-based defaults exposed in `interface.py`:

```python
from interface import get_default_settings, show_settings, solve

show_settings()
settings = {**get_default_settings(), "lr_d": 1e-3, "max_iters": 5000}
result = solve(problem, method="dto", **settings)
```

### `physics.py` - Physics Core

**Alpha flux residual** (autograd-compatible):
PINN/BiLO compute the residual in flux form:
```
q = D^(1-α) * u
J = D^α * dq/dx
residual = dJ/dx - μu
```

**FDM reference solver** (NumPy, for ground truth):
```python
u_true = physics.fdm_solve_alpha_dirichlet(logd, alpha, mu, x, b0, sources)
```

**Regularization:**
```python
# Smoothness on log(D)
reg_smooth = physics.h1_smoothness_logd(x, logd)  # Autograd H1
reg_smooth = physics.tv_smoothness_logd(x, logd)  # Autograd TV
reg_smooth = physics.h1_smoothness_logd_discrete(logd, h)  # DTO H1
reg_smooth = physics.tv_smoothness_logd_discrete(logd, h)  # DTO TV

# Log-normal scale anchor (anchors logD pointwise)
reg_scale = physics.log_scale_anchor(logd, log_target=np.log(d_init_base))
```

### `varpro.py` - Variable Projection

**Field data:**
```python
# Project amplitude
b0_star = varpro.project_b0_field(u_hat, u_true, field_loss="mse")  # or "rle"

# Matched data loss
loss = varpro.field_data_loss(u_hat, u_true, b0_star, field_loss="mse")
```

**Particle data (PPP):**
```python
integral_u = torch.trapezoid(u_hat, x)
b0_star = varpro.project_b0_ppp(n_obs, m_obs, integral_u)
nll = varpro.ppp_nll(u_hat_obs, b0_star, m_obs, integral_u)  # Mean per-snapshot
```

**Key:** All VarPro projections keep `b0_star` in the computation graph for gradient flow.

### `data.py` - Data Generation

**Data-driven initialization (DDI):**
```python
d_init_base = data.estimate_ddi_scale(
    mu=5.0, z=0.5, x_particles=particles, d_min=1e-4, d_max=10.0
)
# Estimates D ~ μ · MAD² from particle or field data
```

**PPP sampling from field:**
```python
ppp = data.sample_ppp_from_field(x_grid, u_field, m_obs=100, rng=rng)
# Returns PPPData(x_particles, m_obs)
```

**SDE particle simulation:**
```python
ppp = data.simulate_particles_alpha(
    d_func, dprime_func, z=0.5, birth_rate=100, death_rate=5.0,
    alpha=0.5, tmax=100, dt=1e-3, m_obs=100, rng=rng
)
# Simulates: dX = α·D'(X)dt + √(2D(X))dW with births/deaths
```

## Methods

### DTO (Discretize-Then-Optimize)

**What:** Optimize discrete `logD` on grid via differentiable tridiagonal solve.

```python
from method_dto import fit, DTOData

data = DTOData(mode="field", x_res=x_res, u_true=u_true)
result = fit(data, cfg)
# Returns: DTOResult(x_res, logd, d_pred, u_hat_unit, u_pred, b0_star, history)
```

**Features:**
- Conservative alpha-discretization (matches `physics.py`)
- Differentiable Thomas solver (autograd-safe via list-based accumulation)
- VarPro projection each iteration
- **p2h-s1 hat delta:** Source terms use a 2-point hat function to handle arbitrary
  source locations. If z falls between grid points, mass is distributed using weights
  `w = 1 - |x - z|/h`. Achieves 2nd-order convergence and reduces to exact placement
  when z is on-grid.

### PINN (Physics-Informed Neural Network)

**What:** Neural networks for `logD(x)` and `u(x)`, coupled through physics residual.

```python
from method_pinn import fit, PINNData

data = PINNData(mode="particles", x_res=x_res, ppp=ppp)
result = fit(data, cfg)
```

**Architecture:**
- `LogDNet`: Optional frozen RFF features → MLP → `logD(x)`
- `LocalOperator` (u network): Geometry `[x, |x-z|]` → MLP → `u(x)` (enforces BCs)
- Separate networks, coupled only through physics loss

**Loss components:**
- Data loss (field or PPP NLL) with VarPro
- Physics residual `L_α[u] - μu` (source point excluded)
- Jump condition at sources: `D(z)·[u'(z⁺) - u'(z⁻)] = -b₀`
- Regularization: H1/TV smoothness + log-normal scale anchor on logD

**Grid handling:** PINN and BiLO use an **aligned grid** where the source location z is
exactly on a grid point. This is constructed by `physics.build_aligned_grid()` which
splits points proportionally on each side of z. The residual is excluded at exactly
that one index (no fuzzy tolerance needed).

### BiLO (Bilevel Local Operator)

**What:** Bilevel optimization with neural operator surrogate.

```python
from method_bilo import fit, BiLOData

data = BiLOData(mode="field", x_res=x_res, u_true=u_true)
result = fit(data, cfg)
```

**Architecture:**
- `LogDNet`: Frozen Fourier features → MLP → `logD(x)`
- `LocalOperator`: Geometry `[x, |x-z|]` + **logD conditioning** → `u(x)`
  - LogD embedding mixed before first activation (trainable parameters; geometry RFF stays frozen when enabled)
  - Note: the original BiLO draft froze this embedding, but later revisions found it unnecessary; we keep it trainable on purpose.

**Training (bilevel):**

*Pretrain:*
1. Anchor `logD_net` to initialization
2. Train `LocalOperator` on physics + supervision to FDM unit-response

*Finetune (alternating):*
1. **Upper step**: Freeze operator, update `logD_net` via data loss + regularization
2. **Lower step**: Freeze `logD_net`, update `LocalOperator` via physics loss (unit source)

**Physics loss (lower):**
- PDE residual + jump condition
- `resgrad` and `jump_resgrad`: Penalize `∂(residual)/∂(logD)` to improve operator robustness
  - Uses deterministic all-ones `grad_outputs` (not stochastic probes)

## Notebook Interface

The recommended entry point is the notebook-friendly API in `interface.py`.

```python
from interface import Problem, solve, compare_methods

problem = Problem.synthetic(
    alpha=0.5,
    mode="field",
    d_profile="sinusoidal",
    d_profile_params=(0.1, 0.04, 4.0),
    mu=5.0,
    b_true=100.0,
    n_obs=201,
)
solution = solve(problem, method="bilo")
solution.plot(problem)

results = compare_methods(problem, max_iters=5000)
for name, sol in results.items():
    print(name, sol.metrics(problem)["d_rel_error"])
```

For a full walkthrough, see `baseline_ito_comparison.py`.

### Problem parameters (synthetic)

Common overrides in `Problem.synthetic(...)`:
- Required: `alpha`, `mode`, `d_profile`, `mu`, `b_true` (plus `m_obs` for particles)
- `d_profile`: `"sinusoidal"`, `"steps"`, or `"custom"`
- `d_profile_params=(mean, amplitude, frequency)` for `"sinusoidal"`/`"steps"`
  - `"steps"` uses a square-wave pattern with values `mean ± amplitude`
- `d_func` for `"custom"`
- `mode` ("field" or "particles"), `m_obs` (particles), `use_pde_sampling` (particles)
- `n_obs` (observation grid), `seed`
- Solver/residual grid is set by `config.grid.n_res` (defaults to 201).
  - For field mode, predictions are interpolated to the observation grid when grids differ.
  - The solver grid used is available as `solution.x_res`.

### solve() knobs

Common training overrides in `solve(...)`:
- `max_iters`, `pretrain_iters`
- `lr_d`, `lr_lower`
- `w_data`, `w_phys`, `wreg_smooth`, `wreg_scale`, `smoothness_type`, `w_jump`, `w_resgrad`
- `field_loss`, `use_ddi`, `d_init_base`, `d_init_scale`, `d_init_freq`
- `use_scheduler`, `use_rff`
- `early_burnin`, `early_patience`, `early_tol`, `log_every`

Use `show_settings()` to print defaults and `get_default_settings()` to start from them.

## Usage Examples

### Example 1: Quick synthetic run

```python
from interface import Problem, solve

problem = Problem.synthetic(
    alpha=0.0,
    mode="field",
    d_profile="sinusoidal",
    d_profile_params=(0.1, 0.04, 4.0),
    mu=5.0,
    b_true=100.0,
    n_obs=201,
)
solution = solve(problem, method="dto", max_iters=1000)
print(f"b0* = {solution.b0_star:.2f}")
```

### Example 2: Particle observations

```python
from interface import Problem, solve

problem = Problem.synthetic(
    alpha=0.0,
    mode="particles",
    d_profile="sinusoidal",
    d_profile_params=(0.1, 0.04, 4.0),
    mu=5.0,
    b_true=100.0,
    m_obs=250,
    n_obs=201,
)
solution = solve(problem, method="bilo", max_iters=5000)
solution.plot(problem)
```

### Example 3: Custom D(x) profile

```python
import numpy as np
from interface import Problem, solve

problem = Problem.synthetic(
    alpha=0.0,
    mode="field",
    d_profile="custom",
    d_profile_params=None,
    d_func=lambda x: 0.15 + 0.05 * np.sin(6 * np.pi * x),
    mu=5.0,
    b_true=100.0,
    n_obs=201,
)
solution = solve(problem, method="pinn", max_iters=2000)
```

### Example 4: Step D(x) profile

```python
from interface import Problem, solve

problem = Problem.synthetic(
    alpha=0.0,
    mode="field",
    d_profile="steps",
    d_profile_params=(0.1, 0.04, 4.0),
    mu=5.0,
    b_true=100.0,
    n_obs=201,
)
solution = solve(problem, method="dto", max_iters=2000)
```

### Example 5: Dict-based settings with overrides

```python
from interface import Problem, solve, get_default_settings

settings = get_default_settings()
settings["n_res"] = 401
settings["wreg_smooth"] = 1e-6
problem = Problem.synthetic(
    alpha=0.25,
    mode="field",
    d_profile="sinusoidal",
    d_profile_params=(0.1, 0.04, 4.0),
    mu=5.0,
    b_true=100.0,
    n_obs=201,
)
solution = solve(problem, method="bilo", **settings)
```

## Configuration Reference

### Key Config Sections

**Physics:**
- `alpha`: Stochastic convention ∈ [0,1]
- `mu`: Death rate (positive)
- `sources`: Tuple of source locations (currently single-source only)
- `b_true`: True amplitude (for synthetic data)

**Data:**
- `mode`: `"field"` or `"particles"`
- `field_loss`: `"mse"` or `"rle"` (relative error with weighted VarPro)
- `m_obs`: Number of particle snapshots

**Integration grid (PPP):**
- `n_int`: Integration grid used to compute `∫u dx` for PPP likelihood/projection.
  All three methods integrate on `n_int` (interpolating from `x_res` when needed).

**Solver grid:**
- `n_res`: Number of solver grid points for D/u (set in config.grid or via `solve(..., n_res=...)`).
- **DTO** uses a uniform grid; source placement uses p2h-s1 hat delta for arbitrary z.
- **PINN/BiLO** use an aligned grid (`physics.build_aligned_grid`) where z is exactly on-grid.

**D Profile:**
- `use_ddi`: Enable data-driven initialization (estimate scale from data)
- `d_init_base`: Base value for initial D(x)
- `d_init_pert_scale`: Initialization perturbation (< 1)
- `d_init_pert_freq`: Initialization perturbation frequency

**Loss weights:**
- `w_data`: Data loss weight (PINN only)
- `w_phys`: Physics loss weight (PINN only)
- `w_jump`: Jump condition weight (PINN/BiLO)
- `w_resgrad`: Residual gradient penalty (BiLO)

**Regularization:**
- `wreg_smooth`: Smoothness weight on log D(x)
- `smoothness_type`: `"h1"` or `"tv"` smoothness selector
- `wreg_scale`: Log-normal scale anchor weight on log D(x)

**Architecture:**
- `use_rff_geom`, `use_rff_logd`: Enable random Fourier features
- `rff_width`: RFF embedding width

**Training:**
- `pretrain_iters`: Pretrain iterations (PINN/BiLO only)
- `finetune_iters`: Main training iterations
- `lr_d_pre`, `lr_d_fine`: Learning rates for D network
- `lr_lower_pre`, `lr_lower_fine`: Learning rates for u network (PINN/BiLO)
- `use_scheduler`: Enable cosine LR scheduling
- `log_every`: Log history every N iterations
- `early_burnin`, `early_patience`, `early_tol`: Early stopping (relative improvement threshold)

## Diagnostics and Metrics

```python
import numpy as np
import diagnostics

# Compute standardized metrics
metrics = diagnostics.compute_field_metrics(
    x, d_true, u_true, d_pred, u_pred, logd_pred, b0_star, b_true
)
# Returns: {D_error_L2, D_error_rel, D_error_u_L2, D_error_u_rel,
#           D_shape_L2, D_shape_rel, D_correlation, logD_error_L2,
#           logD_error_rel, u_error_L2, u_error_rel, b0_star_err,
#           b0_star_rel_err, mean_D, integral_u}
# Relative errors follow sqrt(∫|true-est|^2 / ∫|true|^2).

# Training history plots (loss curves, b₀ evolution, D snapshots)
diagnostics.plot_training_history(
    name,
    result.history,
    b_true,
    outdir,
    mean_d_true=np.mean(d_true),
    weights=result.weights,
)

# D evolution heatmap (D(x) vs iteration)
diagnostics.plot_d_evolution(name, result.history, x, outdir)

# Method comparison plot
diagnostics.plot_solution_comparison(
    x, d_true, u_true, results, metrics, outdir,
    mode="field",  # or "particles" with x_particles
    filename="comparison.png"
)

# BiLO-specific: neighborhood check (local operator generalization)
diagnostics.plot_bilo_neighborhood_check(
    logd_net, local_op, x_res, z, alpha, mu, outdir
)
```

## Implementation Notes

### Variable Projection (VarPro)

**Why project b₀ instead of optimizing?**
- Eliminates one optimization variable
- Closed-form solution each iteration
- **Crucial:** Matches the projection to the loss type:
  - MSE: OLS projection `b* = ⟨û, u⟩/⟨û, û⟩`
  - RLE: Weighted LS with `w = 1/u²`
  - PPP: MLE `b* = N/(M·∫û)`

**Gradient flow:** `b0_star` is NOT detached, so `∂L/∂D` flows through the projection.

### Alpha-Parameterized Discretization

**Flux form implementation:**
```
q = D^(1-α) u
J = D^α dq/dx
L_α[u] = dJ/dx
```

**Conservative stencil** (midpoint diffusion):
- `logD_half = 0.5(logD[i] + logD[i+1])`
- `a_half = D_half^α / h`
- Couples `q[i]` values with flux divergence

**Reduces correctly:**
- α=0 (Itô): `(D u)''` via `q = Du`, `J = dq/dx`
- α=1 (Fickian): `(D u')'` via `q = u`, `J = D·du/dx`

### BiLO Bilevel Structure

**Why bilevel?**
- Upper (D inference): Uses data to infer parameters
- Lower (solver): Learns PDE solution operator at fixed D

**Unit-response convention:** LocalOperator predicts `u_hat_unit` for `b=1`, preventing scale shock in physics training.

**Gradient isolation:** Lower step detaches `logD` from parameters but keeps it differentiable for `resgrad` penalties.

## Known Limitations and Future Extensions

### Current Constraints
- **Single source only** (multi-source planned, architecture supports it)
- **Dirichlet BCs only** (Robin/Neumann planned)
- **1D domain** (2D extension straightforward for DTO/PINN, needs work for BiLO)

### Planned Extensions
- Multiple point sources (sister chromatids)
- Unknown boundary permeability (adds one parameter)
- Coupled PDEs (precursor → product with conversion)
- Time-dependent inference
- **Improved Regularization Scaling:** Standardize regularization weights (H1, TV, scale anchor) to be invariant to the scale of `D` or the choice of loss function (e.g., normalizing by `1/D_init^2` for MSE consistency).
- **Stability Improvement (Trust Region):** Address "Loose Tangent Instability" where the local operator learns the value `u` well but the sensitivity `∇_θ u` poorly (high `rgrad`). A proposed fix is to scale the upper-level update by `1/(1 + γ · L_rgrad)` to dampen parameter jumps when the sensitivity estimate is unreliable.

### Design Extension Points
- `PhysicsConfig.sources`: Already a tuple
- `_build_delta_rhs`: Sums over sources
- `LocalOperator`: Could condition on multiple `z_k`

## Troubleshooting

### Common Issues

**Divergence / NaN losses:**
- Check DDI initialization (set `use_ddi=True`)
- Increase `wreg_scale` (especially with RLE loss)
- Reduce learning rates
- Check particle data has sufficient counts

**D-b₀ drift (mean-D changes during training):**
- Increase `wreg_scale` (try 0.1 to 1.0)
- This is especially important for RLE loss
  - Consider L-curve tuning for `wreg_scale` when data are noisy

**BiLO local operator fails to generalize:**
- Increase pretrain iterations
- Check `w_resgrad` is enabled (try 0.01)
- Verify geometry RFF features are frozen (`use_rff_geom=True`)

**Poor performance at boundaries:**
- Check boundary conditions are enforced (output transform in LocalOperator)
- For PINN/BiLO, the source is exactly on the aligned grid, so masking is precise

**DTO singular matrix errors:**
- Check `mu > 0` (ensures diagonal dominance)
- Verify `D > 0` everywhere (initialization perturbation < 1)

## Citation

If you use this code, please cite the associated paper (reference TBD).

## License

(Add license information here)
