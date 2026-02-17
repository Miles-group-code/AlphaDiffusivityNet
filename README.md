# Alpha-Parameterized Diffusion Inference

A modular framework for inferring spatially-varying diffusivity $D(x)$ from steady-state concentration fields or particle snapshot data. This codebase implements a unified approach to the "Alpha-Flux" inverse problem, supporting various stochastic interpretations of diffusion (Itô, Stratonovich, Fickian) via a tunable parameter $\alpha$.

## The Problem

We solve the inverse problem for the steady-state birth-diffusion-death equation:

$$ \nabla \cdot [D(x)^\alpha \nabla (D(x)^{1-\alpha} u(x))] - \mu u(x) = -b_0 \delta(x-z) $$

Where:
*   $\alpha \in [0, 1]$ controls the stochastic convention (0=Itô, 0.5=Stratonovich, 1=Fickian).
*   $D(x)$ is the unknown diffusivity profile we wish to recover.
*   $b_0$ is the unknown source amplitude.
*   Data comes as either a dense density field $u(x)$ or discrete particle positions (Poisson Point Process).

## Inference Methods

This repository implements three distinct solvers, each with different parameterizations and optimization strategies.

### 1. DTO (Discretize-Then-Optimize)
*   **Mechanism**: Directly optimizes $D(x)$ on a discretized grid using direct parameterization with projection. It solves the forward problem using a **differentiable FDM solver** (Thomas algorithm) at every training step.
*   **Key Properties**:
    *   Provides exact gradients of the discretized physics with respect to $D$.
    *   Hard-encodes the boundary conditions and linearity of the PDE.
    *   Computational cost scales linearly with grid resolution ($O(N)$).
    *   Uses direct parameterization ($D = \theta + D_{min}$, projected) to avoid gradient suppression.

### 2. PINN (Physics-Informed Neural Network)
*   **Mechanism**: Parameterizes both $D(x)$ and $u(x)$ as separate neural networks (`DNet` and `LocalOperator`). The networks are trained jointly to minimize a composite loss: $\mathcal{L} = \mathcal{L}_{data} + \mathcal{L}_{physics}$.
*   **Key Properties**:
    *   Continuous neural network representation of $D(x)$ and $u(x)$.
    *   Uses a "flux-form" residual loss to handle the $\alpha$-differentiation.
    *   Enforces the source singularity via a specific jump condition penalty at $x=z$.
    *   Does not require solving a linear system.
    *   Uses softplus output activation ($D = \text{softplus}(\text{output}) + D_{min}$) for smooth gradients.

### 3. BiLO (Bilevel Local Operator)
*   **Mechanism**: A bilevel optimization approach.
    *   **Inner Loop (Lower)**: Trains a local surrogate operator network ($D \to u$) to satisfy the PDE physics for the *current* estimate of $D$.
    *   **Outer Loop (Upper)**: Updates the $D$ network to minimize the data loss, using gradients backpropagated through the frozen surrogate operator.
*   **Key Properties**:
    *   Decouples the stiff physics constraints from the data fitting manifold.
    *   Uses `resgrad` (residual gradient penalty) to stabilize the operator training.
    *   Often generalizes better to complex profiles than standard PINNs by learning the solution operator locally.
    *   Uses softplus output activation ($D = \text{softplus}(\text{output}) + D_{min}$) for smooth gradients.

## Problem Configuration

The physics and domain settings are fully configurable via the `Problem` interface or `config.py`.

*   **`alpha`**: Stochastic interpretation ($0.0$ to $1.0$).
*   **`mu`**: Degradation rate (controls the exponential decay length scale).
*   **`d_profile`**: Shape of the ground truth D(x) (e.g., "sinusoidal", "steps").
    *   *Note*: The "steps" profile applies a random phase shift to avoid grid alignment.
*   **`bc_type`**: Boundary conditions (`dirichlet` or `neumann`). Neumann (zero-flux) BCs are only implemented for `alpha=1` (Fickian); other alpha values will print a warning about non-identifiability.
*   **`sources`**: Location(s) of the point source(s) $z$.
*   **`domain`**: Spatial extent $[x_{min}, x_{max}]$ (default $[0, 1]$).
*   **`b_true`**: (Synthetic) True source amplitude for generating ground truth.

## Solvers & Options

### Data Modes
*   **`field`**: Learning from a continuous density field $u(x)$ evaluated on a grid.
*   **`particles`**: Learning from discrete particle snapshots (Poisson Point Process).

### Loss Functions
*   **`mse`**: Mean Squared Error (Standard for dense, Gaussian-noise field data).
*   **`rle`**: Relative Log Error ($\|\log u - \log \hat{u}\|^2$). Robust for fields that span many orders of magnitude (common in diffusion-death).
*   **`nll`**: Negative Log Likelihood. The statistically correct loss for particle/count data (Poisson process).

### Regularization
*   **Smoothness**:
    *   **`h1`**: Penalizes the gradient of $\log D$ ($\int (\nabla \log D)^2$).
    *   **`tv`**: Total Variation, penalizes the L1 norm of the gradient ($\int |\nabla \log D|$). Good for piecewise-constant profiles.
*   **Scale Anchor**:
    *   **`wreg_scale`**: Penalizes deviation of the mean log-diffusivity from a prior (derived via scalar fit seeded by DDI), preventing amplitude-diffusivity ambiguity.

### Variable Projection (VarPro)
All methods use **Variable Projection** to handle the unknown source amplitude $b_0$. Instead of optimizing $b_0$ via gradient descent, we compute its optimal value $b_0^*$ in closed form at every step:
*   For **MSE**: $b_0^* = \frac{\langle u_{data}, \hat{u}_{unit} \rangle}{\|\hat{u}_{unit}\|^2}$
*   For **RLE**: Weighted Least Squares solution.
*   For **Particles**: $b_0^* = \frac{N_{particles}}{M_{SNAPSHOTS} \int \hat{u}_{unit} dx}$

**Fixed b0 mode:** When the source amplitude is known a priori (e.g., from experimental calibration), you can bypass VarPro by setting `b0_fixed_value` in `solve()`. This eliminates the amplitude-diffusivity ambiguity.

### Scale Estimation (DDI + Scalar Fit)
**DDI** provides a fast heuristic scale estimate from data spread, then a **constant-D scalar fit** refines the scale by minimizing the actual data loss via a differentiable FDM solve. The resulting scale anchors regularization and stabilizes training.

### Optimizer Selection
You can switch the finetune optimizer for any method via `solve(..., optimizer=...)`:

*   **`"adam"`**: Default and recommended for all methods. Works well with direct/ReLU parameterization.
*   **`"lbfgs"`**: Quasi-Newton, available but empirically does not outperform Adam and often stalls.

```python
solution = solve(problem, method="dto", optimizer="adam")  # Recommended
solution = solve(problem, method="pinn", optimizer="lbfgs")  # Available but not recommended
```

Note: For PINN/BiLO, the pretrain phase always uses Adam to warm up the networks before finetuning.
## Usage

### Quick Start

```python
from interface import Problem, solve

# 1. Define a synthetic problem
problem = Problem.synthetic(
    alpha=0.5,                   # Stratonovich
    mode="field",                # or "particles"
    d_profile="sinusoidal",      # Ground truth shape
    mu=5.0
)

# 2. Solve using BiLO with custom settings
solution = solve(
    problem, 
    method="bilo", 
    field_loss="rle",            # Use Relative Log Error
    wreg_smooth=1e-5             # Light smoothing
)

# 3. Visualize
solution.plot(problem)

# Or write your own plots using raw data in the solution object:
#   solution.x_res, solution.d_pred, solution.u_pred, solution.b0_star, solution.history
```

### Advanced Configuration

You can override any config parameter by passing it as a keyword argument to `solve()`.

```python
from interface import get_default_settings, solve

settings = get_default_settings()
settings["wreg_smooth"] = 1e-4
settings["lr_d"] = 2e-3
settings["use_ddi"] = True      # Seed scalar fit with DDI
settings["scalar_fit_iters"] = 500

solution = solve(problem, method="pinn", **settings)
```

## Known Limitations

*   **Small D values (PINN/BiLO):** Softplus parameterization has gradient suppression proportional to D. At D = 0.01, gradients are ~100x weaker than at D = 1. If your nondimensional D is small (e.g., D/μ ≈ 0.01), use DTO or increase training iterations. See `README.md` troubleshooting for details.
*   **Single source only:** Multi-source support is planned but not yet implemented.

## Running Experiments

### 1. Command Line Interface (CLI)
You can run single experiments using `run.py`. Arguments can be provided in `key value` pairs.

The keys can be either full paths or shortcuts.
*   **Full paths**: `python run.py physics.b_true 100.0`
*   **Shortcuts**: `python run.py b_true 100.0` (works if `b_true` is unique in the config)

**Examples:**
```bash
python run.py method bilo alpha 1.0 train.lr_d_fine 1e-3 reg.wreg_smooth 1e-4

```

### 2. Batch Experiments
For running many experiments (for example, different methods), use `runexp.py` with a YAML file.
This script manages a queue of experiments and distributes them across available GPUs.

**Usage:**
```bash
python runexp.py fic_all.yaml
```
**Example YAML File:**
The nested YAML structure will be flattened into a list of experiments.

```yaml
# 1. Define reusable blocks (anchors) starting with _
_common: &common
  physics: "alpha 1.0 mu 5.0"
  lr: "train.lr_lower_pre 1e-4 "

_methods: &methods
  pinn:
    solver: "method pinn"
  bilo:
    solver: "method bilo"

# 2. Define Experiment Groups
fickian:
  group: "group fic_sweep"  # Sets wandb group
  <<: *common               # Inherits common settings
  
  freq05:
    dparam: "params 0.1,0.05,0.5"
    <<: *methods            # Expands to pinn and bilo
  freq1:
    dparam: "params 0.1,0.05,1"
    <<: *methods
```

**This generates the following experiments:**

| Run Name | Combined Arguments (Simplified) |
| :--- | :--- |
| `fickian_freq05_pinn` | `group fic_sweep alpha 1.0 mu 5.0 train.lr_lower_pre 1e-4  params 0.1,0.05,0.5 method pinn` |
| `fickian_freq05_bilo` | `group fic_sweep alpha 1.0 mu 5.0 train.lr_lower_pre 1e-4  params 0.1,0.05,0.5 method bilo` |
| `fickian_freq1_pinn` | `group fic_sweep alpha 1.0 mu 5.0 train.lr_lower_pre 1e-4  params 0.1,0.05,1 method pinn` |
| `fickian_freq1_bilo` | `group fic_sweep alpha 1.0 mu 5.0 train.lr_lower_pre 1e-4  params 0.1,0.05,1 method bilo` |

This hierarchical structure allows you to define shared physics/training settings once (`*common`), and then run multiple experiments with different parameters or solvers.

### 3. Weights & Biases (WandB)
WandB is used for logging loss histories and visualizations.

Pass `wandb.enabled true` will enable wandb logging. Various options can be set in the config.WandBConfig.
If wandb is not installed, it will log to the local directory `runs/[run_name]/`.

## Repository Structure

*   `interface.py`: **Main Entry Point**. High-level API for defining problems and running solvers.
*   `config.py`: Configuration dataclasses and validation.
*   `physics.py`: Core physics definitions, finite-difference solvers, and regularization terms. Shared across all methods.
*   `varpro.py`: Variable projection logic for amplitude estimation ($b_0$).
*   `data.py`: Synthetic data generation (fields, particle simulations).
*   `scale_estimation.py`: DDI + scalar-fit scale estimation.
*   **Methods**:
    *   `method_dto.py`: Discretize-Then-Optimize implementation.
    *   `method_pinn.py`: PINN implementation.
    *   `method_bilo.py`: BiLO implementation.
*   `training_logger.py`: Training history tracking and progress formatters.
*   `diagnostics.py`: Plotting and metric calculation tools.
*   `run.py`: **CLI Entry Point**. Runs a single experiment from command line arguments.
*   `runexp.py`: **Batch Runner**. Executes multiple experiments defined in a YAML file, handling GPU allocation.
*   `DenseNet.py`: Contains all neural network definitions (MLP, Siren, etc.) shared by BiLO and PINN.


## License

[Insert License Here]
