# Changelog

## [v4.0.1] - 2026-01-09
- Added Neumann (zero-flux) boundary condition support via `bc_type="neumann"` in all three
  methods. Includes `_build_tridiag_alpha_neumann()` in DTO and `_compute_bc_loss_neumann()`
  in PINN/BiLO. Note: Neumann BCs with α≠1 have non-identifiability (warns at runtime).
- Extracted training logging utilities into `training_logger.py` module. Provides
  `TrainingHistory` class with method-specific factories (`.for_dto()`, `.for_pinn()`,
  `.for_bilo()`) and progress formatters (`format_dto_progress()`, etc.). Reduces
  boilerplate in method files.
- Added `b0_fixed_value` option to use a known source amplitude instead of VarPro projection.
  When set, bypasses Variable Projection and uses the fixed value directly. Useful when
  the source amplitude is known from experimental calibration, eliminating the
  amplitude-diffusivity ambiguity. Available in all three methods (DTO, PINN, BiLO).
- Added `varpro.get_b0_field()` and `varpro.get_b0_ppp()` unified helper functions.
- Added scalar-fit scale estimation via differentiable FDM (`scale_estimation.py`).
- Added torch FDM solver (`physics.fdm_solve_alpha_dirichlet_torch`) for autograd use.
- Updated config/init knobs: `scalar_fit_iters`, `pert_scale`, `pert_freq` (removed `d_init_base`).

## [v3.0.1] - 2025-12-29
### Fixed: D Parameterization Gradient Issues

**Problem:** The v3.0.0 `LeakyReLU + clamp(min=0)` parameterization in `DNet` caused gradient death:
- When `raw` went negative, `clamp(min=0)` killed gradients entirely
- D would collapse to `D_MIN` and get stuck (observed as `Ljump ≈ 1.0`, `⟨D⟩ → 1e-6`)

**Problem:** The `* d` scaling in gradient penalties (`jump_rgrad`, `rgrad`) created a positive feedback loop:
- Small D → penalty scaled by small D → penalty vanishes → D can shrink further → collapse

**Fix:**
1. Reverted `DNet` to use `softplus(raw) + D_MIN` (smooth gradients everywhere)
2. Added `d.clamp(min=D_MIN)` to gradient penalty scaling to prevent collapse while preserving log-domain conversion

**Affected files:** `method_bilo.py`, `method_pinn.py`

**Note:** `method_dto.py` uses direct parameterization (`D = θ + D_MIN` with projection), which doesn't have these issues.

## [v3.0.0] - 2025-12-24
- Shifted parameterization from $\log D$ to $D$ (via Softplus) across all solvers and physics helpers.
- Retained internal $\log D$ embedding in the BiLO local operator to maintain sensitivity scaling.
- Updated finite-difference solvers to support non-uniform grids with variable step sizes.
- Introduced shape-sensitive metrics including Pearson correlation and normalized L2 error.
- Refactored training history to plot the change in loss from initialization ($\Delta$ Loss) on a symlog scale.
- Added fine-grained learning rate overrides and RFF frequency scaling to the main `solve()` interface.

## [v2.0.0] - Previous Version
- Corrected the delta function implementation and associated masking logic for point sources.
- Updated the DTO method to utilize a 2-grid approximation strategy.
- Enforced inclusion of the source location $z$ in PINN and BiLO grids to resolve singularities.
