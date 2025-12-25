# Changelog

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