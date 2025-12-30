"""Test script to evaluate loss landscape flatness for DTO.

Sweeps D(x) = k * D_true(x) for k in [k_min, k_max] and plots the
data loss (with VarPro projection) to verify whether the landscape is
genuinely flat or just ill-conditioned.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

import data as data_utils
import physics
import varpro
from interface import Problem

# =============================================================================
# SETTINGS
# =============================================================================

# Physics
alpha = 0         # 0=Itô, 0.5=Stratonovich, 1=Fickian
mu = 1.0             # death rate
b_true = 100.0       # true source amplitude
source = 0.5         # source location

# D profile: "sinusoidal" or "steps"
#   params = (mean, amplitude, frequency)
d_profile = "sinusoidal"
d_profile_params = (0.1, 0.04, 4.0)  # (mean, amp, freq)

# Data mode: "field" or "particles"
mode = "particles"

# For field mode: "mse" or "rle"
field_loss = "rle"

# For particles mode
m_obs = 1000
n_trials = 20

# Grid resolutions
n_obs = 201       # observation grid (Problem.synthetic)
n_res = 201       # solver grid (for FDM solve in sweep)

# Sweep settings
k_min = 0.1
k_max = 10
n_k = 201

# Plot settings
log_scale = True      # plot loss in log scale

# Regularization (match training knobs)
wreg_smooth = 0
wreg_scale = 1e-3
smoothness_type = "h1"   # "h1" or "tv"
d_init_base = None

# Other
seed = 42
domain = (0.0, 1.0)

# =============================================================================
# GENERATE PROBLEM
# =============================================================================

use_pde_sampling = True if mode == "particles" else False

problem = Problem.synthetic(
    alpha=alpha,
    mode=mode,
    d_profile=d_profile,
    d_profile_params=d_profile_params,
    mu=mu,
    b_true=b_true,
    m_obs=m_obs if mode == "particles" else 1,
    source_location=source,
    n_obs=n_obs if mode == "field" else 9999,
    domain=domain,
    seed=seed,
    verbose=False,
    use_pde_sampling=use_pde_sampling
)

# Observation grid (where data lives)
x_obs = problem.x_grid.numpy()
d_true_obs = problem.d_true
u_true_obs = problem.u_true

x_obs_t = problem.x_grid.double()
u_true_t = torch.tensor(u_true_obs, dtype=torch.float64)

# Solver grid (where we solve the PDE)
x_res = np.linspace(domain[0], domain[1], n_res)
x_res_t = torch.tensor(x_res, dtype=torch.float64)
h_res = float(x_res[1] - x_res[0])

# Interpolate D_true onto solver grid
d_true_res = np.interp(x_res, x_obs, d_true_obs)

print(f"D profile: {d_profile}, params={d_profile_params}")
print(f"Mode: {mode}" + (f", loss={field_loss}" if mode == "field" else f", m_obs={m_obs}"))
print(f"Grids: n_obs={n_obs}, n_res={n_res}, alpha={alpha}, mu={mu}")

if mode == "particles":
    ppp_data = problem.particles
    print(f"Particles: {ppp_data.n_obs} total from {ppp_data.m_obs} snapshots")

ppp_trials = None
if mode == "particles":
    ppp_trials = [ppp_data]
    if n_trials > 1:
        if not use_pde_sampling:
            print("n_trials>1 requires use_pde_sampling=True; falling back to n_trials=1.")
            n_trials = 1
        else:
            for trial in range(1, n_trials):
                rng = np.random.default_rng(seed + trial)
                ppp_trials.append(
                    data_utils.sample_ppp_from_field(x_obs_t, u_true_t, m_obs, rng)
                )

# Debug: compare DDI scale to D_true scale on the sweep grid
d_true_mean = float(np.mean(d_true_res))
d_true_sq_mean = float(np.mean(d_true_res ** 2))

# =============================================================================
# SWEEP
# =============================================================================

print(f"\nSweeping k ∈ [{k_min}, {k_max}]...")

k_vals = np.linspace(k_min, k_max, n_k)
losses_trials = []
expected_trials = []
b0_trials = []
k_min_trials = []
d_init_bases = []
k_ddi_vals = []

integral_u_true = None
if mode == "particles":
    integral_u_true = torch.trapezoid(u_true_t.view(-1), x_obs_t.view(-1))

for trial in range(n_trials):
    if mode == "particles":
        ppp_data = ppp_trials[trial]

    if d_init_base is None:
        if mode == "particles":
            d_init_base_trial = data_utils.estimate_ddi_scale(
                mu=mu, z=source, x_particles=ppp_data.x_particles
            )
        else:
            d_init_base_trial = data_utils.estimate_ddi_scale(
                mu=mu, z=source, u_field=u_true_t, x_grid=x_obs_t
            )
    else:
        d_init_base_trial = d_init_base

    d_init_bases.append(d_init_base_trial)
    k_ddi_vals.append(d_init_base_trial * d_true_mean / d_true_sq_mean)

    losses = []
    expected_losses = []
    b0_stars = []

    for k in k_vals:
        # Solve on solver grid
        d_scaled = k * d_true_res
        d_scaled_t = torch.tensor(d_scaled, dtype=torch.float64)
        u_hat_res = physics.fdm_solve_alpha_dirichlet(
            d_scaled, alpha, mu, x_res, 1.0, (source,)
        )
        u_hat_res_t = torch.tensor(u_hat_res, dtype=torch.float64)

        reg_smooth = torch.tensor(0.0, dtype=torch.float64)
        if wreg_smooth > 0.0:
            if smoothness_type == "tv":
                reg_smooth = physics.tv_smoothness_d_discrete(d_scaled_t, h=h_res)
            else:
                reg_smooth = physics.h1_smoothness_d_discrete(d_scaled_t, h=h_res)

        reg_scale = torch.tensor(0.0, dtype=torch.float64)
        if wreg_scale > 0.0:
            reg_scale = physics.scale_anchor(d_scaled_t, d_init_base_trial)

        reg_loss = wreg_smooth * reg_smooth + wreg_scale * reg_scale

        # Interpolate u_hat to observation grid for loss computation
        u_hat_obs_t = varpro.interpolate_1d(u_hat_res_t, x_res_t, x_obs_t)

        if mode == "field":
            b0 = varpro.project_b0_field(u_hat_obs_t, u_true_t, field_loss=field_loss)
            data_loss = varpro.field_data_loss(
                u_hat_obs_t, u_true_t, b0, field_loss=field_loss
            )
        else:
            integral = torch.trapezoid(u_hat_res_t.view(-1), x_res_t.view(-1))
            u_at_pts = varpro.interpolate_1d(
                u_hat_res_t, x_res_t, ppp_data.x_particles.double()
            )
            b0 = varpro.project_b0_ppp(ppp_data.n_obs, ppp_data.m_obs, integral)
            data_loss = varpro.ppp_nll(u_at_pts, b0, ppp_data.m_obs, integral)

            # Expected PPP loss (per snapshot) under the true intensity
            integral_hat_obs = torch.trapezoid(u_hat_obs_t.view(-1), x_obs_t.view(-1))
            b0_expected = (b_true * integral_u_true) / integral_hat_obs.clamp_min(1e-12)
            intensity_expected = torch.clamp(b0_expected * u_hat_obs_t.view(-1), min=1e-12)
            expected_log = torch.trapezoid(
                u_true_t.view(-1) * torch.log(intensity_expected), x_obs_t.view(-1)
            )
            expected_nll = b0_expected * integral_hat_obs - b_true * expected_log
            expected_losses.append((expected_nll + reg_loss).item())

        loss = data_loss + reg_loss

        losses.append(loss.item())
        b0_stars.append(b0.item())

    losses = np.array(losses)
    expected_losses = np.array(expected_losses)
    b0_stars = np.array(b0_stars)
    losses_trials.append(losses)
    if expected_losses.size:
        expected_trials.append(expected_losses)
    b0_trials.append(b0_stars)
    k_min_trials.append(k_vals[np.argmin(losses)])

losses = np.mean(np.stack(losses_trials), axis=0)
expected_losses = (
    np.mean(np.stack(expected_trials), axis=0) if expected_trials else np.array([])
)
b0_stars = np.mean(np.stack(b0_trials), axis=0)

if d_init_bases:
    if n_trials == 1:
        print(f"DDI scale estimate: {d_init_bases[0]:.6g}")
        print(f"<D_true>={d_true_mean:.6g}, <D_true^2>={d_true_sq_mean:.6g}")
        print(f"Implied k from scale anchor: {k_ddi_vals[0]:.6g}")
    else:
        print(f"DDI scale estimate: mean={np.mean(d_init_bases):.6g}, std={np.std(d_init_bases):.6g}")
        print(f"<D_true>={d_true_mean:.6g}, <D_true^2>={d_true_sq_mean:.6g}")
        print(f"Implied k from scale anchor: mean={np.mean(k_ddi_vals):.6g}, std={np.std(k_ddi_vals):.6g}")

# =============================================================================
# RESULTS
# =============================================================================

idx_k1 = np.argmin(np.abs(k_vals - 1.0))  # index closest to k=1
dk = k_vals[1] - k_vals[0]
k_at_min = k_vals[np.argmin(losses)]
if k_min_trials:
    k_at_min_trials = float(np.mean(k_min_trials))
else:
    k_at_min_trials = None
curvature = (losses[idx_k1 + 1] - 2 * losses[idx_k1] + losses[idx_k1 - 1]) / dk**2

print(f"\nMinimum at k = {k_at_min:.4f}")
print(f"Loss at k=1.0 (idx={idx_k1}): {losses[idx_k1]:.6e}")
print(f"Loss at k={k_min}: {losses[0]:.6e} ({losses[0]/losses[idx_k1]:.2f}x)")
print(f"Loss at k={k_max}: {losses[-1]:.6e} ({losses[-1]/losses[idx_k1]:.2f}x)")
print(f"Curvature at k=1: {curvature:.6e}")
print(f"b0* at k=1: {b0_stars[idx_k1]:.2f} (true: {b_true})")

# =============================================================================
# PLOT
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# D profile
axes[0].plot(x_obs, d_true_obs, 'k-', lw=2)
axes[0].set_xlabel('x')
axes[0].set_ylabel('D(x)')
axes[0].set_title(f'True D profile ({d_profile})')
axes[0].grid(True, alpha=0.3)

# Loss landscape
if n_trials > 1:
    for trial_losses in losses_trials:
        trial_plot = trial_losses - trial_losses.min() + 1e-10 if log_scale else trial_losses
        axes[1].plot(k_vals, trial_plot, color='tab:blue', alpha=0.2, lw=1)
    for trial_k in k_min_trials:
        axes[1].axvline(trial_k, color='g', alpha=0.2, lw=1)
    if expected_trials:
        for trial_expected in expected_trials:
            expected_plot = (
                trial_expected - trial_expected.min() + 1e-10 if log_scale else trial_expected
            )
            axes[1].plot(k_vals, expected_plot, color='0.2', alpha=0.2, lw=1, ls='--')

losses_plot = losses - losses.min() + 1e-10 if log_scale else losses
axes[1].plot(k_vals, losses_plot, 'b-', lw=2)
if mode == "particles" and expected_losses.size:
    expected_plot = (
        expected_losses - expected_losses.min() + 1e-10 if log_scale else expected_losses
    )
    axes[1].plot(k_vals, expected_plot, 'k--', lw=1.5, label='expected PPP')
axes[1].axvline(1.0, color='r', ls='--', label='k=1 (true)')
axes[1].axvline(k_at_min, color='g', ls=':', label=f'min @ {k_at_min:.3f}')
if k_at_min_trials is not None and n_trials > 1:
    axes[1].axvline(
        k_at_min_trials, color='g', ls='-.', label=f'mean min @ {k_at_min_trials:.3f}'
    )
axes[1].set_xlabel('k (D = k·D_true)')
axes[1].set_ylabel('Loss - min' if log_scale else 'Loss')
axes[1].set_title(f'Loss Landscape ({mode}, {field_loss if mode=="field" else "PPP"})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
if log_scale:
    axes[1].set_yscale('log')

# b0 projection
axes[2].plot(k_vals, b0_stars, 'b-', lw=2)
axes[2].axhline(b_true, color='r', ls='--', label=f'b_true={b_true}')
axes[2].axvline(1.0, color='r', ls='--', alpha=0.3)
axes[2].set_xlabel('k')
axes[2].set_ylabel('VarPro b0*')
axes[2].set_title('b0* projection')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
