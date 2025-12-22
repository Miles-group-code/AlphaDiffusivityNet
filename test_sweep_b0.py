"""
Sweep over b0 (birth rate) with particle data (PDE sampling).

Runs multiple replicates per parameter value to compare diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt

from interface import Problem, solve

# ============ FIXED SETTINGS ============
COMMON = {
    "max_iters": 5000,
    "lr_d": 1e-3,
    "lr_lower": 1e-3,
    "pretrain_iters": 1000,
    "wreg_scale": 1e-5,
    "wreg_smooth": 1e-4,
    "smoothness_type": "h1",
    "w_jump": 1.0,
    "w_resgrad": 0.1,
    "use_scheduler": True,
    "use_rff": True,
    "n_res": 101,
}

# ============ SWEEP CONFIG ============
SWEEP_LABEL = "b0 (true)"
SWEEP_VALUES = [25.0, 50.0, 100.0, 200.0, 400.0]
N_REPLICATES = 5
METHOD = "bilo"
METRICS = {
    "b0_rel_error": "b0 relative error",
    "d_u_rel_error": "D rel error (u-weighted)",
    "d_correlation": "D correlation",
}
JITTER_FRAC = 0.08

# ============ PROBLEM CONFIG ============
PROBLEM_KWARGS = {
    "alpha": 0.0,
    "mode": "particles",
    "d_profile": "sinusoidal",
    "d_profile_params": (0.1, 0.04, 4.0),
    "mu": 1.0,
    "m_obs": 500,
    "use_pde_sampling": True,
}

# ============ RUN SWEEP ============
print(f"Sweeping {SWEEP_LABEL} over {SWEEP_VALUES}")
print(f"Replicates per value: {N_REPLICATES}")
print(f"Method: {METHOD}\n")

results = {val: [] for val in SWEEP_VALUES}  # {param_value: [metrics_dict, ...]}

for val in SWEEP_VALUES:
    print(f"\n{SWEEP_LABEL} = {val}")
    settings = {**COMMON}

    for rep in range(N_REPLICATES):
        seed = 42 + rep  # Different seed per replicate
        problem = Problem.synthetic(
            **PROBLEM_KWARGS, seed=seed, verbose=False, b_true=val
        )

        solution = solve(problem, method=METHOD, **settings, verbose=False)
        metrics = solution.metrics(problem)
        missing = [key for key in METRICS if key not in metrics]
        if missing:
            raise KeyError(f"Missing metrics {missing}; check problem setup.")
        metric_row = {key: metrics[key] for key in METRICS}
        results[val].append(metric_row)

        print(
            "  rep {rep}: b0_rel={b0_rel:.3e}, "
            "D_u_rel={d_u_rel:.3e}, corr={corr:.3f}".format(
                rep=rep,
                b0_rel=metric_row["b0_rel_error"],
                d_u_rel=metric_row["d_u_rel_error"],
                corr=metric_row["d_correlation"],
            )
        )

# ============ COMPUTE STATISTICS ============
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)

stats = {}
for val in SWEEP_VALUES:
    stats[val] = {}
    print(f"{SWEEP_LABEL}={val:.2f}")
    for key, label in METRICS.items():
        values = [row[key] for row in results[val]]
        stats[val][key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        print(
            "  {label}: mean={mean:.4f} +/- {std:.4f} "
            "(range: {min:.4f} - {max:.4f})".format(
                label=label,
                mean=stats[val][key]["mean"],
                std=stats[val][key]["std"],
                min=stats[val][key]["min"],
                max=stats[val][key]["max"],
            )
        )

# ============ PLOT ============
fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 4), sharex=True)
if len(METRICS) == 1:
    axes = [axes]

rng = np.random.default_rng(0)
for ax, (key, label) in zip(axes, METRICS.items()):
    means = []
    for val in SWEEP_VALUES:
        values = np.array([row[key] for row in results[val]])
        jitter = 1.0 + JITTER_FRAC * rng.uniform(-1.0, 1.0, size=values.size)
        ax.scatter(
            val * jitter,
            values,
            alpha=0.7,
            s=40,
            color="tab:blue",
            edgecolor="none",
        )
        means.append(float(np.mean(values)))
    ax.plot(
        SWEEP_VALUES,
        means,
        color="black",
        linewidth=1.5,
        marker="D",
        markersize=5,
        label="mean",
    )
    ax.set_xscale("log")
    ax.set_xlabel(SWEEP_LABEL)
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    if ax is axes[0]:
        ax.legend()

fig.suptitle(
    f"Sweep over {SWEEP_LABEL} ({METHOD.upper()}, particles, {N_REPLICATES} reps)"
)
plt.tight_layout()
plt.savefig("sweep_b0_diagnostics.png", dpi=150)
plt.show()

print("\nSaved: sweep_b0_diagnostics.png")
