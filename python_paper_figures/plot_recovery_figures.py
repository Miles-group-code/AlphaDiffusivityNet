"""
Recovery figures for the diffusivity-identifiability paper.

Re-plots the Monte-Carlo recovery results produced by the Combined_* runs into the
manuscript figures, without recomputing. Each run saves a pickle

    CCombined_<convention>_b<B0>_t<N>_<freq>_data.pkl

holding {"dto_results", "pinn_results", "bilo_results", "n_trials"}. Each *_results
is a list of per-trial dicts with keys d_pred, x_plot, d_true, x_true, b_pred, b_true.

Three figures are produced, each row showing one condition (recovered D(x) on the
left, the distribution of the inferred source strength b0 on the right):

    fig_ito_recovery.pdf      Ito / Dirichlet,  sin(8 pi x),  b0 = 250 and 1000
    fig_fickian_highfreq.pdf  Fickian / Neumann, sin(8 pi x),  b0 = 1000
    fig_fickian_lowfreq.pdf   Fickian / Neumann, sin(6 pi x),  b0 = 250 and 1000

Run from this folder after the *_data.pkl files are present:  python plot_recovery_figures.py
"""

import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Render math ($x$, $D(x)$, $b_0$) in Computer Modern to match the LaTeX font of
# the analytical (non-identifiability) figures.
plt.rcParams["mathtext.fontset"] = "cm"

# Color-blind friendly palette (Okabe-Ito) and per-method line styles.
COLORS = {"DTO": "#E69F00", "PINN": "#56B4E9", "BiLO": "#009E73"}
STYLES = {"DTO": "-", "PINN": "--", "BiLO": "-."}
METHODS = ["DTO", "PINN", "BiLO"]
RESULT_KEY = {"DTO": "dto_results", "PINN": "pinn_results", "BiLO": "bilo_results"}

# A trial whose recovered b0 exceeds mult * b_true is treated as a non-converged
# outlier. PINN is filtered more strictly because its failures overshoot hardest.
FILTER_MULT = {"DTO": 10, "PINN": 3, "BiLO": 10}

# Font sizes, chosen for the panel sizes used in the manuscript. The y-tick
# numbers are kept small: neither the D(x) scale nor the b0 density matters
# quantitatively, so the axis labels read larger than the tick values.
ROW_TITLE = 19
PANEL_TITLE = 15
PANEL_LABEL = 19
AXIS_LABEL = 19
TICK = 13
Y_TICK = 11
LEGEND = 11


def prepare(path):
    """Load one run and reduce it to the statistics the panels need."""
    with open(path, "rb") as f:
        data = pickle.load(f)

    ref = data["dto_results"][0]
    cond = {
        "b_true": ref["b_true"],
        "x_true": np.asarray(ref["x_true"]).ravel(),
        "d_true": np.asarray(ref["d_true"]).ravel(),
        "d": {},
        "b": {},
    }
    for m in METHODS:
        res = data[RESULT_KEY[m]]
        kept = [r for r in res if r["b_pred"] <= FILTER_MULT[m] * r["b_true"]]
        d_arr = np.array([r["d_pred"] for r in kept])
        cond["d"][m] = {
            "x": np.asarray(res[0]["x_plot"]).ravel(),
            "median": np.median(d_arr, axis=0),
            "p10": np.percentile(d_arr, 10, axis=0),
            "p90": np.percentile(d_arr, 90, axis=0),
        }
        cond["b"][m] = np.array([r["b_pred"] for r in kept])
    return cond


def shared_d_ylim(conds):
    """A common D(x) y-range across conditions, so the b0 effect reads as a
    tightening of the same axis. The lower bound dips slightly below zero to show
    that the recovered profiles stay positive."""
    top = 0.0
    for c in conds:
        top = max(top, c["d_true"].max())
        for m in METHODS:
            s = c["d"][m]
            top = max(top, s["median"].max(), np.maximum(s["p90"], 0).max())
    return (-0.02 * top, 1.08 * top)


def panel_tag(ax, letter):
    ax.text(0.03, 0.95, f"({letter})", transform=ax.transAxes,
            fontsize=PANEL_LABEL, fontweight="bold", va="top", ha="left")


def draw_recovery(ax, cond, ylim, letter):
    """Recovered D(x): true profile, per-method median, 10-90th percentile band."""
    ax.plot(cond["x_true"], cond["d_true"], color="black", ls=":", lw=2.5,
            label="true $D(x)$")
    for m in METHODS:
        s = cond["d"][m]
        ax.plot(s["x"], s["median"], color=COLORS[m], ls=STYLES[m], lw=3,
                label=f"{m} median")
        ax.fill_between(s["x"], np.maximum(s["p10"], 0), np.maximum(s["p90"], 0),
                        color=COLORS[m], alpha=0.15)

    ax.set_xlim(0, 1)
    ax.set_ylim(*ylim)
    ax.margins(x=0)
    ax.set_title("Diffusivity", fontsize=PANEL_TITLE)
    ax.set_xlabel("$x$", fontsize=AXIS_LABEL)
    ax.set_ylabel("$D(x)$", fontsize=AXIS_LABEL)
    ax.tick_params(axis="x", labelsize=TICK)
    ax.tick_params(axis="y", labelsize=Y_TICK)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=LEGEND, loc="upper right")
    panel_tag(ax, letter)


def draw_distribution(ax, cond, letter):
    """Distribution of the inferred source strength b0 across trials."""
    b_true = cond["b_true"]
    ax.axvline(b_true, color="black", ls=":", lw=2, zorder=5,
               label=f"true $b_0 = {b_true}$")
    for m in METHODS:
        data = cond["b"][m]
        # Normalize each method to unit area so the per-method outlier filter
        # (different surviving counts) does not change the bar heights.
        ax.hist(data, bins=15, color=COLORS[m], alpha=0.5, edgecolor="white",
                linewidth=1, rwidth=0.85, density=True, label=f"{m} samples")
        kde = gaussian_kde(data)
        grid = np.linspace(data.min() - 50, data.max() + 50, 500)
        ax.plot(grid, kde(grid), color=COLORS[m], lw=2.5, alpha=0.9)
        ax.axvline(np.median(data), color=COLORS[m], ls=STYLES[m], lw=3,
                   label=f"{m} median: {np.median(data):.0f}")

    ax.set_title("Inferred source strength", fontsize=PANEL_TITLE)
    ax.set_xlabel("$b_0$", fontsize=AXIS_LABEL)
    ax.set_ylabel("density", fontsize=AXIS_LABEL)
    ax.tick_params(axis="x", labelsize=TICK)
    ax.tick_params(axis="y", labelsize=Y_TICK)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND, loc="upper right")
    panel_tag(ax, letter)


def draw_row(subfig, cond, ylim, title, letters):
    axes = subfig.subplots(1, 2)
    draw_recovery(axes[0], cond, ylim, letters[0])
    draw_distribution(axes[1], cond, letters[1])
    # Returned so it can be passed to savefig's bbox_extra_artists; suptitles are
    # otherwise excluded from the tight bounding box and clip at the page top.
    return subfig.suptitle(title, fontsize=ROW_TITLE)


def save(fig, outpath, extra_artists):
    fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.15,
                bbox_extra_artists=extra_artists)
    plt.close(fig)
    print(f"wrote {outpath}")


def figure_two_rows(top, bottom, titles, outpath):
    conds = [prepare(top["pkl"]), prepare(bottom["pkl"])]
    ylim = shared_d_ylim(conds)
    fig = plt.figure(figsize=(13, 10.5), constrained_layout=True)
    subfigs = fig.subfigures(2, 1, hspace=0.04)
    titles_drawn = [draw_row(subfigs[0], conds[0], ylim, titles[0], ("a", "b")),
                    draw_row(subfigs[1], conds[1], ylim, titles[1], ("c", "d"))]
    save(fig, outpath, titles_drawn)


def figure_one_row(cond_spec, title, outpath):
    cond = prepare(cond_spec["pkl"])
    ylim = shared_d_ylim([cond])
    fig = plt.figure(figsize=(13, 5.6), constrained_layout=True)
    axes = fig.subplots(1, 2)
    draw_recovery(axes[0], cond, ylim, "a")
    draw_distribution(axes[1], cond, "b")
    save(fig, outpath, [fig.suptitle(title, fontsize=ROW_TITLE)])


def main():
    figure_two_rows(
        top={"pkl": "CCombined_Ito_b250_t100_4_data.pkl"},
        bottom={"pkl": "CCombined_Ito_b1000_t100_4_data.pkl"},
        titles=["Itô, $b_0 = 250$, high-frequency $D(x)$",
                "Itô, $b_0 = 1000$, high-frequency $D(x)$"],
        outpath="fig_ito_recovery.pdf",
    )

    figure_one_row(
        cond_spec={"pkl": "CCombined_Fickian_b1000_t100_4_data.pkl"},
        title="Fickian, $b_0 = 1000$, high-frequency $D(x)$",
        outpath="fig_fickian_highfreq.pdf",
    )

    figure_two_rows(
        top={"pkl": "CCombined_Fickian_b250_t100_3_data.pkl"},
        bottom={"pkl": "CCombined_Fickian_b1000_t100_3_data.pkl"},
        titles=["Fickian, $b_0 = 250$, low-frequency $D(x)$",
                "Fickian, $b_0 = 1000$, low-frequency $D(x)$"],
        outpath="fig_fickian_lowfreq.pdf",
    )


if __name__ == "__main__":
    main()
