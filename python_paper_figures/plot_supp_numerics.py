"""
Supplement figures and tables on the numerics.

Plots the hyperparameter sweeps of Sweep_hyperparams.py (sweep_results/) and re-reads the
recovery pickles of the Combined_*_slurm.py runs (CCombined_*_data.pkl), without
recomputing any fit. Quantities that need a forward solve (data loss at the true D, PDE
consistency of the network solutions, example data snapshots) come from
supp_numerics_derived.pkl, written by compute_supp_numerics_derived.py; the figures that
need it are skipped if it is absent.

The sweeps are on the Fickian / Neumann, b0 = 1000, D(x) = 0.1 + 0.05 sin(6 pi x)
condition, seeds 1..100, so the x1 setting is exactly the main-text fit. A DTO-only
lambda_smooth sweep at b0 = 250 is added to the error panels of fig_supp_smoothness.pdf.

    fig_supp_smoothness.pdf   lambda_smooth x {0.1 .. 10} for DTO, PINN, BiLO
    fig_supp_pinn_wres.pdf    PINN residual weight w_res x {0.1 .. 100}
    fig_supp_training.pdf     training curves at the main-text settings
    fig_supp_snapshots.pdf    example particle snapshots: u_true, each trial's own fits (u and D)
    fig_supp_failure_modes.pdf collapse / boundary growth of single-snapshot fits
    tab_supp_hyperparams.tex  per-method loss weights and learning rates
    tab_supp_errors.tex       per-trial and median-curve errors, all five conditions
    supp_numerics_summary.csv every sweep number used in the text

Run from this folder:  python plot_supp_numerics.py

Optional environment switches:

    SWEEP_DIR    results root of Sweep_hyperparams.py (default: sweep_results/ here)
    OUT_DIR      output folder, also where supp_numerics_derived.pkl is read from (default:
                 this script's folder; a relative path is taken relative to it).
"""

import os
import csv
import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

# Same look as plot_recovery_figures.py (the main-text recovery figures).
plt.rcParams["mathtext.fontset"] = "cm"
COLORS = {"DTO": "#E69F00", "PINN": "#56B4E9", "BiLO": "#009E73"}
STYLES = {"DTO": "-", "PINN": "--", "BiLO": "-."}
MARKERS = {"DTO": "o", "PINN": "s", "BiLO": "^"}
METHODS = ["DTO", "PINN", "BiLO"]
FILE_KEY = {"DTO": "dto", "PINN": "pinn", "BiLO": "bilo"}

ROW_TITLE = 19
PANEL_TITLE = 15
PANEL_LABEL = 19
AXIS_LABEL = 19
TICK = 13
Y_TICK = 11
LEGEND = 11

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.environ.get("SWEEP_DIR") or os.path.join(HERE, "sweep_results")
# b0 outlier filter of the main-text recovery figures (FILTER_MULT in plot_recovery_figures.py):
# a fit is dropped if b0 exceeds 1.5 times the median b0 over the trials, for every method.
FILTER_MULT = {"DTO": 1.5, "PINN": 1.5, "BiLO": 1.5}
# The sweep statistics keep every fit and count the ones above this multiple of b_true.
COUNT_MULT = 1.5
COUNT_TAG = f"$b_0>{COUNT_MULT:g}b_{{\\mathrm{{true}}}}$"
OUT_DIR = os.path.join(HERE, os.environ.get("OUT_DIR", ""))
DERIVED = os.path.join(OUT_DIR, "supp_numerics_derived.pkl")
MULTS = [0.1, 0.2, 0.5, 1, 2, 5, 10]
SHOWN_MULTS = (0.1, 1, 10)          # the D(x) curves drawn in the sensitivity figures
# The w_res sweep goes on to x20, x50, x100 with 15 fits each (seeds 1-15).
WRES_EXT = (20, 50, 100)
WRES_EXT_N = 15
WRES_MULTS = MULTS + list(WRES_EXT)
N_BOOT = 1000                       # bootstrap resamples for the median-curve error
SEED_BOOT = 0


# ----------------------------------------------------------------------------- loading

def run_dir(sweep, mult, b0=1000):
    """Folder of one setting, as written by Sweep_hyperparams.py."""
    tag = "baseline" if mult == 1 else f"{sweep}_x{mult:g}"
    return os.path.join(SWEEP_DIR, f"Fickian_b{b0}_f3_{tag}")


_cache = {}


def load(sweep, mult, method, b0=1000):
    """All 100 trials of one (setting, method), ordered by seed (15 for the w_res extension)."""
    key = (sweep if mult != 1 else "baseline", mult, method, b0)
    if key not in _cache:
        folder = run_dir(sweep, mult, b0)
        n_expected = WRES_EXT_N if (sweep == "wphys" and mult in WRES_EXT) else 100
        files = sorted(glob.glob(os.path.join(folder, f"{FILE_KEY[method]}_s*.pkl")))
        files = [f for f in files if not f.endswith(".nonfinite.pkl")]
        trials = []
        for f in files:
            with open(f, "rb") as fh:
                trials.append(pickle.load(fh))
        trials.sort(key=lambda t: t["seed"])
        if len(trials) != n_expected:
            raise RuntimeError(f"{folder}: {len(trials)} {method} trials, expected {n_expected}")
        _cache[key] = trials
    return _cache[key]


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def summarize(trials):
    """Per-trial and ensemble statistics of one (setting, method)."""
    x = trials[0]["x_plot"]
    d_true = np.interp(x, trials[0]["x_true"], trials[0]["d_true"])
    D = np.array([t["d_pred"] for t in trials])
    err = np.array([t["d_err_l2_rel"] for t in trials])
    b = np.array([t["b_pred"] for t in trials])
    b_true = trials[0]["b_true"]

    # Error of the pointwise-median curve (what the recovery figures show), with a
    # bootstrap over trials for its uncertainty.
    med_err = rel_l2(np.median(D, axis=0), d_true)
    rng = np.random.default_rng(SEED_BOOT)
    boot = [rel_l2(np.median(D[rng.integers(0, len(D), len(D))], axis=0), d_true)
            for _ in range(N_BOOT)]

    # Squared per-trial error splits into the error of the mean curve (bias) and the
    # spread around it (variance), both relative to ||D_true||^2.
    n2 = np.sum(d_true ** 2)
    bias2 = np.sum((D.mean(0) - d_true) ** 2) / n2
    var = np.sum(D.var(0)) / n2
    return {
        "x": x, "d_true": d_true,
        "d_median": np.median(D, axis=0),
        "d_p10": np.percentile(D, 10, axis=0), "d_p90": np.percentile(D, 90, axis=0),
        "err": err, "err_med": np.median(err),
        "err_p10": np.percentile(err, 10), "err_p90": np.percentile(err, 90),
        "medcurve_err": med_err,
        # 10th-90th percentiles, the same span as the bands everywhere else in the paper
        "medcurve_lo": np.percentile(boot, 10), "medcurve_hi": np.percentile(boot, 90),
        "bias2": bias2, "var": var,
        "b": b, "b_true": b_true,
        "b_med": np.median(b), "b_p10": np.percentile(b, 10), "b_p90": np.percentile(b, 90),
        "b_relerr_med": np.median(np.abs(b - b_true) / b_true),
        # Fits with b0 > COUNT_MULT * b_true. They are kept in every statistic here and
        # counted instead.
        "n_b0_gt3": int(np.sum(b > COUNT_MULT * b_true)),
        "mean_d_med": np.median(np.trapz(D, x, axis=1)),
        "runtime_min_med": np.median([t["runtime_s"] for t in trials]) / 60,
    }


def flat_error(trials, b0=1000):
    """Reference errors that need no spatial information: a constant D equal to the true
    mean, and the per-trial constant D from the scalar pre-fit (iteration 0 of DTO)."""
    x = trials[0]["x_plot"]
    d_true = np.interp(x, trials[0]["x_true"], trials[0]["d_true"])
    flat_true_mean = rel_l2(np.full_like(d_true, np.trapz(d_true, x)), d_true)
    prefit = np.array([t["history"]["d_err_l2_rel"][0] for t in load("baseline", 1, "DTO", b0)])
    return flat_true_mean, np.median(prefit)


# ----------------------------------------------------------------------------- drawing helpers

def shade(color, toward, amount):
    c, t = np.array(to_rgb(color)), np.array(to_rgb(toward))
    return tuple((1 - amount) * c + amount * t)


def panel_tag(ax, letter):
    ax.text(0.03, 0.95, f"({letter})", transform=ax.transAxes,
            fontsize=PANEL_LABEL, fontweight="bold", va="top", ha="left")


def style_axes(ax, title, xlabel, ylabel, letter, grid_alpha=0.2):
    ax.set_title(title, fontsize=PANEL_TITLE)
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL)
    ax.tick_params(axis="x", labelsize=TICK)
    ax.tick_params(axis="y", labelsize=Y_TICK)
    ax.grid(True, alpha=grid_alpha)
    panel_tag(ax, letter)


def mult_axis(ax, mults=MULTS):
    ax.set_xscale("log")
    ax.set_xticks(mults)
    ax.set_xticklabels([f"{m:g}" for m in mults])
    ax.minorticks_off()
    ax.axvline(1, color="0.6", lw=1, zorder=0)


def draw_curves(ax, stats_by_mult, method, ylim, letter, title, mult_symbol=None,
                bands=False, true_label=True, legend_cols=1):
    """Median recovered D(x) at the smallest, main-text and largest multiplier. All three
    are solid (dashed and dash-dot mean PINN and BiLO elsewhere); the multiplier is coded
    by a lighter (x0.1) and darker (x10) shade of the method color. With bands=True the
    10-90th percentile band of each of the three is added in that curve's shade, so the
    three spreads overlap; a band that runs past ylim is cut at the axis. mult_symbol, if
    given, names the multiplied weight in the first legend entry."""
    s1 = stats_by_mult[1]
    c = COLORS[method]
    ax.plot(s1["x"], s1["d_true"], color="black", ls=":", lw=2.5, zorder=4,
            label="true $D(x)$" if true_label else "_nolegend_")
    look = {0.1: dict(color=shade(c, "white", 0.45), lw=1.8, zorder=2),
            1: dict(color=c, lw=3.2, zorder=3),
            10: dict(color=shade(c, "black", 0.45), lw=1.8, zorder=2)}
    if bands:
        for m in SHOWN_MULTS:
            s = stats_by_mult[m]
            ax.fill_between(s["x"], np.maximum(s["d_p10"], 0), np.maximum(s["d_p90"], 0),
                            color=look[m]["color"], alpha=0.15, lw=0, zorder=1)
    for m in SHOWN_MULTS:
        if m == 1:
            label = "main text"
        elif m == SHOWN_MULTS[0] and mult_symbol:
            label = f"${mult_symbol}\\;\\times{m:g}$"
        else:
            label = f"$\\times{m:g}$"
        ax.plot(stats_by_mult[m]["x"], stats_by_mult[m]["d_median"], ls="-", label=label,
                **look[m])
    ax.set_xlim(0, 1)
    ax.set_ylim(*ylim)
    style_axes(ax, title, "$x$", "$D(x)$", letter)
    h, l = ax.get_legend_handles_labels()
    if legend_cols == 2:
        # Columns fill top to bottom: (true D, main text) | (x0.1, x10).
        order = [0, 2, 1, 3] if len(h) == 4 else list(range(len(h)))
        h, l = [h[i] for i in order], [l[i] for i in order]
    ax.legend(h, l, fontsize=LEGEND, loc="upper right", ncol=legend_cols)


def d_ylim(all_stats, band=False, head=1.12):
    """head > 1 leaves room above the curves for the legend."""
    top = max(max(s["d_true"].max(), s["d_median"].max(),
                  np.maximum(s["d_p90"], 0).max() if band else 0) for s in all_stats)
    return (-0.02 * top, head * top)


# The b0 = 250 DTO sweep, drawn next to the b0 = 1000 methods in fig_supp_smoothness.
B250_COLOR = shade("#E69F00", "white", 0.45)


def error_series(stats, methods, stats_b250=None):
    """Line specs for the error-vs-multiplier panels: one per method, plus the b0 = 250
    DTO sweep (lighter DTO shade, hollow circles) when given."""
    out = [dict(label=m, by=stats[m], color=COLORS[m], ls=STYLES[m], marker=MARKERS[m],
                mfc=COLORS[m]) for m in methods]
    if stats_b250 is not None:
        out.append(dict(label="DTO, $b_0 = 250$", by=stats_b250, color=B250_COLOR, ls="-",
                        marker="o", mfc="white"))
    return out


def plot_error_bars(ax, ser, y, lo, hi, offset=0.0, label=None, ytop=None, mults=MULTS):
    """Median-error line with bars from lo to hi. Small horizontal offsets keep the bars
    of the series apart. A bar that runs past ytop is cut at the axis and marked with an
    arrowhead and its true upper end."""
    xs = np.array(mults) * (1 + offset)
    y, lo, hi = map(np.asarray, (y, lo, hi))
    clipped = np.zeros(len(xs), bool) if ytop is None else hi > ytop
    # A cut bar is drawn past the axis, so the axes clip it and its upper cap.
    hi_drawn = np.where(clipped, 1.1 * ytop if ytop is not None else hi, hi)
    ax.errorbar(xs, y, yerr=np.array([y - lo, hi_drawn - y]), color=ser["color"],
                ls=ser["ls"], lw=2.5, marker=ser["marker"], ms=7, mfc=ser["mfc"],
                mec=ser["color"], mew=1.8, capsize=3,
                label=ser["label"] if label is None else label)
    for xi, hv in zip(xs[clipped], hi[clipped]):
        ax.plot(xi, ytop, marker="^", ms=8, color=ser["color"], clip_on=False, zorder=5)
        ax.annotate(f"{hv:.1f}", (xi, ytop), xytext=(4, -3), textcoords="offset points",
                    fontsize=LEGEND - 1, color=ser["color"], ha="left", va="top",
                    fontweight="bold")


def plot_trial_error(ax, ser, offset=0.0, label=None, ytop=None, mults=MULTS):
    """Median per-trial error with 10-90th percentile bars across the 100 trials."""
    e = [ser["by"][k] for k in mults]
    plot_error_bars(ax, ser, [s["err_med"] for s in e], [s["err_p10"] for s in e],
                    [s["err_p90"] for s in e], offset, label, ytop, mults)


def plot_median_error(ax, ser, offset=0.0, label=None, ytop=None, mults=MULTS):
    """Error of the median D with 10-90th percentile bootstrap bars."""
    e = [ser["by"][k] for k in mults]
    plot_error_bars(ax, ser, [s["medcurve_err"] for s in e], [s["medcurve_lo"] for s in e],
                    [s["medcurve_hi"] for s in e], offset, label, ytop, mults)


def error_ylim(series, cap):
    """One range for the per-trial and median-curve error panels, fitted to the bar ends
    below cap. Bars above cap (BiLO's per-trial spread at weak smoothing) are cut at the
    axis and marked by plot_error_bars."""
    ends = [v for ser in series for s in ser["by"].values()
            for v in (s["err_p90"], s["medcurve_hi"]) if v <= cap]
    return (0, 1.08 * max(ends))


FLAT_LABEL = "constant $D$ = true mean"


def flat_line(ax, flat, label=True):
    ax.axhline(flat, color="0.35", ls=":", lw=2, label=FLAT_LABEL if label else "_nolegend_")


def legend_ref_last(ax, **kw):
    """Legend with the reference line(s) after the data series (errorbar containers would
    otherwise be listed after the plain lines)."""
    h, l = ax.get_legend_handles_labels()
    order = sorted(range(len(l)), key=lambda i: l[i] == FLAT_LABEL)
    ax.legend([h[i] for i in order], [l[i] for i in order], fontsize=LEGEND, **kw)


def save(fig, name, extra=()):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    kw = dict(dpi=300, bbox_inches="tight", pad_inches=0.15, bbox_extra_artists=list(extra))
    try:
        fig.savefig(path, **kw)
    except PermissionError:
        # The PDF is open in a viewer that locks it (Windows): write next to it instead.
        name = name.replace(".pdf", "_new.pdf")
        fig.savefig(os.path.join(OUT_DIR, name), **kw)
        print(f"WARNING: target locked, wrote {name} instead")
    plt.close(fig)
    print(f"wrote {name}")


# ----------------------------------------------------------------------------- figures

def figure_smoothness(stats, refs, stats_b250=None):
    """lambda_smooth sweep: median D(x) per method (top), per-trial error and error of the
    median D against the multiplier (bottom), with the b0 = 250 DTO sweep when given."""
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 6)
    top = [fig.add_subplot(gs[0, 2 * j:2 * j + 2]) for j in range(3)]
    ax_d, ax_e = fig.add_subplot(gs[1, :3]), fig.add_subplot(gs[1, 3:])

    # 10-90 bands at all three shown multipliers. The range is fitted to the medians and
    # the main-text band; the BiLO x0.1 band (boundary growth) is cut at the top.
    top_d = max(max(stats[m][k]["d_median"].max() for k in SHOWN_MULTS)
                for m in METHODS)
    top_d = max(top_d, max(np.maximum(stats[m][1]["d_p90"], 0).max() for m in METHODS))
    ylim = (-0.02 * top_d, 1.15 * top_d)
    for j, (ax, m) in enumerate(zip(top, METHODS)):
        draw_curves(ax, stats[m], m, ylim, "abc"[j], m, bands=True, true_label=(j == 0),
                    legend_cols=2 if j == 0 else 1)

    xlabel = r"$\lambda_{\mathrm{smooth}} / \bar{\lambda}_{\mathrm{smooth}}$"
    series = error_series(stats, METHODS, stats_b250)
    # (d) bars: 10-90th percentiles over the 100 trials; (e) bars: bootstrap. Same range.
    ylim = error_ylim(series, cap=1.8)
    n = len(series)
    for i, ser in enumerate(series):
        off = 0.06 * (i - (n - 1) / 2)
        plot_trial_error(ax_d, ser, offset=off, ytop=ylim[1])
        plot_median_error(ax_e, ser, offset=off, ytop=ylim[1])
    flat_line(ax_d, refs[0])
    flat_line(ax_e, refs[0])
    for ax, title, letter in ((ax_d, "Per-trial error of $D$", "d"),
                              (ax_e, "Error of the median $D$", "e")):
        mult_axis(ax)
        ax.set_xlim(0.05, 14)           # room left of x0.1 for the panel letter
        ax.set_ylim(*ylim)
        style_axes(ax, title, xlabel, r"rel. $L^2$ error", letter)
    # One legend for (d) and (e), in the free upper part of (e).
    legend_ref_last(ax_e, loc="upper right", ncol=2)
    save(fig, "fig_supp_smoothness.pdf")


def figure_pinn_wres(stats, refs):
    """PINN residual-weight sweep: median D(x), failure counts, error of D."""
    fig = plt.figure(figsize=(16, 5.2), constrained_layout=True)
    axes = fig.subplots(1, 3)
    xlabel = r"$w_{\mathrm{res}} / \bar{w}_{\mathrm{res}}$"

    draw_curves(axes[0], stats, "PINN", d_ylim([stats[k] for k in SHOWN_MULTS], band=True, head=1.3),
                "a", "PINN", r"w_{\mathrm{res}}", bands=True, legend_cols=2)

    # (b) How often the fit fails. "Collapse" is an operational definition:
    # D < mean(D_true) / FAIL_FACTOR on at least COLLAPSE_FRAC of [0, 1].
    # Percent, not counts: the x20-x100 points have 15 fits, the rest 100.
    ax = axes[1]
    pct = []
    for k in WRES_MULTS:
        trials = load("wphys", k, "PINN")
        pct.append({q: 100 * v / len(trials) for q, v in failure_counts(trials).items()})
    ax.plot(WRES_MULTS, [c["collapse"] for c in pct], color=COLORS["PINN"], ls="-", lw=2.5,
            marker=MARKERS["PINN"], ms=7, label="collapse of $D$")
    ax.plot(WRES_MULTS, [c["b0"] for c in pct], color=COLORS["PINN"], ls=":", lw=2.5,
            marker=MARKERS["PINN"], ms=7, mfc="white", mew=1.8, label=COUNT_TAG)
    mult_axis(ax, WRES_MULTS)
    ax.set_ylim(0, 115)                 # headroom for the panel letter above 96%
    ax.set_yticks(range(0, 101, 20))
    style_axes(ax, "Failed fits (PINN)", xlabel, "% of fits", "b")
    ax.legend(fontsize=LEGEND, loc="upper right")

    # (c) Per-trial error (median, 10-90 bars over trials) and the error of the median D
    # (bootstrap bars), offset so the bars stay apart.
    ax = axes[2]
    ser = error_series({"PINN": stats}, ["PINN"])[0]
    plot_trial_error(ax, ser, offset=-0.04, label="per-trial error (median)", mults=WRES_MULTS)
    plot_median_error(ax, dict(ser, mfc="white"), offset=0.04, label="error of the median $D$",
                      mults=WRES_MULTS)
    flat_line(ax, refs[0])
    mult_axis(ax, WRES_MULTS)
    top = max(max(s["err_p90"], s["medcurve_hi"]) for s in stats.values())
    ax.set_ylim(0, 1.22 * max(top, refs[0]))
    style_axes(ax, "Error of $D$ (PINN)", xlabel, r"rel. $L^2$ error", "c")
    legend_ref_last(ax, loc="upper right")
    save(fig, "fig_supp_pinn_wres.pdf")


# ----------------------------------------------------------------------------- tables

def fmt_num(v):
    if v is None:
        return "--"
    if isinstance(v, str):
        return v
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    if -2 <= e <= 3:
        return f"{v:g}"
    mant = v / 10 ** e
    return f"$10^{{{e}}}$" if np.isclose(mant, 1) else f"${mant:g}\\times10^{{{e}}}$"


def table_hyperparams():
    import sys
    sys.path.insert(0, os.path.dirname(HERE))
    from config import Config
    sys.path.insert(0, HERE)
    import Combined_Fickian_slurm as fick_mod
    import Combined_Ito_slurm as ito_mod

    base = Config()
    defaults = {"wreg_smooth": base.reg.wreg_smooth, "wreg_scale": base.reg.wreg_scale,
                "w_phys": base.reg.w_phys, "w_jump": base.reg.w_jump, "w_bc": base.reg.w_bc,
                "w_resgrad": base.reg.w_resgrad, "lr_d_pre": base.train.lr_d_pre,
                "lr_lower_pre": base.train.lr_lower_pre, "lr_d_fine": base.train.lr_d_fine,
                "lr_lower_fine": base.train.lr_lower_fine}
    ito = {k: getattr(ito_mod, k) for k in ("dto_config", "pinn_config", "bilo_config")}
    fick = {k: getattr(fick_mod, k) for k in ("dto_config", "pinn_config", "bilo_config")}
    # With constant-D pretraining D is frozen at the scalar pre-fit, so PINN has no
    # pretraining learning rate for D (lr_d_pre does not act).
    for cfg in (fick["pinn_config"], ito["pinn_config"]):
        assert cfg.get("pinn_pretrain", base.train.pinn_pretrain) == "constant_d"

    # (row label, config key, methods that use it, Neumann only). The scripts set
    # lower_tol = 0.01 through make_config(); DTO has no network learning rates or
    # pretraining, and the boundary penalty only enters with Neumann conditions.
    rows = [
        (r"$\lambda_{\mathrm{smooth}}$", "wreg_smooth", ("dto", "pinn", "bilo"), False),
        (r"$\lambda_{\mathrm{scale}}$", "wreg_scale", ("dto", "pinn", "bilo"), False),
        (r"$w_{\mathrm{res}}$", "w_phys", ("pinn",), False),
        (r"$w_{\mathrm{jump}}$", "w_jump", ("pinn", "bilo"), False),
        (r"$w_{\mathrm{bc}}$", "w_bc", ("pinn", "bilo"), True),
        (r"$w_{\mathrm{rgrad}}$", "w_resgrad", ("bilo",), False),
        (r"lower-level tolerance $\epsilon$", "lower_tol", ("bilo",), False),
        (r"learning rate, $D$ (pretrain)", "lr_d_pre", ("pinn", "bilo"), False),
        (r"learning rate, $\hat u$ or $v$ (pretrain)", "lr_lower_pre", ("pinn", "bilo"), False),
        (r"learning rate, $D$", "lr_d_fine", ("dto", "pinn", "bilo"), False),
        (r"learning rate, $\hat u$ or $v$", "lr_lower_fine", ("pinn", "bilo"), False),
    ]
    lines = [r"\begin{tabular}{l ccc ccc}", r"\toprule",
             r" & \multicolumn{3}{c}{It\^o, Dirichlet} & \multicolumn{3}{c}{Fickian, Neumann} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r" & DTO & PINN & BiLO & DTO & PINN & BiLO \\", r"\midrule"]
    for label, key, used, neumann_only in rows:
        cells = []
        for cond, cfgs in (("ito", ito), ("fick", fick)):
            for m in ("dto", "pinn", "bilo"):
                if m not in used or (neumann_only and cond == "ito"):
                    cells.append("--")
                elif key == "lower_tol":
                    cells.append(fmt_num(0.01))
                elif m == "pinn" and key == "lr_d_pre":
                    cells.append("--")
                else:
                    cells.append(fmt_num(cfgs[f"{m}_config"].get(key, defaults[key])))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT_DIR, "tab_supp_hyperparams.tex"), "w", encoding="utf-8") as f:
        f.write("% Written by plot_supp_numerics.py from the run scripts. Do not edit by hand.\n")
        f.write("% PINN: constant-D pretraining, D is fixed at the scalar pre-fit and frozen, "
                "so it has no pretraining learning rate.\n")
        f.write("\n".join(lines) + "\n")
    print("wrote tab_supp_hyperparams.tex")


RECOVERY_CONDITIONS = [
    ("It\\^o, Dirichlet", r"$\sin 8\pi x$", 250, "CCombined_Ito_b250_t100_4_data.pkl"),
    ("It\\^o, Dirichlet", r"$\sin 8\pi x$", 1000, "CCombined_Ito_b1000_t100_4_data.pkl"),
    ("Fickian, Neumann", r"$\sin 8\pi x$", 1000, "CCombined_Fickian_b1000_t100_4_data.pkl"),
    ("Fickian, Neumann", r"$\sin 6\pi x$", 250, "CCombined_Fickian_b250_t100_3_data.pkl"),
    ("Fickian, Neumann", r"$\sin 6\pi x$", 1000, "CCombined_Fickian_b1000_t100_3_data.pkl"),
]


def recovery_stats(path):
    with open(os.path.join(HERE, path), "rb") as f:
        data = pickle.load(f)
    out = {}
    for m in METHODS:
        res = data[f"{FILE_KEY[m]}_results"]
        x = np.ravel(res[0]["x_plot"])
        d_true = np.interp(x, np.ravel(res[0]["x_true"]), np.ravel(res[0]["d_true"]))
        D = np.array([np.ravel(r["d_pred"]) for r in res])
        b = np.array([float(r["b_pred"]) for r in res])
        b_true = float(res[0]["b_true"])
        err = np.array([rel_l2(d, d_true) for d in D])
        cutoff = FILTER_MULT[m] * np.median(b)
        n_filtered = int(np.sum(b > cutoff))
        # The statistics are over the fits the recovery figures keep, so the error of the
        # median D is the error of the plotted median.
        keep = b <= cutoff
        D, b, err = D[keep], b[keep], err[keep]
        out[m] = {"err": err, "medcurve": rel_l2(np.median(D, axis=0), d_true),
                  "b_rel": np.abs(b - b_true) / b_true,
                  "n_filtered": n_filtered,
                  "flat": rel_l2(np.full_like(d_true, np.trapz(d_true, x)), d_true)}
    return out


def table_errors():
    lines = [r"\begin{tabular}{ll r l ccc}", r"\toprule",
             r"Model & $D_{\mathrm{true}}-0.1$ & $b_0$ & Method & per-trial error of $D$ & "
             r"error of median $D$ & per-trial error of $b_0$ \\", r"\midrule"]
    total_filtered = 0
    removed = []
    for i, (model, prof, b0, path) in enumerate(RECOVERY_CONDITIONS):
        st = recovery_stats(path)
        # Shade the smallest error among the methods in each column, compared at the
        # printed precision so a tie shades both cells. \cellcolor needs colortbl and a
        # color named "mint" in the document preamble.
        best_err = min(round(float(np.median(st[m]["err"])), 2) for m in METHODS)
        best_med = min(round(float(st[m]["medcurve"]), 2) for m in METHODS)
        best_b = min(round(float(np.median(st[m]["b_rel"])), 3) for m in METHODS)

        def shade(value, best, digits):
            return r"\cellcolor{mint}" if round(float(value), digits) == best else ""

        for j, m in enumerate(METHODS):
            s = st[m]
            total_filtered += s["n_filtered"]
            if s["n_filtered"]:
                removed.append(f"{path} {m}: {s['n_filtered']}")
            lead = f"{model} & $0.05{prof[1:]} & {b0}" if j == 0 else " & & "
            lines.append(
                f"{lead} & {m} & {shade(np.median(s['err']), best_err, 2)}{np.median(s['err']):.2f} "
                f"[{np.percentile(s['err'], 10):.2f}, {np.percentile(s['err'], 90):.2f}] & "
                f"{shade(s['medcurve'], best_med, 2)}{s['medcurve']:.2f} & "
                f"{shade(np.median(s['b_rel']), best_b, 3)}{np.median(s['b_rel']):.3f} "
                f"[{np.percentile(s['b_rel'], 10):.3f}, {np.percentile(s['b_rel'], 90):.3f}] \\\\")
        lines.append(f" & & & constant $D$ & {st['DTO']['flat']:.2f} & {st['DTO']['flat']:.2f} & -- \\\\")
        if i < len(RECOVERY_CONDITIONS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT_DIR, "tab_supp_errors.tex"), "w", encoding="utf-8") as f:
        f.write("% Written by plot_supp_numerics.py from the CCombined_*_data.pkl runs. "
                "Do not edit by hand.\n")
        f.write(f"% Relative L2 errors, median [10th, 90th percentile] over the trials kept by "
                f"the b0 outlier filter of the recovery figures (drop b0 > {FILTER_MULT['DTO']:g} "
                f"times the median over trials, all methods), which removed {total_filtered} of 1500 trials"
                + (": " + "; ".join(removed) if removed else "") + ".\n")
        f.write("\n".join(lines) + "\n")
    print(f"wrote tab_supp_errors.tex (outlier filter removed {total_filtered} trials)")


def write_summary(stats_wreg, stats_wres, refs, stats_b250=None, refs_b250=None):
    path = os.path.join(OUT_DIR, "supp_numerics_summary.csv")
    keys = ["err_med", "err_p10", "err_p90", "medcurve_err", "medcurve_lo", "medcurve_hi",
            "bias2", "var", "b_med", "b_p10", "b_p90", "b_relerr_med", "n_b0_gt3", "mean_d_med",
            "runtime_min_med"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # The b0 count column is named after COUNT_MULT (n_b0_gt1.5).
        header = [f"n_b0_gt{COUNT_MULT:g}" if q == "n_b0_gt3" else q for q in keys]
        w.writerow(["sweep", "method", "mult", "paired_diff_med", "frac_seeds_better"] + header)
        tables = [("wreg", stats_wreg), ("wres", {"PINN": stats_wres})]
        if stats_b250 is not None:
            tables.append(("wreg_b250", {"DTO": stats_b250}))
        for sweep, table in tables:
            for m, by_mult in table.items():
                for k in by_mult:
                    s = by_mult[k]
                    # Trials are ordered by seed, so the 15-fit w_res extension (seeds 1-15)
                    # pairs with the first 15 main-text fits.
                    diff = s["err"] - by_mult[1]["err"][:len(s["err"])]
                    better = np.nan if k == 1 else np.mean(diff < 0)
                    w.writerow([sweep, m, k, f"{np.median(diff):.4f}", f"{better:.2f}"]
                               + [f"{s[q]:.4f}" for q in keys])
        w.writerow([])
        w.writerow(["reference", "constant D = true mean", f"{refs[0]:.4f}"])
        w.writerow(["reference", "scalar pre-fit (median over seeds)", f"{refs[1]:.4f}"])
        if refs_b250 is not None:
            w.writerow(["reference_b250", "constant D = true mean", f"{refs_b250[0]:.4f}"])
            w.writerow(["reference_b250", "scalar pre-fit (median over seeds)", f"{refs_b250[1]:.4f}"])
    print("wrote supp_numerics_summary.csv")


# ----------------------------------------------------------------------------- main

def main():
    stats_wreg = {m: {k: summarize(load("wreg", k, m)) for k in MULTS} for m in METHODS}
    stats_wres = {k: summarize(load("wphys", k, "PINN")) for k in WRES_MULTS}
    refs = flat_error(load("baseline", 1, "DTO"))

    stats_b250 = {k: summarize(load("wreg", k, "DTO", b0=250)) for k in MULTS}
    refs_b250 = flat_error(load("baseline", 1, "DTO", b0=250), b0=250)
    # fig_supp_smoothness draws one "constant D = true mean" line for both b0.
    assert np.isclose(refs_b250[0], refs[0], atol=1e-3)

    figure_smoothness(stats_wreg, refs, stats_b250)
    figure_pinn_wres(stats_wres, refs)
    figure_failure_modes()
    table_hyperparams()
    table_errors()
    write_summary(stats_wreg, stats_wres, refs, stats_b250, refs_b250)

    derived = None
    if os.path.exists(DERIVED):
        with open(DERIVED, "rb") as f:
            derived = pickle.load(f)
        figure_training(derived)
        figure_snapshots(derived)
    else:
        print(f"skipped training and snapshot figures: {os.path.basename(DERIVED)} not found")
    caption_numbers(derived, refs, refs_b250)


# Failure modes of single-snapshot fits, with constant thresholds set symmetrically by a
# factor FAIL_FACTOR around the mean of the true D (0.1, so 0.02 and 0.5):
#   collapse  D(x) < mean(D_true) / FAIL_FACTOR on at least COLLAPSE_FRAC of [0, 1]
#   growth    D(x) > FAIL_FACTOR * mean(D_true) somewhere on [0, 1]
FAIL_FACTOR = 5
COLLAPSE_FRAC = 0.1
N_EXAMPLES = 12


def failure_masks(trials):
    """Per-fit, per-point collapse and growth masks on each fit's x_plot grid."""
    D, low, high = [], [], []
    for t in trials:
        d_mean = np.mean(t["d_true"])
        D.append(t["d_pred"])
        low.append(t["d_pred"] < d_mean / FAIL_FACTOR)
        high.append(t["d_pred"] > FAIL_FACTOR * d_mean)
    return np.array(D), np.array(low), np.array(high)


def failure_counts(trials):
    _, low, high = failure_masks(trials)
    b = np.array([t["b_pred"] for t in trials])
    return {"collapse": int(np.sum(np.mean(low, axis=1) >= COLLAPSE_FRAC)),
            "blowup": int(np.sum(high.any(axis=1))),
            "b0": int(np.sum(b > COUNT_MULT * trials[0]["b_true"]))}


def figure_failure_modes():
    fig = plt.figure(figsize=(16, 5.2), constrained_layout=True)
    axes = fig.subplots(1, 3)

    # All three panels use the main-text settings (the baseline fits); the low-weight
    # extremes are in the sweep figures.
    # (a) PINN: D collapses towards zero in the sparse tails, where the soft PDE residual is
    # nearly free. Trials 1-12, unselected.
    ax = axes[0]
    trials = load("baseline", 1, "PINN")
    x, d_true = trials[0]["x_plot"], np.interp(trials[0]["x_plot"], trials[0]["x_true"], trials[0]["d_true"])
    for t in trials[:N_EXAMPLES]:
        ax.plot(x, t["d_pred"], color=COLORS["PINN"], lw=1.2, alpha=0.7)
    ax.plot([], [], color=COLORS["PINN"], lw=1.2, label=f"PINN, trials 1-{N_EXAMPLES}")
    ax.plot(x, d_true, color="black", ls=":", lw=2.5, label="true $D(x)$")
    ax.axhline(np.mean(trials[0]["d_true"]) / FAIL_FACTOR, color="0.35", ls=(0, (6, 3)), lw=1.5,
               label=f"collapse threshold $\\langle D_{{\\mathrm{{true}}}}\\rangle/{FAIL_FACTOR}$")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 10)                         # deeper collapses run off the bottom
    ax.set_xlim(0, 1)
    style_axes(ax, "Collapse of $D$ (PINN)", "$x$", "$D(x)$", "a")
    ax.legend(fontsize=LEGEND, loc="upper right")

    # (b) BiLO: D grows near the reflecting boundaries. Every fit that crosses the growth
    # threshold is drawn.
    ax = axes[1]
    trials = load("baseline", 1, "BiLO")
    grown = [t for t in trials if t["d_pred"].max() > FAIL_FACTOR * np.mean(t["d_true"])]
    for t in grown:
        ax.plot(x, t["d_pred"], color=COLORS["BiLO"], lw=1.2, alpha=0.7)
    ax.plot([], [], color=COLORS["BiLO"], lw=1.2, label=f"BiLO, all {len(grown)} growth fits")
    ax.plot(x, d_true, color="black", ls=":", lw=2.5, label="true $D(x)$")
    ax.axhline(FAIL_FACTOR * np.mean(trials[0]["d_true"]), color="0.35", ls="--", lw=1.5,
               label=f"growth threshold ${FAIL_FACTOR}\\langle D_{{\\mathrm{{true}}}}\\rangle$")
    ax.set_xlim(0, 1)
    top = max(t["d_pred"].max() for t in grown)
    ax.set_ylim(0, 1.45 * top)                    # headroom for the legend
    style_axes(ax, "Growth of $D$ (BiLO)", "$x$", "$D(x)$", "b")
    ax.legend(fontsize=LEGEND, loc="upper right")

    # (c) Where D fails at the main-text settings, against the expected particle density.
    # Collapse (D < mean D_true / FAIL_FACTOR) is drawn upward and growth (D > FAIL_FACTOR mean D_true)
    # downward on a mirrored axis, so the two opposite failures are never added together;
    # a method's growth curve is drawn only if some fit has it.
    ax = axes[2]
    top = bottom = 0
    for m in METHODS:
        trials = load("baseline", 1, m)
        _, low, high = failure_masks(trials)
        counts = failure_counts(trials)
        frac = np.mean(low, axis=0)
        top = max(top, frac.max())
        ax.plot(x, frac, color=COLORS[m], ls=STYLES[m], lw=2.5,
                label=f"{m}, collapse ({counts['collapse']}/100 fits)")
        if counts["blowup"]:
            frac = np.mean(high, axis=0)
            bottom = max(bottom, frac.max())
            ax.fill_between(x, -frac, 0, color=COLORS[m], alpha=0.15, lw=0)
            ax.plot(x, -frac, color=COLORS[m], ls=STYLES[m], lw=1.8,
                    label=f"{m}, growth ({counts['blowup']}/100 fits)")
    ax.axhline(0, color="black", lw=1)
    # Same scale both ways; the growth side only gets the room it needs.
    ylim = (-max(0.1, 1.6 * bottom), min(1, 1.25 * top))
    ref = load("baseline", 1, "DTO")[0]
    u = np.interp(x, ref["x_true"], ref["u_true"])
    ax2 = ax.twinx()
    ax2.fill_between(x, 0, u / u.max(), color="0.85", zorder=0)
    # Same zero as the main axis; density peak below the legend.
    f0 = -ylim[0] / (ylim[1] - ylim[0])
    ax2.set_ylim(-1.5 * f0 / (1 - f0), 1.5)
    ax2.set_yticks([])
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.fill_between([], [], color="0.85", label="expected particle density")
    ax.set_xlim(0, 1)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):g}"))
    style_axes(ax, "Where $D$ fails", "$x$",
               "fraction of fits\n$\\leftarrow$ growth      collapse $\\rightarrow$", "c")
    ax.legend(fontsize=LEGEND, loc="upper center")
    save(fig, "fig_supp_failure_modes.pdf")


MAX_ITERS = 7000
N_THIN = 25                       # per-seed curves drawn in the training figure
# Right end of the training plots. on_grid holds a stopped run at its final value, which
# would draw flat tails past the stop. Every DTO fit and most BiLO fits stop by ~5,100
# iterations; many PINN fits run on to the 7,000-iteration cap and still change there.
X_ITERS = {"DTO": 5200, "PINN": MAX_ITERS, "BiLO": 5200}


def on_grid(trials, key, grid):
    """One history curve per seed on a common iteration grid. A run that stopped early
    keeps its final value, since that is the fit it returns."""
    out = np.empty((len(trials), len(grid)))
    for i, t in enumerate(trials):
        h = t["history"]
        it = h["d_snap_iters"] if key == "d_err_l2_rel" else h["iter"]
        out[i] = np.interp(grid, it, h[key])
    return out


def term_on_grid(trials, fn, grid):
    """A weighted loss term, fn(settings, history), per seed on the iteration grid."""
    out = np.empty((len(trials), len(grid)))
    for i, t in enumerate(trials):
        h = {k: np.asarray(v, float) for k, v in t["history"].items()}
        out[i] = np.interp(grid, h["iter"], fn(t["settings"], h))
    return out


# Weighted loss terms of the training objectives, as assembled in method_dto.py,
# method_pinn.py and method_bilo.py (s = the run's settings, h = its history). The logged
# res, jump, bc, rgrad, jump_rgrad, bc_grad, reg_smooth and reg_scale are unweighted;
# PINN's phys is already w_res res + w_jump jump + w_bc bc. With w_data = 1 and
# wreg_d_neumann = 0 (make_config defaults), the terms add up to the logged objective:
#   DTO   total = data + lam_smooth reg_smooth + lam_scale reg_scale
#   PINN  total = data + w_res res + w_jump jump + w_bc bc + lam_smooth reg_smooth + lam_scale reg_scale
#   BiLO  upper = data + lam_smooth reg_smooth + lam_scale reg_scale
#         lower = res + w_jump jump + w_bc bc + w_rgrad (rgrad + jump_rgrad + bc_grad)
# check_loss_bookkeeping() asserts this for every logged iteration of every fit.
# Each entry: (label, fn, shade toward black (+) or white (-), line width, marker).
SMOOTH_TERM = (r"$\lambda_{\mathrm{smooth}}\mathcal{R}_{\mathrm{smooth}}$",
               lambda s, h: s["wreg_smooth"] * h["reg_smooth"], 0, 2.5, None)
SCALE_TERM = (r"$\lambda_{\mathrm{scale}}\mathcal{R}_{\mathrm{scale}}$",
              lambda s, h: s["wreg_scale"] * h["reg_scale"], -0.5, 1.8, "v")
LOSS_TERMS = {
    "DTO": [SMOOTH_TERM, SCALE_TERM],
    "PINN": [
        (r"$w_{\mathrm{res}}\mathcal{L}_{\mathrm{res}}$",
         lambda s, h: s["w_phys"] * h["res"], 0.5, 2.5, "D"),
        (r"$w_{\mathrm{jump}}\mathcal{L}_{\mathrm{jump}} + w_{\mathrm{bc}}\mathcal{L}_{\mathrm{bc}}$",
         lambda s, h: s["w_jump"] * h["jump"] + s.get("w_bc", 1.0) * h["bc"], 0.25, 1.8, "x"),
        SMOOTH_TERM, SCALE_TERM],
    "BiLO": [
        (SMOOTH_TERM[0] + " (upper)",) + SMOOTH_TERM[1:],
        (SCALE_TERM[0] + " (upper)",) + SCALE_TERM[1:],
        (r"$\mathcal{L}_{\mathrm{op}}$ (lower)", lambda s, h: h["lower"], 0.5, 2.5, "D"),
        (r"$w_{\mathrm{rgrad}}\mathcal{L}_{\mathrm{rgrad}}$ (part of $\mathcal{L}_{\mathrm{op}}$)",
         lambda s, h: s["w_resgrad"] * (h["rgrad"] + h["jump_rgrad"] + h["bc_grad"]),
         0.25, 1.8, "x")],
}
LOWER_TOL = 0.01                  # make_config() sets lower_tol; it acts on the weighted lower loss


def check_loss_bookkeeping():
    """Assert that the weighted terms rebuild the logged objectives of the main-text fits
    (every seed, every logged iteration) and return the largest relative mismatch."""
    worst = 0.0
    for m in METHODS:
        for t in load("baseline", 1, m):
            s = t["settings"]
            h = {k: np.asarray(v, float) for k, v in t["history"].items()}
            reg = s["wreg_smooth"] * h["reg_smooth"] + s["wreg_scale"] * h["reg_scale"]
            if m == "DTO":
                pairs = [(h["total"], h["data"] + reg)]
            elif m == "PINN":
                phys = s["w_phys"] * h["res"] + s["w_jump"] * h["jump"] + s.get("w_bc", 1.0) * h["bc"]
                pairs = [(h["phys"], phys), (h["total"], h["data"] + phys + reg)]
            else:
                lower = (h["res"] + s["w_jump"] * h["jump"] + s.get("w_bc", 1.0) * h["bc"]
                         + s["w_resgrad"] * (h["rgrad"] + h["jump_rgrad"] + h["bc_grad"]))
                pairs = [(h["upper"], h["data"] + reg), (h["lower"], lower)]
            for logged, rebuilt in pairs:
                rel = np.max(np.abs(logged - rebuilt) / np.maximum(np.abs(logged), 1e-300))
                worst = max(worst, rel)
    assert worst < 1e-10, f"loss terms do not rebuild the logged objective (rel. {worst:.1e})"
    return worst


def draw_history(ax, grid, curves, color, ls, label=None, band=True, thin=True):
    for c in (curves[:N_THIN] if thin else []):
        ax.plot(grid, c, color=color, lw=0.6, alpha=0.25)
    if band:
        ax.fill_between(grid, np.percentile(curves, 10, axis=0),
                        np.percentile(curves, 90, axis=0), color=color, alpha=0.15, lw=0)
    ax.plot(grid, np.median(curves, axis=0), color=color, ls=ls, lw=2.5, label=label)


# The objective each method minimises over D (BiLO: the upper level; its lower-level loss
# is a separate objective, drawn in the bottom row of fig_supp_training).
TOTAL_KEY = {"DTO": "total", "PINN": "total", "BiLO": "upper"}


def figure_training(derived):
    """Training at the main-text settings (the baseline fits): data loss and total objective
    relative to the data loss at the true D (top), error of D (middle) and the weighted
    penalty terms (bottom), per method."""
    tl = derived["true_d_loss"]
    loss_true = np.asarray(tl["loss_true_d"], float)
    grid = np.arange(0, MAX_ITERS + 1, 10)
    flat, _ = flat_error(load("baseline", 1, "DTO"))

    data = {}
    for m in METHODS:
        trials = load("baseline", 1, m)
        assert [t["seed"] for t in trials] == list(tl["seed"])
        data[m] = {"excess": on_grid(trials, "data", grid) - loss_true[:, None],
                   "total": on_grid(trials, TOTAL_KEY[m], grid) - loss_true[:, None],
                   "err": on_grid(trials, "d_err_l2_rel", grid),
                   "medcurve": summarize(trials)["medcurve_err"]}

    def band_range(key):
        # Shared range fitted to the median and 10-90 band over each plotted window.
        lo, hi = np.inf, -np.inf
        for m in METHODS:
            c = data[m][key][:, grid <= X_ITERS[m]]
            lo = min(lo, np.percentile(c, 10, axis=0).min())
            hi = max(hi, np.percentile(c, 90, axis=0).max())
        return lo, hi

    lo, hi = band_range("excess")
    for m in METHODS:                   # the total-objective medians must fit as well
        med = np.median(data[m]["total"][:, grid <= X_ITERS[m]], axis=0)
        lo, hi = min(lo, med.min()), max(hi, med.max())
    # Extra room at the top keeps the panel letters off the early-iteration band.
    ylim_top = (lo - 0.05 * (hi - lo), hi + 0.22 * (hi - lo))
    _, hi = band_range("err")
    ylim_bot = (0, 1.08 * max(hi, flat))

    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    axes = fig.subplots(3, 3)
    for j, m in enumerate(METHODS):
        c, ls = COLORS[m], STYLES[m]
        xmax = X_ITERS[m]

        # Top: data loss (summed over particles) minus the same loss at the true D. Below
        # zero the fit explains this snapshot better than the true D does. The total
        # objective, with the same reference subtracted, is drawn as a median only; its gap
        # to the data-loss median is the weighted penalty total.
        ax = axes[0, j]
        ax.axhspan(ylim_top[0], 0, color="0.93", zorder=0, lw=0,
                   label="fits this snapshot better\nthan the true $D$" if j == 0 else "_nolegend_")
        draw_history(ax, grid, data[m]["excess"], c, ls, label=f"{m} data loss")
        ax.plot(grid, np.median(data[m]["total"], axis=0), color=shade(c, "black", 0.4),
                ls=ls, lw=1.6, label=f"{m} total objective")
        ax.axhline(0, color="black", ls=":", lw=2, label="true $D$" if j == 0 else "_nolegend_")
        ax.set_xlim(0, xmax)
        ax.set_ylim(*ylim_top)
        style_axes(ax, m, "iteration", "loss $-$ data loss at true $D$", "abc"[j])
        ax.legend(fontsize=LEGEND, loc="upper right")

        # Middle: relative L2 error of D along training.
        ax = axes[1, j]
        draw_history(ax, grid, data[m]["err"], c, ls, label=f"{m}, single fits")
        flat_line(ax, flat, label=(j == 0))
        ax.axhline(data[m]["medcurve"], color="black", ls="--", lw=1.8,
                   label="median $D$ over 100 fits")
        ax.set_xlim(0, xmax)
        ax.set_ylim(*ylim_bot)
        style_axes(ax, "", "iteration", r"rel. $L^2$ error of $D$", "def"[j])
        legend_ref_last(ax, loc="upper center")

    # Bottom: the weighted penalty terms of each objective (medians over the 100 fits).
    # The data term is a negative log-likelihood with a per-seed offset, so it cannot go
    # on the log axis; the top row compares it with the total objective instead.
    check_loss_bookkeeping()
    for j, m in enumerate(METHODS):
        c, ls = COLORS[m], STYLES[m]
        trials = load("baseline", 1, m)
        ax = axes[2, j]
        if m == "BiLO":
            ax.axhline(LOWER_TOL, color="0.35", ls=(0, (6, 3)), lw=1.5,
                       label=r"tolerance $\epsilon$ on $\mathcal{L}_{\mathrm{op}}$")
        for k, (label, fn, amount, lw, marker) in enumerate(LOSS_TERMS[m]):
            color = shade(c, "black" if amount > 0 else "white", abs(amount))
            med = np.median(term_on_grid(trials, fn, grid), axis=0)
            med = np.where(med > 0, med, np.nan)         # zero at a constant initial D
            ax.plot(grid, med, color=color, ls=ls, lw=lw, label=label, marker=marker,
                    ms=7, mfc="white" if marker == "v" else color, mec=color, mew=1.6,
                    markevery=(10 + 12 * k, 50))
        ax.set_yscale("log")
        ax.set_xlim(0, X_ITERS[m])
        ax.set_ylim(*TERM_YLIM)
        style_axes(ax, "", "iteration", "weighted penalty term", "ghi"[j])
        ax.legend(fontsize=LEGEND, loc="lower right")
    save(fig, "fig_supp_training.pdf")


TERM_YLIM = (1e-7, 1e2)              # bottom row of fig_supp_training


def caption_numbers(derived, refs, refs_b250):
    """Print the numbers quoted in the figure captions (stdout only)."""
    print("\nCaption numbers:")
    print(f"  failure rules: collapse = D < mean(D_true)/{FAIL_FACTOR} on >= {COLLAPSE_FRAC:g} of [0, 1]; "
          f"growth = D > {FAIL_FACTOR} mean(D_true) somewhere")
    for m in METHODS:
        trials = load("baseline", 1, m)
        c = failure_counts(trials)
        _, low, _ = failure_masks(trials)
        print(f"  main-text settings, {m}: collapse {c['collapse']}/100 "
              f"(below the collapse threshold anywhere: {int(np.sum(low.any(axis=1)))}/100), "
              f"growth {c['blowup']}/100")
    for k in (0.1, 1):
        print(f"  BiLO growth (D > {FAIL_FACTOR} mean D_true), lambda_smooth x{k:g}: "
              f"{failure_counts(load('wreg', k, 'BiLO'))['blowup']}/100")
    print("  PINN w_res sweep, collapse / b0>COUNT_MULT b_true: " + ", ".join(
        f"x{k:g} {failure_counts(load('wphys', k, 'PINN'))['collapse']}/"
        f"{failure_counts(load('wphys', k, 'PINN'))['b0']}"
        f"{'' if k not in WRES_EXT else f' (of {WRES_EXT_N})'}" for k in WRES_MULTS))
    print(f"  constant D = true mean: {refs[0]:.4f}; scalar pre-fit (median) b0=1000: {refs[1]:.4f}"
          + (f"; b0=250: {refs_b250[1]:.4f} (true mean {refs_b250[0]:.4f})" if refs_b250 else ""))
    for m in METHODS:
        print(f"  error of the median D at x1, {m}: {summarize(load('baseline', 1, m))['medcurve_err']:.4f}")
    if derived is not None:
        for key in ("pinn", "bilo"):
            fc = np.asarray(derived["forward_consistency"][key]["rel"])
            print(f"  ||u_NN-u_FDM||/||u_FDM|| {key}: median {np.median(fc):.4f} "
                  f"[{np.percentile(fc, 10):.4f}, {np.percentile(fc, 90):.4f}]")
    trials = load("baseline", 1, "BiLO")
    _, _, high = failure_masks(trials)
    x = trials[0]["x_plot"]
    print(f"  BiLO growth at main-text settings: {failure_counts(trials)['blowup']}/100 fits "
          f"({int(np.sum(high[:, x < 0.5].any(1)))} near x = 0, "
          f"{int(np.sum(high[:, x >= 0.5].any(1)))} near x = 1); largest pointwise "
          f"fraction {np.mean(high, axis=0).max():.2f}")
    print(f"  loss bookkeeping: weighted terms rebuild the logged objectives to rel. "
          f"{check_loss_bookkeeping():.1e}")
    grid = np.arange(0, MAX_ITERS + 1, 10)
    for m in METHODS:
        trials = load("baseline", 1, m)
        data = term_on_grid(trials, lambda s, h: h["data"], grid)
        total = term_on_grid(trials, lambda s, h: h[TOTAL_KEY[m]], grid)
        print(f"  weighted terms, {m}: median data-loss decrease over training "
              f"{np.median(data[:, 0] - data[:, -1]):.2f}")
        if derived is not None:
            ref = np.asarray(derived["true_d_loss"]["loss_true_d"], float)[:, None]
            med = np.median(total - ref, axis=0)[grid <= X_ITERS[m]]
            print(f"    total objective ({TOTAL_KEY[m]}) - data loss at true D, median: iteration 0 "
                  f"{med[0]:.2f}, end {med[-1]:.2f}, largest {med.max():.2f} at iteration "
                  f"{grid[np.argmax(med)]}; {int(np.sum(total[:, -1] < total[:, 0]))}/100 fits "
                  f"end below their start")
        for label, fn, *_ in LOSS_TERMS[m]:
            c = term_on_grid(trials, fn, grid)     # last column = each fit's final value
            print(f"    {label}: median at iteration 0 {np.median(c[:, 0]):.2e}, 1000 "
                  f"{np.median(c[:, 100]):.2e}, final {np.median(c[:, -1]):.2e} "
                  f"[{np.percentile(c[:, -1], 10):.1e}, {np.percentile(c[:, -1], 90):.1e}]")
    for m in METHODS:
        last = np.array([t["history"]["iter"][-1] for t in load("baseline", 1, m)])
        print(f"  training stop, {m}: {int(np.sum(last <= 5200))}/100 by 5,200 iterations, "
              f"{int(np.sum(last >= MAX_ITERS))}/100 reach the {MAX_ITERS} cap; plotted to {X_ITERS[m]}")


def figure_snapshots(derived):
    """Two example snapshots per source strength (the seeds whose fits recover D best,
    picked in compute_supp_numerics_derived.py; labelled trial 1 and 2). Top row
    of each b0: particle histogram scaled to an intensity, the true u, and per method the
    FDM solve of THAT trial's fitted D with that fit's own b0. Row below: the fitted D(x)
    of the same trial against the true D. The D rows share one y-range."""
    snaps = derived["snapshots"]
    if "seed_order" not in snaps[1000]:
        print("skipped fig_supp_snapshots.pdf: derived pkl has no per-trial fits "
              "(rerun compute_supp_numerics_derived.py)")
        return
    fig = plt.figure(figsize=(13, 17), constrained_layout=True)
    subfigs = fig.subfigures(2, 1, hspace=0.04)
    letters = iter("abcdefgh")
    titles = []
    d_hi = 1.25 * max(max(f["d_pred"].max() for f in snaps[b0]["seeds"][s]["fits"].values())
                      for b0 in (250, 1000) for s in snaps[b0]["seed_order"])
    d_hi = max(d_hi, 1.25 * max(snaps[b0]["d_true"].max() for b0 in (250, 1000)))
    for sf, b0 in zip(subfigs, (250, 1000)):
        snap = derived["snapshots"][b0]
        axes = sf.subplots(2, 2, sharex=True, gridspec_kw={"height_ratios": [1, 0.72]})
        u_axes, d_axes = axes
        u_axes[1].sharey(u_axes[0])
        d_axes[1].sharey(d_axes[0])
        top = 0
        order = snap["seed_order"]
        for i, (ax, seed) in enumerate(zip(u_axes, order)):
            sd = snap["seeds"][seed]
            xp, N = sd["x_particles"], sd["N"]
            edges = np.linspace(0, 1, 26)
            # counts / bin width, so the bars are on the scale of the intensity u(x)
            counts, _ = np.histogram(xp, bins=edges)
            dens = counts / np.diff(edges)
            ax.bar(edges[:-1], dens, width=np.diff(edges), align="edge", color="0.8",
                   edgecolor="white", label=f"particles ($N = {N}$)")
            ax.plot(xp, np.full_like(xp, -0.03 * dens.max()), "|", color="0.3", ms=8)
            ax.plot(snap["x_grid"], snap["u_true"], color="black", ls=":", lw=2.5,
                    label=r"true $u(x)$")
            for m in METHODS:
                f = sd["fits"][FILE_KEY[m]]
                ax.plot(f["x"], f["u_fdm"], color=COLORS[m], ls=STYLES[m], lw=2.5,
                        label=f"{m} fit")
                top = max(top, f["u_fdm"].max())
            top = max(top, dens.max(), snap["u_true"].max())
            ax.set_xlim(0, 1)
            style_axes(ax, f"trial {i + 1}", "", "intensity $u(x)$", next(letters))
            ax.legend(fontsize=LEGEND, loc="upper right")
        for ax in u_axes:
            ax.set_ylim(-0.07 * top, 1.42 * top)
        for ax, seed in zip(d_axes, order):
            fits = snap["seeds"][seed]["fits"]
            ax.plot(snap["x_grid"], snap["d_true"], color="black", ls=":", lw=2.5,
                    label=r"true $D(x)$")
            for m in METHODS:
                f = fits[FILE_KEY[m]]
                ax.plot(f["x"], f["d_pred"], color=COLORS[m], ls=STYLES[m], lw=2.5,
                        label=f"{m} fit")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, d_hi)
            style_axes(ax, "", "$x$", "$D(x)$", next(letters))
        d_axes[1].legend(fontsize=LEGEND, loc="upper right", ncol=2)
        titles.append(sf.suptitle(f"Fickian, $b_0 = {b0}$, $D(x) = 0.10 + 0.05\\sin(6\\pi x)$",
                                  fontsize=ROW_TITLE))
    save(fig, "fig_supp_snapshots.pdf", titles)


if __name__ == "__main__":
    main()
