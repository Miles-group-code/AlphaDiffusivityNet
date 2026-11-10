"""
Hyperparameter sweeps for the supplement: one (convention, b0, D-frequency) condition, many
seeds, optionally with one hyperparameter scaled relative to the paper value.

The paper settings are imported unchanged from the standard scripts
(Combined_Fickian_slurm.py, Combined_Ito_slurm.py: make_config, make_problem and the
per-method dto_config / pinn_config / bilo_config), and seed s gives the same data snapshot
as trial s of those scripts. So a baseline run reproduces the paper's recovery fits, and
a multiplier run is compared with them on identical data.

Environment inputs:

    CONV       fickian | ito             picks the module for configs and problem
    B0         true source strength, e.g. 250 or 1000
    FREQ       D(x) = 0.1 + 0.05 sin(2 pi FREQ x), e.g. 3 or 4
    SWEEP      baseline                  paper settings
               wreg   (with MULT=<c>)    smoothness weight wreg_smooth -> c * paper value
               wphys  (with MULT=<c>)    PINN residual weight w_phys   -> c * paper value
    METHODS    comma list from dto,pinn,bilo (default: all methods the sweep applies to)
    N_TRIALS   number of seeds (default 100), starting at SEED0 (default 1)
    N_WORKERS  worker processes (SLURM_CPUS_PER_TASK takes precedence when set)
    SWEEP_DIR  results root (default: sweep_results/ next to this script)
    PILOT=1    very short training, results under PILOT_<tag>/ (smoke test only)

Each fit is written to its own pickle,

    <SWEEP_DIR>/<Conv>_b<B0>_f<FREQ>_<sweep>[_x<mult>]/<method>_s<seed>.pkl
    e.g. sweep_results/Fickian_b1000_f3_wphys_x0.5/pinn_s007.pkl

holding D, u, b0, the particles, the training history, the compressed solver log, run
time, settings, problem parameters and environment. A rerun skips every fit that is
already on disk with the same settings fingerprint and stops with an error if a saved fit
was made with different settings or code, so an interrupted job resumes where it stopped.
Failed fits write <name>.err; non-finite fits are kept as <name>.nonfinite.pkl and count
as failures. The exit status is nonzero if any fit failed.

Examples:

    CONV=fickian B0=1000 FREQ=3 SWEEP=wreg MULT=10 METHODS=dto N_TRIALS=4 python Sweep_hyperparams.py
    PILOT=1 CONV=ito B0=250 FREQ=4 METHODS=pinn N_TRIALS=2 python Sweep_hyperparams.py   # smoke test
"""

import io
import os
import bz2
import sys
import glob
import time
import pickle
import socket
import hashlib
import platform
import importlib
import traceback
import contextlib
import functools
import subprocess
import multiprocessing as mp
from dataclasses import asdict, is_dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)
from interface import solve

# Convention -> (module with the paper settings, label used in result folder names).
CONVENTIONS = {"fickian": ("Combined_Fickian_slurm", "Fickian"),
               "ito": ("Combined_Ito_slurm", "Ito")}
SWEEP_KEY = {"wreg": "wreg_smooth", "wphys": "w_phys"}
SWEEP_METHODS = {"baseline": ["bilo", "pinn", "dto"],     # slowest first, so the pool
                 "wreg": ["bilo", "pinn", "dto"],         # doesn't end on a tail of BiLO fits
                 "wphys": ["pinn"]}

# Short settings for a local smoke test of the whole pipeline (not for the figures).
PILOT_OVERRIDES = {"max_iters": 60, "pretrain_iters": 40, "scalar_fit_iters": 40,
                   "early_burnin": 10_000}

OUT_ROOT = os.environ.get("SWEEP_DIR") or os.path.join(HERE, "sweep_results")


def conv_module(conv):
    return importlib.import_module(CONVENTIONS[conv][0])


def run_tag(cond, sweep, mult, pilot):
    tag = (f"{CONVENTIONS[cond['conv']][1]}_b{cond['b0']}_f{cond['freq']:g}_{sweep}"
           + ("" if sweep == "baseline" else f"_x{mult:g}"))
    return ("PILOT_" + tag) if pilot else tag


def trial_settings(cond, method, sweep, mult, pilot):
    paper = getattr(conv_module(cond["conv"]), f"{method}_config")
    settings = dict(paper)
    if sweep != "baseline":
        key = SWEEP_KEY[sweep]
        if key not in settings:
            raise SystemExit(f"{method}_config of {cond['conv']} has no {key!r} to scale")
        settings[key] = mult * settings[key]
    if pilot:
        settings.update(PILOT_OVERRIDES)
    return settings


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def code_version():
    """Git commit of the framework, or None when not run from a git checkout."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


@functools.lru_cache(maxsize=None)
def code_md5(conv):
    """Line-ending-normalized md5 of the framework files, the settings module and this
    runner, so every fit records exactly which code produced it."""
    files = sorted(glob.glob(os.path.join(REPO_ROOT, "*.py")))
    files += [os.path.join(HERE, CONVENTIONS[conv][0] + ".py"), os.path.abspath(__file__)]
    out = {}
    for f in files:
        with open(f, "rb") as fh:
            out[os.path.relpath(f, REPO_ROOT).replace(os.sep, "/")] = \
                hashlib.md5(fh.read().replace(b"\r", b"")).hexdigest()
    return out


def fingerprint(cond, method, seed, settings):
    """Everything that defines a fit. Resume reuses a saved fit only if its fingerprint
    matches, so a stale result from changed settings or changed code is never mistaken
    for a new one (the code hashes keep two solver versions out of one folder)."""
    cfg = conv_module(cond["conv"]).make_config()
    return {"conv": cond["conv"], "method": method, "seed": seed, "b0": cond["b0"],
            "freq": cond["freq"], "settings": settings,
            "config": repr(asdict(cfg)) if is_dataclass(cfg) else repr(vars(cfg)),
            "code_md5": code_md5(cond["conv"])}


def existing_ok(outpath, fp):
    if not os.path.exists(outpath):
        return False
    try:
        with open(outpath, "rb") as f:
            old = pickle.load(f)
    except Exception:
        return False          # unreadable: fit again
    if old.get("fingerprint") != fp:
        raise RuntimeError(f"{outpath} was made with different settings or code. Move it "
                           "away or use a new results folder before rerunning.")
    return True


def run_trial(job):
    """One (method, seed) fit. Mirrors run_single_trial_internal in the standard scripts
    and additionally keeps the particles, training history, console log, and wall time."""
    cond, method, seed, settings, outpath = job
    fp = fingerprint(cond, method, seed, settings)
    if existing_ok(outpath, fp):
        return outpath, "skipped (exists)", 0.0
    # make_problem reads FREQ from the environment, which workers inherit from main().
    assert float(os.environ["FREQ"]) == cond["freq"], "FREQ changed after launch"
    mod = conv_module(cond["conv"])

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device, dtype = "cpu", torch.float64

    t0 = time.time()
    log = io.StringIO()
    try:
        custom_config = mod.make_config()
        problem = mod.make_problem(seed, cond["b0"], device, dtype)
        # The solvers print every log step (tens of MB per run), so the console output
        # is kept per fit, compressed, instead of going to the job log.
        # It is also the only record of the PINN/BiLO pretraining phase.
        with contextlib.redirect_stdout(log):
            sol = solve(problem, method=method, config=custom_config, **settings)
        runtime = time.time() - t0

        x_plot = sol.x_res.detach().cpu().numpy().reshape(-1)
        x_true = problem.x_grid.detach().cpu().numpy().reshape(-1)
        d_true = np.asarray(problem.d_true).reshape(-1)
        d_true_on_plot = np.interp(x_plot, x_true, d_true)

        # Scalar curves of the training history, plus the relative L2 error of each logged
        # D snapshot (the snapshots themselves are dropped to keep the files small).
        hist = {}
        for k, v in sol.history.items():
            if k in ("d_snapshots", "d_snap_iters"):
                continue
            try:
                hist[k] = np.asarray(v, dtype=np.int64 if k == "iter" else np.float64)
            except (TypeError, ValueError):
                pass
        snaps = sol.history.get("d_snapshots")
        snaps = [] if snaps is None else list(snaps)
        snap_iters = sol.history.get("d_snap_iters")
        hist["d_snap_iters"] = np.asarray([] if snap_iters is None else snap_iters, dtype=np.int64)
        hist["d_err_l2_rel"] = np.array([rel_l2(np.asarray(s).reshape(-1), d_true_on_plot)
                                         for s in snaps])

        d_pred = sol.d_pred.detach().cpu().numpy().reshape(-1)
        u_pred = sol.u_pred.detach().cpu().numpy().reshape(-1)
        result = {
            "fingerprint": fp,
            "method": method,
            "seed": seed,
            "settings": settings,
            "problem": {"conv": cond["conv"], "alpha": problem.alpha, "mu": problem.mu,
                        "z": problem.source_location, "bc_type": problem.bc_type,
                        "b_true": float(problem.b_true),
                        "d_profile_params": (0.1, 0.05, float(cond["freq"]))},
            "runtime_s": runtime,
            "x_plot": x_plot,
            "d_pred": d_pred,
            "u_pred": u_pred,
            "b_pred": float(sol.b0_star),
            "b_true": float(problem.b_true),
            "x_true": x_true,
            "d_true": d_true,
            "u_true": np.asarray(problem.u_true).reshape(-1),
            "x_particles": problem.particles.x_particles.detach().cpu().numpy().reshape(-1),
            "m_obs": int(problem.particles.m_obs),
            "d_err_l2_rel": rel_l2(d_pred, d_true_on_plot),
            "history": hist,
            "log_bz2": bz2.compress(log.getvalue().encode("utf-8")),
            "env": {"host": socket.gethostname(), "platform": platform.platform(),
                    "python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "git": code_version(),
                    "code_md5": code_md5(cond["conv"])},
        }
        finite = bool(np.all(np.isfinite(d_pred)) and np.all(np.isfinite(u_pred))
                      and np.isfinite(result["b_pred"]))
    except Exception:
        err = traceback.format_exc()
        with open(outpath.replace(".pkl", ".err"), "w", encoding="utf-8") as f:
            f.write(err + "\n--- solver output (tail) ---\n" + log.getvalue()[-20000:])
        return outpath, "FAILED: " + err.strip().splitlines()[-1], time.time() - t0

    # A non-finite fit is kept for diagnosis under another name and reported as failed,
    # so it is neither plotted by accident nor silently skipped on resume.
    final = outpath if finite else outpath.replace(".pkl", ".nonfinite.pkl")
    tmp = final + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(result, f)
    os.replace(tmp, final)   # atomic, so a killed job never leaves a half-written trial
    if not finite:
        return outpath, "FAILED: non-finite D, u or b0", runtime
    # A success replaces any record of an earlier failed attempt at the same fit.
    for stale in (outpath.replace(".pkl", ".err"), outpath.replace(".pkl", ".nonfinite.pkl")):
        if os.path.exists(stale):
            os.remove(stale)
    return outpath, f"ok  b0={result['b_pred']:.1f}  relL2(D)={result['d_err_l2_rel']:.3f}", runtime


def required_env(name):
    val = os.environ.get(name, "").strip()
    if not val:
        raise SystemExit(f"set {name} in the environment (see the header of this file)")
    return val


def main():
    conv = required_env("CONV").lower()
    if conv not in CONVENTIONS:
        raise SystemExit(f"CONV must be one of {list(CONVENTIONS)}, got {conv!r}")
    cond = {"conv": conv, "b0": int(required_env("B0")), "freq": float(required_env("FREQ"))}
    sweep = os.environ.get("SWEEP", "baseline")
    if sweep not in SWEEP_METHODS:
        raise SystemExit(f"SWEEP must be one of {list(SWEEP_METHODS)}, got {sweep!r}")
    mult = float(os.environ.get("MULT", "1"))
    if sweep == "baseline" and mult != 1.0:
        raise SystemExit("SWEEP=baseline takes no MULT")
    if sweep != "baseline" and mult == 1.0:
        raise SystemExit(f"SWEEP={sweep} with MULT=1 is the baseline; use SWEEP=baseline")
    n_trials = int(os.environ.get("N_TRIALS", "100"))
    seed0 = int(os.environ.get("SEED0", "1"))
    pilot = os.environ.get("PILOT", "0") == "1"
    methods = ([m.strip() for m in os.environ["METHODS"].split(",") if m.strip()]
               if os.environ.get("METHODS") else SWEEP_METHODS[sweep])
    bad = [m for m in methods if m not in SWEEP_METHODS[sweep]]
    if bad or not methods:
        raise SystemExit(f"methods {bad or methods} are not part of SWEEP={sweep}")

    tag = run_tag(cond, sweep, mult, pilot)
    outdir = os.path.join(OUT_ROOT, tag)
    os.makedirs(outdir, exist_ok=True)

    jobs = [(cond, m, s, trial_settings(cond, m, sweep, mult, pilot),
             os.path.join(outdir, f"{m}_s{s:03d}.pkl"))
            for m in methods for s in range(seed0, seed0 + n_trials)]

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("N_WORKERS")
                    or max(1, mp.cpu_count() - 1))
    print(f"[run] {tag} | {CONVENTIONS[conv][0]} | methods {methods} | seeds "
          f"{seed0}..{seed0 + n_trials - 1} | {len(jobs)} fits on {n_workers} workers", flush=True)
    for m in methods:
        print(f"  {m}: {trial_settings(cond, m, sweep, mult, pilot)}", flush=True)
    for f, h in code_md5(conv).items():
        print(f"  md5 {h}  {f}", flush=True)

    # Check saved fits before starting any work: a fingerprint mismatch stops here.
    pending = [j for j in jobs if not existing_ok(j[4], fingerprint(*j[:4]))]
    print(f"[run] {len(jobs) - len(pending)} fits already on disk, {len(pending)} to run",
          flush=True)

    # ProcessPoolExecutor rather than mp.Pool: if a worker is killed (e.g. out of memory)
    # it raises BrokenProcessPool instead of hanging until the wall-time limit. Finished
    # fits are already on disk, so resubmitting the task resumes.
    t_start = time.time()
    n_failed = 0
    if pending:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(pending)),
                                 mp_context=mp.get_context("spawn")) as ex:
            futures = [ex.submit(run_trial, j) for j in pending]
            for i, fut in enumerate(as_completed(futures), 1):
                path, status, rt = fut.result()
                n_failed += status.startswith("FAILED")
                print(f"[{i:4d}/{len(pending)}] {os.path.basename(path)}  {status}  "
                      f"({rt / 60:.1f} min)", flush=True)
    print(f"[run] done in {(time.time() - t_start) / 3600:.2f} h, {n_failed} failed -> {outdir}",
          flush=True)
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
