"""
Derived quantities for the supplement figures on the numerics.

Reads the baseline sweep fits (Fickian / Neumann, b0 = 1000, D(x) = 0.1 + 0.05 sin(6 pi x),
seeds 1..100, from Sweep_hyperparams.py) and the recovery pickles of
Combined_Fickian_slurm.py, and computes the quantities that need the framework (forward
solves, the PPP data loss), so the plotting script only has to load one pickle:

    true_d_loss          per seed: the DTO training data loss Ldata(ppp) at the TRUE D,
                         at a flat D = 0.1, and re-evaluated at the stored DTO fit (check)
    forward_consistency  per seed and method: ||u_pred - u_FDM[d_pred, b_pred]|| / ||u_FDM||
    snapshots            b0 in {250, 1000}, two example seeds each: particles, true u(x), and
                         each method's fit to that snapshot (D, b0, and the FDM solution of
                         that D with that b0)

The data loss is evaluated with the DTO solver's own pieces (DtoAlphaVarPro for the
forward solve on the DTO grid, varpro.get_b0_ppp for the profiled b0, varpro.ppp_nll),
assembled exactly as in method_dto.fit._compute_losses. Losses use the particles stored
in the sweep pickles; rebuilding them with make_problem (same seeding as the sweep)
reproduces them to round-off (~1e-15) on another machine, not necessarily bit for bit.

Inputs (SWEEP_DIR defaults to sweep_results/ next to this script):

    <SWEEP_DIR>/Fickian_b1000_f3_baseline/{dto,pinn,bilo}_s001..100.pkl
    <SWEEP_DIR>/Fickian_b250_f3_baseline/dto_s001..100.pkl
    CCombined_Fickian_b250_t100_3_data.pkl, CCombined_Fickian_b1000_t100_3_data.pkl

Run from anywhere (takes a few minutes, CPU only):

    python compute_supp_numerics_derived.py

Writes supp_numerics_derived.pkl next to this script (or into OUT_DIR, if set; a relative
path is taken relative to this script's folder).
"""

import io
import os
import sys
import time
import pickle
import contextlib

import numpy as np
import torch

os.environ.setdefault("FREQ", "3")
FREQ = int(float(os.environ["FREQ"]))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import physics
import varpro
from method_dto import DtoAlphaVarPro, D_MIN
from Combined_Fickian_slurm import make_problem

B0_SWEEP = 1000
N_SEEDS = 100
METHODS = ["dto", "pinn", "bilo"]
SWEEP_ROOT = os.environ.get("SWEEP_DIR") or os.path.join(HERE, "sweep_results")
SWEEP_DIR = os.path.join(SWEEP_ROOT, f"Fickian_b{B0_SWEEP}_f{FREQ}_baseline")
RECOVERY_PKL = {250: os.path.join(HERE, f"CCombined_Fickian_b250_t100_{FREQ}_data.pkl"),
                1000: os.path.join(HERE, f"CCombined_Fickian_b1000_t100_{FREQ}_data.pkl")}
OUT_DIR = os.path.join(HERE, os.environ.get("OUT_DIR", ""))
OUT_PATH = os.path.join(OUT_DIR, "supp_numerics_derived.pkl")

# Physics of make_problem (checked against the Problem object below).
ALPHA, MU, Z, BC = 1.0, 5.0, 0.5, "neumann"
DEVICE, DTYPE = "cpu", torch.float64


def build_problem(seed, b0):
    """Same seeding as Sweep_hyperparams.run_trial, so the snapshot is identical."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        problem = make_problem(seed, b0, DEVICE, DTYPE)
    assert (problem.alpha, problem.mu, problem.source_location, problem.bc_type) == (ALPHA, MU, Z, BC)
    return problem


def dto_grid(problem):
    """The DTO solver grid built in interface.solve (uniform, n_res = 201)."""
    x = problem.x_grid.view(-1)
    return torch.linspace(float(x[0]), float(x[-1]), 201, device=DEVICE, dtype=DTYPE)


def dto_data_loss(d_on_grid, x_res, x_particles, m_obs):
    """DTO data loss Ldata(ppp) for a given D on the DTO grid, with b0 profiled out.

    Mirrors method_dto.fit._compute_losses (particles branch). Returns (loss, b0_star)."""
    d = torch.as_tensor(np.asarray(d_on_grid, dtype=np.float64), device=DEVICE, dtype=DTYPE)
    xp = torch.as_tensor(np.asarray(x_particles, dtype=np.float64), device=DEVICE, dtype=DTYPE)
    model = DtoAlphaVarPro(x_res=x_res, alpha=ALPHA, mu=MU, sources=[Z], d_init=d,
                           d_min=D_MIN, bc_type=BC).to(device=DEVICE, dtype=DTYPE)
    interp = varpro.precompute_interp_1d(x_res, xp)
    with torch.no_grad():
        d_full = model.build_d_full()
        u_hat_unit = model.solve_unit_u_hat(d_full)
        integral_unit = torch.trapezoid(u_hat_unit.view(-1), x_res.view(-1))
        b0_star = varpro.get_b0_ppp(xp.numel(), m_obs, integral_unit)
        u_hat_obs = varpro.interpolate_1d_precomputed(u_hat_unit, *interp)
        loss = varpro.ppp_nll(u_hat_obs, b0_star, m_obs, integral_unit)
    return float(loss.item()), float(b0_star.item())


def fdm(d, x, b0):
    """Framework FDM forward solve (physics.fdm_solve_alpha, same scheme as DTO)."""
    return physics.fdm_solve_alpha(np.asarray(d), ALPHA, MU, np.asarray(x), float(b0), (Z,), bc_type=BC)


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def load_fit(method, seed):
    with open(os.path.join(SWEEP_DIR, f"{method}_s{seed:03d}.pkl"), "rb") as f:
        return pickle.load(f)


def final_logged(fit):
    """True if the last history entry is the step whose D was returned (early stop or
    max_iters on a log step). BiLO stops one step after its last log, so it is False there."""
    import bz2
    log = bz2.decompress(fit["log_bz2"]).decode("utf-8", errors="replace")
    key = "Early stopping triggered at step "
    i = log.rfind(key)
    if i < 0:
        return int(fit["history"]["iter"][-1]) == int(fit["settings"]["max_iters"]), None
    stop = int(log[i + len(key):].split(".")[0])
    return int(fit["history"]["iter"][-1]) == stop, stop


# =============================================================================
# 1 + 2. Per-seed loss at the true D, and forward consistency of each method's fit
# =============================================================================
def per_seed():
    true_rows = []
    fc = {m: {k: np.full(N_SEEDS, np.nan) for k in
              ("rel", "rel_b0_ppp", "rel_shape", "b_pred", "b0_ppp", "u_fdm_err_rle", "hist_u_fdm_err")}
          for m in METHODS}
    # Regenerated vs stored particles. A rebuild on another machine agrees to round-off
    # (~1e-15), not necessarily bit for bit; the stored particles are used below.
    particle_maxdiff = np.zeros(N_SEEDS)
    particle_methods_identical = np.zeros(N_SEEDS, dtype=bool)

    for s in range(1, N_SEEDS + 1):
        problem = build_problem(s, B0_SWEEP)
        x_res = dto_grid(problem)
        xp_new = problem.particles.x_particles.detach().cpu().numpy().reshape(-1)
        m_obs = int(problem.particles.m_obs)
        fits = {m: load_fit(m, s) for m in METHODS}
        xp = fits["dto"]["x_particles"]          # the data the solvers actually saw
        n = xp.size
        assert n == xp_new.size and m_obs == fits["dto"]["m_obs"]
        particle_maxdiff[s - 1] = np.max(np.abs(xp - xp_new)) if n else 0.0
        particle_methods_identical[s - 1] = all(np.array_equal(xp, fits[m]["x_particles"])
                                                for m in METHODS)

        # The true D is on problem.x_grid, which is the same 201-point linspace as x_res.
        x_true = problem.x_grid.detach().cpu().numpy().reshape(-1)
        assert np.array_equal(x_true, x_res.numpy())
        d_true = np.asarray(problem.d_true).reshape(-1)

        dto = fits["dto"]
        assert np.allclose(dto["x_plot"], x_res.numpy(), rtol=0, atol=1e-15)
        loss_true, b0_true_prof = dto_data_loss(d_true, x_res, xp, m_obs)
        loss_flat, _ = dto_data_loss(np.full_like(d_true, 0.1), x_res, xp, m_obs)
        loss_check, _ = dto_data_loss(dto["d_pred"], x_res, xp, m_obs)
        dto_logged, dto_stop = final_logged(dto)
        row = {"seed": s, "n_particles": n,
               "loss_true_d": loss_true, "loss_flat_d": loss_flat,
               "loss_dto_final_check": loss_check,
               "dto_hist_final": float(dto["history"]["data"][-1]),
               "dto_final_step_logged": dto_logged, "dto_stop_step": dto_stop,
               "b0_profiled_true_d": b0_true_prof}
        # PDE-consistent loss of the PINN / BiLO fits: their d_pred pushed through the
        # DTO forward solve (their grids coincide with the DTO grid to ~1e-16).
        for m in ("pinn", "bilo"):
            d_on = np.interp(x_res.numpy(), fits[m]["x_plot"], fits[m]["d_pred"])
            row[f"loss_fdm_{m}_d"] = dto_data_loss(np.maximum(d_on, D_MIN), x_res, xp, m_obs)[0]
            row[f"{m}_hist_final"] = float(fits[m]["history"]["data"][-1])
            row[f"{m}_final_step_logged"], row[f"{m}_stop_step"] = final_logged(fits[m])
        true_rows.append(row)

        # Forward consistency: FDM solve with the method's own D and b0 on its own grid.
        for m in METHODS:
            f = fits[m]
            x, d, u_pred, b = f["x_plot"], f["d_pred"], f["u_pred"], f["b_pred"]
            u_hat = fdm(d, x, 1.0)                      # unit-source response
            u_fdm = b * u_hat
            b0_ppp = n / (m_obs * np.trapz(u_hat, x))   # varpro.get_b0_ppp for this D
            b_shape = float(u_pred @ u_hat / (u_hat @ u_hat))  # L2-best amplitude
            fc[m]["rel"][s - 1] = rel_l2(u_pred, u_fdm)
            fc[m]["rel_b0_ppp"][s - 1] = rel_l2(u_pred, b0_ppp * u_hat)
            fc[m]["rel_shape"][s - 1] = rel_l2(u_pred, b_shape * u_hat)
            fc[m]["b_pred"][s - 1] = b
            fc[m]["b0_ppp"][s - 1] = b0_ppp
            # BiLO's logged 'u_fdm_err' metric (field_loss default 'rle'):
            # mean(((u_net - u_FDM) / (u_FDM + 1e-6))^2) on x_res, at the logged step.
            fc[m]["u_fdm_err_rle"][s - 1] = float(np.mean(((u_pred - u_fdm) / (u_fdm + 1e-6)) ** 2))
            if "u_fdm_err" in f["history"]:
                fc[m]["hist_u_fdm_err"][s - 1] = float(f["history"]["u_fdm_err"][-1])
        if s % 10 == 0:
            print(f"  seed {s:3d} done", flush=True)
    return true_rows, fc, (particle_maxdiff, particle_methods_identical)


# =============================================================================
# 3. Data snapshots: true intensity, each method's fit to that snapshot (D and the FDM
#    solve of that D with the fit's own b0), and each method's median-D forward solve
# =============================================================================
SWEEP_DIR_B250 = os.path.join(SWEEP_ROOT, f"Fickian_b250_f{FREQ}_baseline")


_RECOVERY = {}


def _recovery(b0):
    if b0 not in _RECOVERY:
        with open(RECOVERY_PKL[b0], "rb") as f:
            _RECOVERY[b0] = pickle.load(f)
    return _RECOVERY[b0]


def snapshot_fit_raw(method, b0, seed):
    """One trial's fit at the main-text settings. Returns (fit dict, source label). At
    b0 = 1000 every method has a per-seed sweep pickle (load_fit). At b0 = 250 DTO has one
    in the b0 = 250 sweep folder (which also stores the particles), and PINN and BiLO come
    from the recovery pickle of Combined_Fickian_slurm.py (trial index i = seed i + 1,
    checked below against the DTO sweep pickle)."""
    if b0 == B0_SWEEP:
        return load_fit(method, seed), os.path.relpath(SWEEP_DIR, HERE)
    if method == "dto":
        with open(os.path.join(SWEEP_DIR_B250, f"{method}_s{seed:03d}.pkl"), "rb") as f:
            return pickle.load(f), os.path.relpath(SWEEP_DIR_B250, HERE)
    rec = _recovery(b0)
    # Seed mapping of the recovery list: its DTO entry must match the DTO sweep pickle (the
    # same fit; bit for bit when both ran on the same machine).
    with open(os.path.join(SWEEP_DIR_B250, f"dto_s{seed:03d}.pkl"), "rb") as f:
        dto_sweep = pickle.load(f)
    dto_rec = rec["dto_results"][seed - 1]
    for k in ("d_pred", "u_pred", "u_true", "x_plot"):
        assert np.allclose(np.asarray(dto_rec[k]).reshape(-1), np.asarray(dto_sweep[k]).reshape(-1),
                           rtol=1e-10, atol=1e-12), k
    assert np.isclose(dto_rec["b_pred"], dto_sweep["b_pred"], rtol=1e-10, atol=0)
    return rec[f"{method}_results"][seed - 1], os.path.basename(RECOVERY_PKL[b0]) + f"[{seed - 1}]"


def snapshot_fits(b0, seed):
    """Per method: the trial's fitted D, its b0, and u_FDM[D, b0] on the fit's grid."""
    fits = {}
    for m in METHODS:
        f, src = snapshot_fit_raw(m, b0, seed)
        assert "seed" not in f or int(f["seed"]) == seed
        x = np.asarray(f["x_plot"]).reshape(-1)
        d = np.asarray(f["d_pred"]).reshape(-1)
        d_true = np.asarray(f["d_true"]).reshape(-1)
        assert np.allclose(np.asarray(f["x_true"]).reshape(-1), x, rtol=0, atol=1e-15)
        b = float(f["b_pred"])
        fits[m] = {"x": x, "d_pred": d, "b_pred": b, "b_true": float(f["b_true"]),
                   "u_fdm": fdm(d, x, b), "d_err": rel_l2(d, d_true), "source": src,
                   "x_particles": (np.asarray(f["x_particles"]).reshape(-1)
                                   if "x_particles" in f else None)}
    return fits, fits["dto"]["x_particles"]


# Example snapshots, picked by eye among the seeds whose fits recover D best (lowest mean
# relative L1/L2 error of D over the three methods), so the example is clean.
SNAPSHOT_SEEDS = {250: (32, 76), 1000: (96, 71)}


def snapshot_seeds(b0):
    return list(SNAPSHOT_SEEDS[b0])


def snapshots():
    out = {}
    for b0 in (250, 1000):
        rec = _recovery(b0)
        for m in METHODS:
            assert len(rec[f"{m}_results"]) == N_SEEDS
        seeds = {}
        for s in snapshot_seeds(b0):
            problem = build_problem(s, b0)
            xg = problem.x_grid.detach().cpu().numpy().reshape(-1)
            u_true = np.asarray(problem.u_true).reshape(-1)
            xp = problem.particles.x_particles.detach().cpu().numpy().reshape(-1)
            maxdiff = None
            if b0 == B0_SWEEP:
                # Use the stored particles; a rebuild elsewhere differs only by round-off.
                xp_sweep = load_fit("dto", s)["x_particles"]
                assert xp_sweep.size == xp.size
                maxdiff = float(np.max(np.abs(xp_sweep - xp)))
                xp = xp_sweep
            # The recovery pickles do not store particles; check the true u instead
            # (round-off level).
            u_rec = np.asarray(rec["dto_results"][s - 1]["u_true"]).reshape(-1)
            fits, xp_fit = snapshot_fits(b0, s)
            if b0 != B0_SWEEP:
                # The b0 = 250 sweep fits store their particles too: show those.
                assert xp_fit.size == xp.size
                maxdiff = float(np.max(np.abs(xp_fit - xp)))
                xp = xp_fit
            for m, f in fits.items():
                # Each stored particle set is exactly the one shown (same seed, same data).
                if f["x_particles"] is not None:
                    assert np.array_equal(f["x_particles"], xp), (b0, s, m)
                assert np.allclose(f["x"], xg, rtol=0, atol=1e-15), (b0, s, m)
                f["x_particles_match"] = None if f["x_particles"] is None else True
                del f["x_particles"]
            seeds[s] = {"x_particles": xp, "N": int(xp.size), "m_obs": int(problem.particles.m_obs),
                        "particles_maxdiff_vs_sweep": maxdiff,
                        "u_true_reldiff_vs_recovery": rel_l2(u_true, u_rec),
                        "fits": fits}
        out[b0] = {"x_grid": xg, "u_true": u_true, "d_true": np.asarray(problem.d_true).reshape(-1),
                   "int_u_true": float(np.trapz(u_true, xg)), "expected_count": b0 / MU,
                   "seeds": seeds, "seed_order": list(seeds)}
    return out


def pct(a):
    a = np.asarray(a, dtype=float)
    return {"median": float(np.nanmedian(a)), "p10": float(np.nanpercentile(a, 10)),
            "p90": float(np.nanpercentile(a, 90))}


def main():
    torch.set_num_threads(1)
    t0 = time.time()
    print(f"[derived] per-seed losses and forward solves ({N_SEEDS} seeds)", flush=True)
    rows, fc, (pdiff, psame) = per_seed()
    print("[derived] snapshots", flush=True)
    snaps = snapshots()

    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    result = {
        "meta": {
            "condition": f"Fickian alpha=1, Neumann, mu={MU}, z={Z}, m_obs=1, "
                         f"D=0.1+0.05 sin({2 * FREQ}pi x), b0={B0_SWEEP} (sweep baseline)",
            "sweep_dir": os.path.relpath(SWEEP_DIR, HERE),
            "recovery_pkls": {k: os.path.basename(v) for k, v in RECOVERY_PKL.items()},
            "loss_definition": "varpro.ppp_nll / m_obs: n*log(int u_hat) - sum_i log u_hat(x_i), "
                               "b0 profiled as n/(m_obs int u_hat); for Fickian+Neumann int u_hat = 1/mu, "
                               "so b0* = mu*n. Evaluated on the DTO 201-point uniform grid with the "
                               "DTO FDM solve and linear interpolation at particles.",
            "same_loss_definition": {"pinn": True, "bilo": True},
            "same_loss_definition_note": "Same formula and normalisation (varpro.get_b0_ppp + "
                                         "varpro.ppp_nll), but PINN/BiLO evaluate u_hat with the "
                                         "network (integral by trapezoid on n_int=1001 points, u_hat "
                                         "at particles from the net), not the FDM solve. Their logged "
                                         "values are therefore not constrained to a PDE solution; "
                                         "loss_fdm_{pinn,bilo}_d gives the PDE-consistent value.",
            "runtime_s": None,
        },
        "true_d_loss": {"rows": rows, **cols},
        "forward_consistency": {**fc,
                                "seeds": np.arange(1, N_SEEDS + 1),
                                "notes": {"rel": "||u_pred - b_pred*u_hat_FDM[d_pred]|| / ||.||_FDM on x_plot",
                                          "rel_b0_ppp": "b0 re-profiled as n/(m_obs int u_hat_FDM) (PPP-optimal)",
                                          "rel_shape": "b0 = L2-best amplitude (shape mismatch only)",
                                          "u_fdm_err_rle": "BiLO's u_fdm_err metric recomputed at the "
                                                           "returned fit; hist_u_fdm_err is the last logged value"}},
        "snapshots": snaps,
        "particle_check": {"maxdiff_regenerated_vs_stored": pdiff,
                           "stored_identical_across_methods": psame},
    }
    result["meta"]["runtime_s"] = time.time() - t0

    # Short console summary.
    d = (cols["dto_hist_final"] - cols["loss_true_d"]) / cols["n_particles"]
    print(f"  particles: stored identical across methods {psame.all()}, "
          f"max |regenerated - stored| = {pdiff.max():.2e}")
    print(f"  DTO loss check |re-eval - hist|: max {np.max(np.abs(cols['loss_dto_final_check'] - cols['dto_hist_final'])):.3e}"
          f"  (final step logged in {cols['dto_final_step_logged'].sum()}/{N_SEEDS})")
    print(f"  (dto_hist_final - loss_true_d)/N: {pct(d)}")
    for m in ("pinn", "bilo"):
        print(f"  ({m} FDM loss - loss_true_d)/N: {pct((cols[f'loss_fdm_{m}_d'] - cols['loss_true_d']) / cols['n_particles'])}")
        print(f"  ({m} hist final - loss_true_d)/N: {pct((cols[f'{m}_hist_final'] - cols['loss_true_d']) / cols['n_particles'])}")
    for m in METHODS:
        print(f"  fwd {m}: rel {pct(fc[m]['rel'])} | rel_b0_ppp {pct(fc[m]['rel_b0_ppp'])} | rel_shape {pct(fc[m]['rel_shape'])}")
    for m in METHODS:
        st = [x for x in cols[f"{m}_stop_step"] if x is not None]
        print(f"  {m}: early-stopped {len(st)}/{N_SEEDS}, stop steps {np.unique(st).tolist()}, "
              f"final step logged {int(np.sum(cols[f'{m}_final_step_logged']))}/{N_SEEDS}")
    print(f"  BiLO u_fdm_err recomputed vs logged: {pct(fc['bilo']['u_fdm_err_rle'] / fc['bilo']['hist_u_fdm_err'])} (ratio)")
    for b0, sn in snaps.items():
        print(f"  snapshot b0={b0}: int u = {sn['int_u_true']:.4f} (b0/mu = {sn['expected_count']}), "
              f"seeds {sn['seed_order']}, N = {[sn['seeds'][s]['N'] for s in sn['seed_order']]}, particle diff vs sweep = "
              f"{[sn['seeds'][s]['particles_maxdiff_vs_sweep'] for s in sn['seed_order']]}, "
              f"u_true rel diff vs recovery pickle: "
              f"{[sn['seeds'][s]['u_true_reldiff_vs_recovery'] for s in sn['seed_order']]}")
        for s in sn["seed_order"]:
            fs = sn["seeds"][s]["fits"]
            print(f"    seed {s}: " + " | ".join(
                f"{m} relL2(D) {fs[m]['d_err']:.3f} b0/b_true {fs[m]['b_pred'] / fs[m]['b_true']:.3f} "
                f"[{fs[m]['source']}]" for m in METHODS))

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(result, f)
    os.replace(tmp, OUT_PATH)
    print(f"[derived] done in {result['meta']['runtime_s']:.0f} s -> {OUT_PATH}")


if __name__ == "__main__":
    main()
