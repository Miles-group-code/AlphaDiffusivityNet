import numpy as np
import torch
import multiprocessing as mp
import matplotlib.pyplot as plt
from functools import partial
from scipy.stats import gaussian_kde
# Make the framework at the repo root importable when run from python_paper_figures/
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from interface import Problem, solve, show_settings
import diagnostics

def run_single_trial_internal(args, b0=600, method="pinn", device="cpu"):
    """
    Combined worker function that defines the problem and solves it.
    """
    seed, dto_config = args
    
    # Ensure reproducibility within the process
    np.random.seed(seed)
    torch.manual_seed(seed)
    dtype = torch.float32 if device == "mps" else torch.float64

    # 1. Setup Configuration
    custom_config = Config()
    custom_config.arch.d_net_arch = "mlp"
    custom_config.arch.d_net_width = 64
    custom_config.arch.d_net_depth = 3
    custom_config.arch.u_net_arch = "mmlp"
    custom_config.arch.u_net_width = 256
    custom_config.arch.rff_scale = 250.0 
    #custom_config.reg.wreg_d_neumann = 1.0  
    custom_config.train.lower_tol = 0.01

    # 2. Generate Problem
    problem = Problem.synthetic(
        alpha=1.0,
        mode="particles",
        d_profile="sinusoidal",
        d_profile_params=(0.1, 0.05, 3.0),
        mu=5.0,
        source_location=0.5,
        b_true=b0,
        m_obs=1, 
        use_pde_sampling=True,
        bc_type="neumann",
        seed=seed,
        device=device,
        dtype=dtype,
    )

    # 3. Solve
    sol = solve(problem, method=method, config=custom_config, **dto_config)
    
    # 4. Return results as NumPy for plotting
    return {
        "u_pred": sol.u_pred.detach().cpu().numpy().reshape(-1),
        "d_pred": sol.d_pred.detach().cpu().numpy().reshape(-1),
        "x_plot": sol.x_res.detach().cpu().numpy().reshape(-1),
        "u_true": problem.u_true, 
        "d_true": problem.d_true,
        "x_true": problem.x_grid.detach().cpu().numpy(),
        "b_pred": sol.b0_star,   
        "b_true": problem.b_true, 
    }

def main():
    # --- CONFIGURATION ---
    n_trials = 50
    
    SHARED_CONFIG = {
        "max_iters": 7000,
        #"field_loss": "mse",
        "smoothness_type": "h1",
        "pert_scale": 0.0,#0.9  0.7
        "pert_freq": 2,#0.000001  10
        "scalar_fit_iters": 500,
        "log_every": 10,
        "use_scheduler": True,
        "use_ddi": True, 
        "early_burnin": 5000,
        "early_patience": 100,
        "early_tol": 1e-4,
        "n_res": 201,
    }
    DTO_PARTICLE_CONFIG = {
        "pretrain_iters": 2000,
        #"lr_d_pre": 0.001,
        #"lr_lower_pre": 0.001,
        "wreg_smooth": 1, #10
        "wreg_scale": 1e-1, #20
        "lr_d_fine": 1e-4,
        "lr_lower_fine": 1e-6,
        "w_bc": 1.0,
    }   
    dto_config = {**SHARED_CONFIG, **DTO_PARTICLE_CONFIG}

    PINN_PARTICLE_CONFIG = {
        "pretrain_iters": 2000,
        #"lr_d_pre": 0.001,
        #"lr_lower_pre": 0.001,
        "wreg_smooth": 2, #10
        "wreg_scale": 1, #20
        "w_phys": 350,
        "lr_d_fine": 1e-2,
        "lr_lower_fine": 1e-4,
        "w_jump": 10,
        "w_bc": 1.0,
    }   
    pinn_config = {**SHARED_CONFIG, **PINN_PARTICLE_CONFIG}

    BILO_PARTICLE_CONFIG = {
        "pretrain_iters": 2000,
        "lr_d_pre": 0.001,
        "lr_lower_pre": 0.001,
        "wreg_smooth": 1,#1 
        "wreg_scale": 1,#10
        "lr_d_fine": 1e-4,#5e-4,
        "lr_lower_fine": 1e-3,#5e-3,
        "w_jump": 10,#1
        #"w_data": 2,
        #"w_phys": 350,
        "w_resgrad": 0.05,
    }   
    bilo_config = {**SHARED_CONFIG, **BILO_PARTICLE_CONFIG}
    # --- MULTIPROCESSING EXECUTION ---
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running {n_trials} Trials on {device} ---")
    
    tasks_dto = [(seed, dto_config) for seed in range(1, n_trials+1)]
    tasks_pinn = [(seed, pinn_config) for seed in range(1, n_trials+1)]
    tasks_bilo = [(seed, bilo_config) for seed in range(1, n_trials+1)]

    dto_worker = partial(run_single_trial_internal, b0=250, method="dto", device=device)
    pinn_worker = partial(run_single_trial_internal, b0=250, method="pinn", device=device)
    bilo_worker = partial(run_single_trial_internal, b0=250, method="bilo", device=device)

    num_processes = min(mp.cpu_count() - 1, 4) 
    
    with mp.Pool(processes=num_processes) as pool:
        dto_results = pool.map(dto_worker, tasks_dto)
        pinn_results = pool.map(pinn_worker, tasks_pinn)
        bilo_results = pool.map(bilo_worker, tasks_bilo)
    
    # --- FILTERING OUTLIERS ---
    def filter_res(res, thresh_mult=10):
        b_true = res[0]['b_true']
        return [r for r in res if r['b_pred'] <= thresh_mult * b_true]

    filtered_dto = filter_res(dto_results, 10)
    filtered_pinn = filter_res(pinn_results, 3) # Stricter filter for PINN
    filtered_bilo = filter_res(bilo_results, 10)

    # --- DATA EXTRACTION ---
    dto_all_d_preds = np.array([r['d_pred'] for r in filtered_dto])
    dto_all_b_preds = np.array([r['b_pred'] for r in filtered_dto])
    pinn_all_d_preds = np.array([r['d_pred'] for r in filtered_pinn])
    pinn_all_b_preds = np.array([r['b_pred'] for r in filtered_pinn])
    bilo_all_d_preds = np.array([r['d_pred'] for r in filtered_bilo])
    bilo_all_b_preds = np.array([r['b_pred'] for r in filtered_bilo])

    dto_x_plot, dto_x_true, dto_d_true = dto_results[0]['x_plot'], dto_results[0]['x_true'], dto_results[0]['d_true']
    pinn_x_plot = pinn_results[0]['x_plot']
    bilo_x_plot = bilo_results[0]['x_plot']

    # --- PLOTTING SETUP ---
    def get_stats(data):
        return {'median': np.median(data, axis=0), 'p10': np.percentile(data, 10, axis=0), 'p90': np.percentile(data, 90, axis=0)}

    # Color-blind friendly palette (Okabe-Ito)
    colors = {'DTO': '#E69F00', 'PINN': '#56B4E9', 'BiLO': '#009E73'}
    styles = {'DTO': '-', 'PINN': '--', 'BiLO': '-.'}

    fig, ax = plt.subplots(1, 2, figsize=(22, 7))

    # Subplot 0: Recovery of D(x)
    ax[0].plot(dto_x_true, dto_d_true, color='black', ls=':', lw=3, label="True $D(x)$")
    
    for label, stats_data, x_coords in [('DTO', dto_all_d_preds, dto_x_plot), 
                                        ('PINN', pinn_all_d_preds, pinn_x_plot), 
                                        ('BiLO', bilo_all_d_preds, bilo_x_plot)]:
        s = get_stats(stats_data)
        ax[0].plot(x_coords, s['median'], color=colors[label], ls=styles[label], lw=3, label=f"{label} Median")
        ax[0].fill_between(x_coords, np.maximum(s['p10'], 0), np.maximum(s['p90'], 0), color=colors[label], alpha=0.15)

    ax[0].set_title("Recovery of $D(x)$ Comparison (Low Birth Value)", fontsize=14)
    ax[0].set_xlabel("Spatial coordinate $x$")
    ax[0].set_ylabel("Diffusion Coefficient $D(x)$")
    ax[0].text(-0.05, 1.05, "(c)", transform=ax[0].transAxes, fontsize=16, fontweight='bold')
    ax[0].grid(True, alpha=0.2)
    ax[0].legend(fontsize=12)

    # Subplot 1: Parameter b Distribution
    b_data_list = [(dto_all_b_preds, 'DTO'), (pinn_all_b_preds, 'PINN'), (bilo_all_b_preds, 'BiLO')]
    b_true = dto_results[0]['b_true']
    ax[1].axvline(b_true, color='black', linestyle=':', lw=2, label=f'True $b$: {b_true}', zorder=5)

    for data, label in b_data_list:
        color = colors[label]
        median, std = np.median(data), np.std(data)
    
        # 1. Histogram Bars (with white edges for clarity)
        counts, bins, _ = ax[1].hist(data, bins=15, color=color, alpha=0.5, edgecolor='white', linewidth=1, rwidth=0.85, label=f'{label} Dist')
    
        # 2. Bell Curve (KDE) - Scaled to Frequency
        kde = gaussian_kde(data)
        x_range = np.linspace(min(data) - 50, max(data) + 50, 500)
        bin_width = bins[1] - bins[0]
        ax[1].plot(x_range, kde(x_range) * len(data) * bin_width, color=color, lw=2.5, alpha=0.9)

        # 3. Median line with unique style
        ax[1].axvline(median, color=color, linestyle=styles[label], lw=3, label=f'{label} Median: {median:.2f} (±{std:.2f})')

    ax[1].set_title("Parameter $b$ Distribution Comparison (Low Birth Value)", fontsize=14)
    ax[1].set_xlabel("Value of $b$")
    ax[1].set_ylabel("Frequency")
    ax[1].grid(True, linestyle='-', alpha=0.3)
    ax[1].text(-0.05, 1.05, "(d)", transform=ax[1].transAxes, fontsize=16, fontweight='bold') # Label (b)
    ax[1].legend(loc='upper right',fontsize=12)

    plt.tight_layout()
    plt.savefig("CCombined_Fickian_b250_t50_3.pdf", dpi=300)
    print("Results saved to CCombined_Fickian_b250_t50_3.pdf")
    plt.show()

if __name__ == '__main__':
    main()
