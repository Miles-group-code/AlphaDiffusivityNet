#!/usr/bin/env python3
"""
Command-line entry point for AlphaDiffusivityNet simulations.

Usage:
    # Key-value pairs (flat or nested dot-notation)
    python run.py method dto mode field wandb.enabled true

    # Traditional flags also work (keys stripped of --)
    python run.py --method pinn --alpha 1.0 --train.lr_d 1e-3
"""

import sys
import os
from typing import Any, List, Dict

import torch
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from interface import Problem, solve

try:
    import wandb
except ImportError:
    wandb = None


def parse_value(v: str) -> Any:
    """Parse string value to int/float/bool/str."""
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() == "none":
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def map_shortcut(key: str) -> str:
    """Map common shortcuts to full config paths."""
    shortcuts = {
        # Solver
        "method": "solver.method",
        
        # Physics
        "alpha": "physics.alpha",
        "mu": "physics.mu",
        "bc_type": "physics.bc_type",
        "b_true": "physics.b_true",
        "source_loc": "physics.sources", # special handling needed, but usually 1 source
        
        # Data
        "mode": "data.mode",
        "m_obs": "data.m_obs",
        "field_loss": "data.field_loss",
        
        # D Profile
        "d_profile": "d_profile.profile_type",
        "d_params": "d_profile.params",

        # Run
        "device": "run.device",
        "seed": "run.seed",
        "outdir": "run.outdir",
        
        # WandB
        "wandb": "wandb.enabled",
        "project": "wandb.project",
        "entity": "wandb.entity",
        "group": "wandb.group",
        "name": "wandb.name",
        "tags": "wandb.tags",
        
        # Common training
        "lr_d": "train.lr_d_fine", # usually what people mean
        "max_iters": "train.finetune_iters",
    }
    return shortcuts.get(key, key)


def update_config_value(cfg: Config, key: str, value: Any) -> None:
    """Update config object with a dot-notation key."""
    # Handle special case: source_loc -> tuple
    if key == "physics.sources" and not isinstance(value, tuple):
        value = (float(value),)
    
    # Handle d_params -> tuple
    if key == "d_profile.params":
        if isinstance(value, str):
             # Remove brackets if present
             value = value.strip("[]()")
             value = tuple(float(x.strip()) for x in value.split(","))
        elif isinstance(value, (list, tuple)):
             value = tuple(float(x) for x in value)

    # Handle tags -> tuple
    if key == "wandb.tags":
        # If passed as string "tag1 tag2", split it. 
        # But our parser splits by space, so we might get individual tags if passed as separate args?
        # The parser expects key-value pairs. 
        # So user should pass tags "tag1,tag2" and we parse here, OR we assume simple values.
        # Let's support comma-separated string
        if isinstance(value, str):
            value = tuple(t.strip() for t in value.split(","))

    parts = key.split(".")
    obj = cfg
    
    # Traverse to leaf
    for i, part in enumerate(parts[:-1]):
        if not hasattr(obj, part):
             print(f"Warning: Unknown config section/field '{part}' in '{key}'")
             return
        obj = getattr(obj, part)
        
    leaf = parts[-1]
    if hasattr(obj, leaf):
        # Type casting if possible (to match existing type)
        current_val = getattr(obj, leaf)
        if current_val is not None and not isinstance(value, type(current_val)) and value is not None:
             # Basic casting for safety if types differ but value is compatible
             # (parse_value already does int/float/bool inference)
             pass 
             
        setattr(obj, leaf, value)
        print(f"Config: {key} = {value}")
    else:
        print(f"Warning: Unknown config field '{leaf}' in '{key}'")


def main():
    # 1. Setup Config
    cfg = Config()
    
    # 2. Parse arguments manually
    raw_args = sys.argv[1:]
    
    if len(raw_args) == 0:
        print("No arguments provided. Using defaults.")
        print("Usage: python run.py key value [key value ...]")
        
    i = 0
    while i < len(raw_args):
        key = raw_args[i]
        
        # Handle --flag (boolean true) if no value follows or next arg is a key
        # But user said "arg1 val1 arg2 val2", implying strict pairs.
        # However, standard CLI flags often don't take values.
        # Let's stick to key-value pairs for simplicity and robustness as requested.
        
        # Strip leading dashes
        while key.startswith("-"):
            key = key[1:]
            
        if i + 1 >= len(raw_args):
            # Trailing key without value?
            # Could be a boolean flag like "wandb" meaning "wandb true"
            # Let's assume boolean true if it maps to a boolean field, otherwise warn.
            print(f"Warning: Missing value for argument '{key}', assuming True.")
            val = True
            i += 1 
        else:
            val_str = raw_args[i+1]
            
            # Heuristic: if val_str looks like a key (starts with --), maybe current key is a bool flag?
            if val_str.startswith("--"):
                 print(f"Warning: Value '{val_str}' looks like a key. Assuming '{key}' is boolean True.")
                 val = True
                 # Don't consume the next arg
                 i += 1
            else:
                val = parse_value(val_str)
                i += 2
        
        full_key = map_shortcut(key)
        update_config_value(cfg, full_key, val)

    # 3. Validate
    try:
        cfg.validate()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

    # 4. Setup WandB
    if cfg.wandb.enabled:
        if wandb is None:
            print("Error: wandb requested but not installed/imported.")
            sys.exit(1)
            
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            group=cfg.wandb.group,
            entity=cfg.wandb.entity,
            tags=cfg.wandb.tags,
            config=cfg.to_dict(),
            settings=wandb.Settings(_disable_stats=True),
        )
        print(f"[WandB] Run initialized: {wandb.run.name}")
    
    # 5. Create Problem
    print(f"\n--- Setting up Problem ({cfg.data.mode}) ---")
    
    problem = Problem.synthetic(
        alpha=cfg.physics.alpha,
        mode=cfg.data.mode,
        d_profile=cfg.d_profile.profile_type,
        d_profile_params=cfg.d_profile.params,
        mu=cfg.physics.mu,
        b_true=cfg.physics.b_true,
        source_location=cfg.physics.sources[0],
        n_obs=cfg.grid.n_res,
        m_obs=cfg.data.m_obs,
        bc_type=cfg.physics.bc_type,
        seed=cfg.run.seed,
        device=cfg.run.device,
        verbose=True
    )
    
    # 6. Run Solver
    print(f"\n--- Running {cfg.solver.method.upper()} ---")
    solution = solve(
        problem,
        method=cfg.solver.method,
        config=cfg,
        verbose=True
    )
    
    # 7. Metrics & Visualization
    metrics = solution.metrics(problem)
    print("\n--- Final Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4e}")
        
    if cfg.wandb.enabled and wandb.run is not None:
        # Log metrics as summary only (not as charts)
        for k, v in metrics.items():
            wandb.run.summary[k] = v
        
        print("[WandB] Generating plots...")
        fig = solution.plot(problem, show=False)
        wandb.log({"solution_plot": wandb.Image(fig)})
        plt.close(fig)
        
        try:
            from diagnostics import plot_d_evolution_color, plot_bilo_d_variation

            
            x_res_np = solution._get_x_array()
            fig_evo = plot_d_evolution_color(
                cfg.solver.method.upper(),
                solution.history,
                x_res_np,
                outdir=None,
                show=False
            )
            if fig_evo:
                wandb.log({"d_evolution": wandb.Image(fig_evo)})
                plt.close(fig_evo)
            
            # For BiLO, also plot D variation sensitivity
            if cfg.solver.method.lower() == "bilo":
                fig_var = plot_bilo_d_variation(
                    solution,
                    problem,
                    outdir=None,
                    show=False
                )
                if fig_var:
                    wandb.log({"bilo_d_variation": wandb.Image(fig_var)})
                    plt.close(fig_var)
                
        except Exception as e:
            print(f"[WandB] Warning: Failed to generate auxiliary plots: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()
