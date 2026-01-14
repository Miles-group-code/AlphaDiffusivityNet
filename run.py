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
import dataclasses
from typing import Any, List, Dict, Set

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


def build_key_map(cfg_obj) -> Dict[str, List[str]]:
    """
    Recursively find all leaf keys in the config object.
    Returns a dict mapping leaf_key -> list of full dot-paths.
    """
    mapping = {}
    
    def traverse(prefix, obj):
        if dataclasses.is_dataclass(obj):
            for field in dataclasses.fields(obj):
                name = field.name
                val = getattr(obj, name)
                full_path = f"{prefix}.{name}" if prefix else name
                
                # Check if val is a nested config (dataclass instance)
                if dataclasses.is_dataclass(val):
                    traverse(full_path, val)
                else:
                    # Leaf field
                    if name not in mapping:
                        mapping[name] = []
                    mapping[name].append(full_path)
    
    traverse("", cfg_obj)
    return mapping


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
        setattr(obj, leaf, value)
        print(f"Config: {key} = {value}")
    else:
        print(f"Warning: Unknown config field '{leaf}' in '{key}'")


def main():
    # 1. Setup Config
    cfg = Config()
    
    # Build map of unique names to full paths
    key_map = build_key_map(cfg)
    
    # Collect all valid full paths to allow explicit dot-notation
    valid_full_paths = set()
    for paths in key_map.values():
        for p in paths:
            valid_full_paths.add(p)
    
    # 2. Parse arguments manually
    raw_args = sys.argv[1:]
    
    if len(raw_args) == 0:
        print("No arguments provided. Using defaults.")
        print("Usage: python run.py key value [key value ...]")
        
    i = 0
    while i < len(raw_args):
        key = raw_args[i]
        
        # Determine value (assume strict pairs: key value)
        if i + 1 < len(raw_args):
            val_str = raw_args[i+1]
            i += 2
        else:
            # Trailing key implies boolean True flag
            val_str = "true"
            i += 1
            
        val = parse_value(val_str)
        
        # Resolve key
        full_key = None
        
        if key in valid_full_paths:
            # Explicit full path (e.g. "solver.method")
            full_key = key
        elif key in key_map:
            # Shortcut leaf name (e.g. "method")
            paths = key_map[key]
            if len(paths) == 1:
                full_key = paths[0]
            else:
                print(f"Error: Ambiguous key '{key}'. Matches parent configs: {paths}")
                print("Please specify the parent config (e.g. 'parent.key').")
                sys.exit(1)
        else:
            print(f"Error: Unknown configuration key '{key}'")
            sys.exit(1)
        
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
    
    # Determine output directory for saving plots to disk
    # Skip local saving when wandb is enabled (user preference)
    outdir = cfg.run.outdir if not cfg.wandb.enabled else None
    
    if cfg.wandb.enabled and wandb.run is not None:
        # Log metrics as summary only (not as charts)
        for k, v in metrics.items():
            wandb.run.summary[k] = v
    
    print("[Generating plots...]")
    try:
        fig = solution.plot(problem, show=False)
        if fig is not None:
            if cfg.wandb.enabled and wandb.run is not None:
                wandb.log({"solution_plot": wandb.Image(fig)}, commit=False)
                print(f"  ✓ Logged solution_plot to wandb (run: {wandb.run.name})")
            if outdir:
                os.makedirs(outdir, exist_ok=True)
                save_path = os.path.join(outdir, "solution_plot.png")
                fig.savefig(save_path, dpi=150)
                print(f"  ✓ Saved solution_plot to: {save_path}")
            plt.close(fig)
        else:
            print("Warning: solution.plot() returned None")
    except Exception as e:
        print(f"Warning: Failed to generate solution plot: {e}")
    
    # Generate D evolution plot
    try:
        from diagnostics import plot_d_evolution_color
        x_res_np = solution._get_x_array()
        # When wandb is enabled, we need the figure returned (not saved), so pass outdir=None
        # and handle saving separately
        fig_evo = plot_d_evolution_color(
            cfg.solver.method.upper(),
            solution.history,
            x_res_np,
            outdir=None,  # Don't save yet, we'll handle it below
            show=False
        )
        if fig_evo:
            if cfg.wandb.enabled and wandb.run is not None:
                wandb.log({"d_evolution": wandb.Image(fig_evo)}, commit=False)
                print(f"  ✓ Logged d_evolution to wandb (run: {wandb.run.name})")
            if outdir:
                os.makedirs(outdir, exist_ok=True)
                save_path = os.path.join(outdir, f"{cfg.solver.method.lower()}_d_evolution.png")
                fig_evo.savefig(save_path, dpi=150)
                print(f"  ✓ Saved d_evolution to: {save_path}")
            plt.close(fig_evo)
        else:
            print("  ⚠ D evolution plot not generated (insufficient snapshots)")
    except Exception as e:
        print(f"Warning: Failed to generate D evolution plot: {e}")
    
    # Commit all plots to wandb at once
    if cfg.wandb.enabled and wandb.run is not None:
        wandb.log({}, commit=True)
        print(f"  ✓ All plots committed to wandb")
        wandb.finish()

    print("\nDone.")

if __name__ == "__main__":
    main()
