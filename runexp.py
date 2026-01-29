#!/usr/bin/env python3
import subprocess
import time
import os
import multiprocessing
import sys
import argparse
import yaml
from typing import List, Dict, Any

# Import process_yaml from parseyaml
from parseyaml import process_yaml

# ==========================================
# 1. AVAILABLE RESOURCES
# ==========================================
# Default available GPUs (can be overridden by --gpus)
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

# ==========================================
# 2. EXECUTION ENGINE
# ==========================================

def run_experiment(gpu_id: int, command_args: List[str], log_name: str):
    """Worker function to run a single experiment on a specific GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"[{time.strftime('%H:%M:%S')}] Starting {log_name} on GPU {gpu_id}...")
    
    # We override the 'device' arg to be generic 'cuda' since we control visibility via env var
    full_cmd = ["python", "run.py"] + command_args + ["run.device", "cuda:0"]
    
    try:
        # Run the command and capture output
        result = subprocess.run(
            full_cmd, 
            env=env, 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            print(f"[{time.strftime('%H:%M:%S')}] ✅ Finished {log_name}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] ❌ Failed {log_name}")
            print(f"Error output:\n{result.stderr[-500:]}") # Print last 500 chars of error
            
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] 💥 Exception in {log_name}: {e}")

def worker_with_gpu_queue(task_data):
    """Module-level worker function that can be pickled.
    
    Args:
        task_data: Tuple of (run_name, cmd_args, gpu_queue)
    """
    run_name, cmd_args, gpu_queue = task_data
    
    # Get a GPU from the queue
    gpu_id = gpu_queue.get()
    try:
        run_experiment(gpu_id, cmd_args, run_name)
    finally:
        # Return GPU to pool
        gpu_queue.put(gpu_id)

def main():
    parser = argparse.ArgumentParser(description="Run sweep from YAML configuration")
    parser.add_argument("yaml_file", help="Path to the YAML configuration file")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated list of GPU IDs (e.g., 0,1,2)")
    
    args = parser.parse_args()
    
    # Update available GPUs if provided
    global AVAILABLE_GPUS
    if args.gpus:
        AVAILABLE_GPUS = [int(g) for g in args.gpus.split(",")]
        
    print(f"Using GPUs: {AVAILABLE_GPUS}")
    
    # 1. Parse YAML and generate tasks
    try:
        with open(args.yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Flatten the YAML using parseyaml
        flat_config = process_yaml(data)
    except Exception as e:
        print(f"Error reading/parsing YAML: {e}")
        sys.exit(1)
        
    tasks = []
    for run_name, config_str in flat_config.items():
        # Ignore keys starting with underscore (convention for templates/anchors)
        if run_name.startswith("_") or run_name in ["logging", "common"]:
            continue

        # Split config string into args
        cmd_args = config_str.split()
        
        # Patch "group" -> "run.group" to avoid ambiguity in run.py
        cmd_args = ["run.group" if arg == "group" else arg for arg in cmd_args]
        
        # Add run.name to arguments if not already present (derived from YAML key)
        # We append it so it overrides any previous definition if necessary, 
        # or just sets it if missing.
        cmd_args.extend(["run.name", run_name])
        
        tasks.append((run_name, cmd_args))

    print(f"Generated {len(tasks)} experiments from {args.yaml_file}.")
    
    if not tasks:
        print("No tasks found. Exiting.")
        return

    # 2. Worker Queue System
    manager = multiprocessing.Manager()
    gpu_queue = manager.Queue()
    for gpu in AVAILABLE_GPUS:
        gpu_queue.put(gpu)
    
    # Prepare tasks with the shared queue
    tasks_with_queue = [(run_name, cmd_args, gpu_queue) for run_name, cmd_args in tasks]

    # 3. Run in parallel
    # The number of parallel processes is limited by the number of GPUs
    with multiprocessing.Pool(processes=len(AVAILABLE_GPUS)) as pool:
        pool.map(worker_with_gpu_queue, tasks_with_queue)

if __name__ == "__main__":
    main()
