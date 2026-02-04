#!/usr/bin/env python3
import subprocess
import re
import time
import os
import multiprocessing
import sys
import argparse
import yaml
from typing import List, Dict, Any

# Import process_yaml from parseyaml
from parseyaml import process_yaml
import utilgpu

# ==========================================
# 2. EXECUTION ENGINE
# ==========================================

def run_experiment(gpu_id: int, command_args: List[str], log_name: str):
    """Worker function to run a single experiment on a specific GPU."""
    env = os.environ.copy()
    # env["CUDA_VISIBLE_DEVICES"] = str(gpu_id) # Removed as per requirement
    
    print(f"[{time.strftime('%H:%M:%S')}] Starting {log_name} on GPU cuda:{gpu_id}...")
    
    # Use run.device to set the specific GPU
    full_cmd = ["python", "run.py"] + command_args + ["run.device", f"cuda:{gpu_id}"]
    
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
    parser.add_argument("-e", "--exclude_gpu", type=str, default=None, help="Comma-separated list of nvidia-smi GPU IDs to exclude (e.g., 1,2)")
    parser.add_argument("-f", "--filter", type=str, default=None, help="Regex to filter run_name (e.g., .*pinn)")
    
    args = parser.parse_args()
    
    # 1. Determine Available GPUs
    # Get all available GPUs from nvidia-smi
    try:
        all_smi_ids = utilgpu.list_available_gpus()
    except Exception as e:
        print(f"Error listing GPUs: {e}")
        sys.exit(1)

    print(f"Detected GPUs (nvidia-smi IDs): {all_smi_ids}")

    # Parse excluded GPUs
    exclude_ids = []
    if args.exclude_gpu:
        try:
            exclude_ids = [int(x) for x in args.exclude_gpu.split(",")]
        except ValueError:
            print("Error: --exclude_gpu must be a comma-separated list of integers.")
            sys.exit(1)
    
    # Filter candidates
    candidate_smi_ids = [gid for gid in all_smi_ids if gid not in exclude_ids]
    
    if not candidate_smi_ids:
        print("No GPUs available after exclusion.")
        sys.exit(1)

    # Map to Torch IDs
    try:
        nv_to_torch = utilgpu.get_nv_to_torch_map()
    except Exception as e:
        print(f"Error getting GPU mapping: {e}")
        sys.exit(1)
        
    valid_torch_ids = []
    for smi_id in candidate_smi_ids:
        if smi_id in nv_to_torch:
            valid_torch_ids.append(nv_to_torch[smi_id])
        else:
            print(f"Warning: GPU {smi_id} (nvidia-smi) not found in PyTorch devices. Skipping.")
            
    if not valid_torch_ids:
        print("No valid PyTorch GPUs found.")
        sys.exit(1)
        
    print(f"Using GPUs (torch IDs): {valid_torch_ids}")
    
    # 2. Parse YAML and generate tasks
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
        cmd_args.extend(["run.name", run_name])
        
        tasks.append((run_name, cmd_args))

    # Apply regex filter if provided
    if args.filter:
        try:
            pattern = re.compile(args.filter)
        except re.error as e:
            print(f"Error: invalid regex for --filter: {e}")
            sys.exit(1)
        tasks = [(rn, ca) for rn, ca in tasks if pattern.search(rn)]
        print(f"Filtered to {len(tasks)} experiments matching '{args.filter}'.")

    print(f"Generated {len(tasks)} experiments from {args.yaml_file}.")
    
    if not tasks:
        print("No tasks found. Exiting.")
        return

    # 3. Worker Queue System
    manager = multiprocessing.Manager()
    gpu_queue = manager.Queue()
    
    # Populate queue with torch IDs (for rotation)
    for torch_id in valid_torch_ids:
        gpu_queue.put(torch_id)
    
    # Prepare tasks with the shared queue
    tasks_with_queue = [(run_name, cmd_args, gpu_queue) for run_name, cmd_args in tasks]

    # 4. Run in parallel
    # The number of parallel processes is limited by the number of GPUs
    num_workers = len(valid_torch_ids)
    print(f"Starting {num_workers} workers...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(worker_with_gpu_queue, tasks_with_queue)

if __name__ == "__main__":
    main()