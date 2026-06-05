#!/bin/bash
#SBATCH -p standard               ## standard partition
#SBATCH -N 1                      ## single node
#SBATCH --ntasks=1                ## ONE task -- Python multiprocessing spawns the workers itself
#SBATCH --cpus-per-task=40        ## CPUs for the trial-parallel pool (bump to node size if you like)
#SBATCH -A your_slurm_account      ## set to your SLURM allocation/account
## optional job email (set your address and uncomment):
##SBATCH --mail-type=ALL
##SBATCH --mail-user=you@example.edu
#SBATCH -t 1-00:00:00             ## 1 day wall limit (the CPU run is only a few hours)
#SBATCH --job-name=diff_infer
#SBATCH -o diff_infer_%j.out      ## combined stdout+stderr -> diff_infer_<jobid>.out

## ---------------------------------------------------------------------------
## WHAT THIS RUNS
##   Combined_Fickian_slurm.py  then  Combined_Ito_slurm.py
##   (these are the cluster variants of Combined_Fickian.py / Combined_Ito.py:
##    identical numerics, plus a raw-results .pkl dump, 1 thread/worker, and a
##    worker count read from $SLURM_CPUS_PER_TASK. The originals are untouched.)
##   Each: 100 trials x {DTO, PINN, BiLO}. Dumps raw results to a .pkl, then a PDF.
##   The scripts auto-read $SLURM_CPUS_PER_TASK for the worker count and pin
##   one thread per worker (so 40 cpus == 40 trials in parallel).
##
## ONE-TIME ENV SETUP (do this once on a LOGIN node, not in this script).
## On many clusters the default python has numpy but NOT torch, so build a conda env:
##   conda create -y -n diff python=3.10
##   conda activate diff
##   pip install --upgrade pip
##   pip install torch --index-url https://download.pytorch.org/whl/cpu
##   pip install numpy scipy matplotlib
##   python -c "import torch,scipy,matplotlib,numpy; print(torch.__version__)"
## (Do NOT load a python module alongside conda.)
##
## SUBMIT (from inside the python_paper_figures/ folder):
##   sbatch run_inference.slurm
## ---------------------------------------------------------------------------

## make `conda activate` work in this non-interactive batch job, then activate
## the env that has torch (sourcing ~/.bashrc is unreliable here):
eval "$(conda shell.bash hook)"
conda activate diff                ## <-- the env you built in the one-time setup

## keep each worker single-threaded; parallelism is across the trials
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg               ## headless: savefig writes the PDF, plt.show() is a no-op
export PYTHONUTF8=1                  ## avoid the locale unicode-print crash

cd "$SLURM_SUBMIT_DIR"              ## run from where you submitted (python_paper_figures/)

echo "Host: $(hostname) | cpus/task: ${SLURM_CPUS_PER_TASK} | start: $(date)"
python Combined_Fickian_slurm.py
echo "Fickian finished: $(date)"
python Combined_Ito_slurm.py
echo "Ito finished: $(date)"
echo "ALL DONE: $(date)"

## OUTPUTS (written into this folder; 100 trials each):
##   CCombined_Fickian_b250_t100_3_data.pkl  +  CCombined_Fickian_b250_t100_3.pdf
##   CCombined_Ito_b250_t100_4_data.pkl      +  CCombined_Ito_b250_t100_4.pdf
