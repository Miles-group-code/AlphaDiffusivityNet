#!/bin/bash
#SBATCH -p standard               ## standard partition
#SBATCH -N 1                      ## one node per array task
#SBATCH --ntasks=1                ## one task; Python multiprocessing spawns the workers
#SBATCH --cpus-per-task=32        ## CPUs per array task. 32 backfills onto partly-free
                                  ## nodes; 40 needs a whole 40-core node and queues much longer.
#SBATCH -A your_slurm_account     ## set to your SLURM allocation/account
## optional job email (set your address and uncomment):
##SBATCH --mail-type=ALL
##SBATCH --mail-user=you@example.edu
#SBATCH -t 1-00:00:00
#SBATCH --job-name=diff_sweep
#SBATCH --array=1-5               ## the 5 manuscript recovery conditions (run in parallel)
#SBATCH -o diff_sweep_%A_%a.out   ## one log per array task: diff_sweep_<jobid>_<taskid>.out

## ---------------------------------------------------------------------------
## Reproduces ALL recovery panels in the manuscript (fig:Combined_Ito, fig:Combined_Fickian).
## Each array task runs one (convention, b0, D-frequency) condition, 100 trials,
## and writes its own  CCombined_<conv>_b<B0>_t100_<FREQ>.pdf  +  _data.pkl.
##
##   task  script                      B0     FREQ  ->  D(x)        output
##   1     Combined_Ito_slurm.py       250    4     ->  sin(8pi x)  CCombined_Ito_b250_t100_4.pdf
##   2     Combined_Ito_slurm.py       1000   4     ->  sin(8pi x)  CCombined_Ito_b1000_t100_4.pdf
##   3     Combined_Fickian_slurm.py   1000   4     ->  sin(8pi x)  CCombined_Fickian_b1000_t100_4.pdf
##   4     Combined_Fickian_slurm.py   250    3     ->  sin(6pi x)  CCombined_Fickian_b250_t100_3.pdf
##   5     Combined_Fickian_slurm.py   1000   3     ->  sin(6pi x)  CCombined_Fickian_b1000_t100_3.pdf
##
## Submit from inside python_paper_figures/ :   sbatch run_recovery_sweep.slurm
## (one-time conda env setup is documented in run_inference.slurm and the README)
## ---------------------------------------------------------------------------

eval "$(conda shell.bash hook)"
conda activate diff

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUTF8=1
export N_TRIALS=100

cd "$SLURM_SUBMIT_DIR"

case $SLURM_ARRAY_TASK_ID in
  1) SCRIPT=Combined_Ito_slurm.py;     export B0=250  FREQ=4 ;;
  2) SCRIPT=Combined_Ito_slurm.py;     export B0=1000 FREQ=4 ;;
  3) SCRIPT=Combined_Fickian_slurm.py; export B0=1000 FREQ=4 ;;
  4) SCRIPT=Combined_Fickian_slurm.py; export B0=250  FREQ=3 ;;
  5) SCRIPT=Combined_Fickian_slurm.py; export B0=1000 FREQ=3 ;;
  *) echo "bad array index $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "Task $SLURM_ARRAY_TASK_ID | $SCRIPT | B0=$B0 FREQ=$FREQ | cpus=$SLURM_CPUS_PER_TASK | host $(hostname) | start $(date)"
python "$SCRIPT"
echo "Task $SLURM_ARRAY_TASK_ID done: $(date)"
