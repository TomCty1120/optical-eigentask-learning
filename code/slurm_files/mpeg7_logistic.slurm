#!/bin/bash
#SBATCH --job-name=mpeg7_logistic # create a short name for your job
#SBATCH --output=logs/array_job_%A_%a.out
#SBATCH --error=logs/array_job_%A_%a.err
#SBATCH --array=0-209
#SBATCH --nodes=1                 # node count
#SBATCH --ntasks=1                # total number of tasks across all nodes
#SBATCH --cpus-per-task=1         # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=1G                  # memory per cpu-core (4G per CPU-core is default)
#SBATCH --time=23:59:00           # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2025.6      # adjust to your cluster
conda activate eigentask          # adjust to your environment

python code/experiments/mpeg7_logistic.py $SLURM_ARRAY_TASK_ID
