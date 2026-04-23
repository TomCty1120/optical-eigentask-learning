# Slurm Job Scripts

Reference job scripts for reproducing the training results on an HPC cluster with the Slurm scheduler. These were originally run on Princeton University's Della cluster.

## Before using

Adapt the following to your cluster:

- `module load` — replace with your cluster's module system and versions
- `conda activate` — replace with your environment name
- `--mem` — adjust based on available memory per node
- `--time` — your hardware may differ
- Create a `logs/` directory in the repo root: `mkdir -p logs`

All scripts should be submitted **from the repository root**, e.g.:

```bash
sbatch slurm/mnist_logistic.slurm
```

## Job summary

| Script | Array range | Tasks | 
|--------|------------|-------|
| mnist_logistic | 0–559 | 560 |
| mnist_nn | 0–559 | 560 | 
| mpeg7_logistic | 0–209 | 210 | 
| mpeg7_nn | 0–209 | 210 |
| spdnn_expts_logistic | 0–139 | 140 | 
| spdnn_expts_nn | 0–139 | 140 |
| spdnn_expts_sim_logistic | 0–139 | 140 | 
| spdnn_expts_sim_nn | 0–139 | 140 | 
| spdnn_sim_logistic | 0–139 | 140 |
| spdnn_sim_nn | 0–139 | 140 | ~30 min |

Times are approximate. If a GPU is available, PyTorch will use it automatically
(the code checks `torch.cuda.is_available()`), in which case the NN scripts
will run significantly faster. Add `#SBATCH --gres=gpu:1` to request a GPU.

## Parameter space

Each script uses `itertools.product(...)` indexed by `$SLURM_ARRAY_TASK_ID` to sweep over its parameter space. The array range in each `.slurm` file matches the total number of parameter combinations. See the top of each Python script for the exact parameter lists.
