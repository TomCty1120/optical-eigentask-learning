# Measurement-adapted eigentask representations for photon-limited optical readout: code

Code for training, analysis, calibration, and figure generation accompanying:

> T. Chen\*, M. M. Sohoni\*, S. A. Khan, J. Laydevant, S.-Y. Ma, T. Wang,  
> P. L. McMahon†, and H. E. Türeci†,  
> *Measurement-adapted eigentask representations for photon-limited optical readout* (2026).

---

## Related links

- **GitHub repository:** <https://github.com/TomCty1120/optical-eigentask-learning>
- **Zenodo archive DOI:** `10.5281/zenodo.19888614` 
- **Paper / preprint:** to be added when available

The GitHub repository is the primary location for source code. The companion
Zenodo archive is a manually deposited combined record, not a Zenodo-GitHub
auto-ingested software release. It contains the datasets, precomputed results,
optional pre-rendered figures, and an archival snapshot of the code corresponding
to the paper.

---

## What this repository contains

This repository contains the code used for:

- eigentask, PCA, Fourier low-pass, and coarse-graining feature extraction
- logistic-regression and neural-network classifier training
- figure generation
- EMCCD calibration and photon-count estimation
- experiment entry points and Slurm job scripts

Large datasets and precomputed outputs are **not** stored in GitHub. They are
hosted in the companion Zenodo archive.

---

## Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/TomCty1120/optical-eigentask-learning.git
cd optical-eigentask-learning

# 2. Set up the environment
conda env create -f environment.yml
conda activate eigentask

# 3. Download the needed archives from the Zenodo record
#    DOI: 10.5281/zenodo.19888614
#    Place the downloaded archives in this repository root, then extract
#    only the archives you downloaded:
mkdir -p data

[ -f lens_mnist_low_power.tar ]  && tar -xf lens_mnist_low_power.tar  -C data/
[ -f lens_mnist_high_power.tar ] && tar -xf lens_mnist_high_power.tar -C data/
[ -f lens_mpeg7.tar ]            && tar -xf lens_mpeg7.tar            -C data/
[ -f spdnn_mnist.tar ]           && tar -xf spdnn_mnist.tar           -C data/
[ -f results.tar ]               && tar -xf results.tar
[ -f figures.zip ]               && unzip figures.zip                 # optional; pre-rendered figures only

# 4. Open the figure-generation notebook
jupyter notebook code/paper_figures.ipynb
```

`figures.zip` is optional and is **not** required to regenerate figures.

---

## Repository structure

```text
optical-eigentask-learning/
├── README.md
├── LICENSE                      # MIT License for code
├── CITATION.cff
├── config.py                    # Repository-relative path configuration
├── environment.yml              # Conda environment specification
├── requirements.txt             # pip dependencies (alternative to conda)
└── code/
    ├── paper_figures.ipynb
    ├── training.py              # Core library: eigentask solver, classifiers, transforms
    ├── data_processing_generation/
    │   ├── emccd_calibration_power_estimation.ipynb
    │   └── spdnn_data_generation.ipynb
    ├── experiments/
    │   ├── mnist_logistic.py, mnist_nn.py
    │   ├── mpeg7_logistic.py, mpeg7_nn.py
    │   └── spdnn_{expts,expts_sim,sim}_{logistic,nn}.py
    └── slurm_files/
        ├── README.md
        └── *.slurm
```

---

## Environment setup

```bash
# Option A: conda
conda env create -f environment.yml
conda activate eigentask

# Option B: pip
pip install -r requirements.txt
```

PyTorch may require a platform-specific installation command depending on your
Python and CUDA setup.

---

## Companion Zenodo archive

The companion Zenodo archive is a manually deposited combined record, not a
Zenodo-GitHub auto-ingested software release. It contains:

- `code.zip` — archival snapshot of the `code/` directory
- `config.py`, `environment.yml`, `requirements.txt`, `LICENSE`, `CITATION.cff`
- `lens_mnist_low_power.tar`
- `lens_mnist_high_power.tar`
- `lens_mpeg7.tar`
- `spdnn_mnist.tar`
- `results.tar`
- `figures.zip` *(optional pre-rendered figure outputs)*

**Zenodo DOI:** `10.5281/zenodo.19888614` 

If you are using this GitHub repository, you generally do **not** need to
download `code.zip` from Zenodo; download only the data/result archives needed
for your task.

---

## Download guide: what do I need?

| Task | `results` | `lens_mnist_low_power` | `lens_mnist_high_power` | `lens_mpeg7` | `spdnn_mnist` |
|---|:---:|:---:|:---:|:---:|:---:|
| Inspect code only |  |  |  |  |  |
| Regenerate Fig. 1 (schematic) |  |  |  |  |  |
| Regenerate Fig. 2 — accuracy panels | ✓ |  |  |  |  |
| Regenerate Fig. 2 — full (SNR + masks) | ✓ | ✓ |  |  |  |
| Regenerate Fig. 3 (MPEG-7) | ✓ |  |  | ✓ |  |
| Regenerate Fig. 4 (SPDNN) | ✓ |  |  |  |  |
| Regenerate Figs. S3–S6 (MNIST supplementary) | ✓ | ✓ | ✓ |  |  |
| Regenerate Figs. S8, S9 (SPDNN supplementary) | ✓ |  |  |  |  |
| Regenerate Fig. S2 (calibration) † |  | (dark) | (dark) | (dark) |  |
| Re-run training from scratch — MNIST |  | ✓ | ✓ |  |  |
| Re-run training from scratch — MPEG-7 |  |  |  | ✓ |  |
| Re-run training from scratch — SPDNN |  |  |  |  | ✓ |
| Full reproduction from raw data |  | ✓ | ✓ | ✓ | ✓ |

† For Fig. S2 you only need the `raw_dark_frames/` subfolder of each lens
dataset, not the full per-frame data.

Fig. 2 is split into two rows: the accuracy panels (e–h) are produced entirely
from `results/`, while the SNR histograms (c, d) and eigentask masks (b)
re-solve the generalized eigenvalue problem on the raw low-power frames.

Fig. 4 and Figs. S8–S9 are generated from the precomputed SPDNN result files
under `results/`; the raw `spdnn_mnist` archive is only needed if you want to
re-run the SPDNN training/analysis from scratch.

---

## Expected local layout after extracting the Zenodo archives

```text
optical-eigentask-learning/
├── README.md
├── LICENSE
├── CITATION.cff
├── config.py
├── environment.yml
├── requirements.txt
├── code/
├── data/
│   ├── lens_mnist_low_power/
│   ├── lens_mnist_high_power/
│   ├── lens_mpeg7/
│   └── spdnn_mnist/
├── results/
└── figures/                     # optional
```

---

## Reproducing figures

Open the repository root in VS Code or start Jupyter from the repository root,
then open:

```text
code/paper_figures.ipynb
```

For a clean reproduction, run the notebook **top-to-bottom once**. Later
sections reuse setup and data-loading state from earlier cells; if you execute
only part of the notebook, run at least the relevant setup/load-data cells first.

The notebook writes output into `figures/` and may overwrite files with the same
names.

---

## Re-running training

Each script in `code/experiments/` is a Slurm-array entry point whose integer
CLI argument indexes a parameter combination.

```bash
# HPC
mkdir -p logs
sbatch code/slurm_files/mnist_logistic.slurm

# one configuration locally
python code/experiments/mnist_logistic.py 0
```

---

## Re-running EMCCD calibration

```bash
jupyter notebook code/data_processing_generation/emccd_calibration_power_estimation.ipynb
```

The values reported in the calibration table are the calibrated camera gain
`g` estimated from the variance-vs-mean slope, not `ηg`. The photon-count
estimates then use this calibrated gain together with the detector quantum
efficiency convention used in the paper.

---

## Citation

If you use this repository, please cite the associated paper and the Zenodo
archive. See `CITATION.cff` for repository citation metadata.

---

## License

- Code in this repository: **MIT License**
- Companion data and precomputed results: **CC BY 4.0**
