# Optical Eigentask Learning

Code for training, analysis, and figure generation accompanying:

> T. Chen\*, M. M. Sohoni\*, S. A. Khan, J. Laydevant, S.-Y. Ma, T. Wang,  
> P. L. McMahon†, and H. E. Türeci†,  
> *Representation learning for optical readout using Eigentasks* (2026).

---

## Related records

- **Paper:** `<ARTICLE_DOI_OR_URL>`
- **Code DOI (Zenodo software record):** `10.5281/zenodo.<CODE_DOI>`
- **Data DOI (Zenodo data record):** `10.5281/zenodo.<DATA_DOI>`

---

## What this repository contains

This repository contains the code used for:

- feature extraction and classifier training
- figure generation
- EMCCD calibration analysis
- experiment entry points and Slurm job scripts

Large datasets and precomputed outputs are **not** stored in GitHub.
They are hosted in the companion Zenodo data record.

---

## Repository structure

```text
optical-eigentask-learning/
├── README.md
├── LICENSE
├── CITATION.cff
├── .zenodo.json
├── config.py
├── environment.yml
├── requirements.txt
└── code/
    ├── paper_figures.ipynb
    ├── training.py
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

PyTorch may need a platform-specific install command depending on your
Python / CUDA setup.

---

## Companion data and results

The companion Zenodo data record contains:

- `lens_mnist_low_power.tar`
- `lens_mnist_high_power.tar`
- `lens_mpeg7.tar`
- `spdnn_mnist.tar`
- `results.tar`
- `figures.zip` *(optional)*

**Data DOI:** `10.5281/zenodo.<DATA_DOI>`

The code in this repository expects the extracted data under `data/`,
the precomputed outputs under `results/`, and optional regenerated or
pre-rendered figures under `figures/`, all at the repository root.

---

## Expected local layout after extracting the data record

```text
optical-eigentask-learning/
├── README.md
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
└── figures/   # optional
```

---

## Downloading and extracting the companion data record

Create or enter the repository root, then unpack the data archives there:

```bash
# already inside optical-eigentask-learning/
mkdir -p data
tar -xf lens_mnist_low_power.tar  -C data/
tar -xf lens_mnist_high_power.tar -C data/
tar -xf lens_mpeg7.tar            -C data/
tar -xf spdnn_mnist.tar           -C data/
tar -xf results.tar
unzip figures.zip                 # optional
```

---

## Reproducing figures

Open the repository root in VS Code or start Jupyter from the repository root,
then open:

```text
code/paper_figures.ipynb
```

For a clean reproduction, run the notebook **top-to-bottom once**.
Later sections reuse setup and data-loading state from earlier cells.

```bash
conda activate eigentask
jupyter notebook
# then open code/paper_figures.ipynb
```

The notebook writes output into `figures/` and may overwrite files with the
same names.

---

## Re-running training

Each script in `code/experiments/` is a Slurm-array entry point whose integer
CLI argument indexes a parameter combination.

```bash
# HPC
sbatch code/slurm_files/mnist_logistic.slurm

# one configuration locally
python code/experiments/mnist_logistic.py 0
```

---

## Re-running EMCCD calibration

```bash
jupyter notebook code/data_processing_generation/emccd_calibration_power_estimation.ipynb
```

This notebook regenerates the calibration outputs used for the lens-based
datasets.

---

## Citation

If you use this repository, please cite the **software release** and the
associated paper.

- See `CITATION.cff` for the repository citation metadata.
- Use the Zenodo **code DOI** for a frozen release corresponding to the paper.

---

## License

- Code in this repository: **MIT License**
- Companion data record: see the data-record README for dataset licensing and attribution
