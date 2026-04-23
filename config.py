"""
Central configuration for the Eigentask Optical Readout project.
All paths are derived from this file's location (the repository root).

To run on a different machine, no changes are needed as long as data/
and results/ are placed alongside this file.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

DATA_DIR = REPO_ROOT / "data"
RESULT_DIR = REPO_ROOT / "results"
FIGURE_DIR = REPO_ROOT / "figures"

# Dataset-specific paths
MNIST_LENS_DIR = {
    "high": DATA_DIR / "lens_mnist_high_power",
    "low": DATA_DIR / "lens_mnist_low_power",
}
MPEG7_LENS_DIR = DATA_DIR / "lens_mpeg7"
SPDNN_DIR = DATA_DIR / "spdnn_mnist"

# Result subdirectories (one per experiment script)
RESULT_SUBDIRS = {
    "mnist_logistic": RESULT_DIR / "mnist_logistic",
    "mnist_nn": RESULT_DIR / "mnist_nn",
    "mpeg7_logistic": RESULT_DIR / "mpeg7_logistic",
    "mpeg7_nn": RESULT_DIR / "mpeg7_nn",
    "spdnn_expts_logistic": RESULT_DIR / "spdnn_expts_logistic",
    "spdnn_expts_nn": RESULT_DIR / "spdnn_expts_nn",
    "spdnn_expts_sim_logistic": RESULT_DIR / "spdnn_expts_sim_logistic",
    "spdnn_expts_sim_nn": RESULT_DIR / "spdnn_expts_sim_nn",
    "spdnn_sim_logistic": RESULT_DIR / "spdnn_sim_logistic",
    "spdnn_sim_nn": RESULT_DIR / "spdnn_sim_nn",
}
