import sys
from pathlib import Path
from itertools import product
import os

import numpy as np
from tqdm import tqdm

# Add repo root and code/ to path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "code"))

from config import MNIST_LENS_DIR, RESULT_SUBDIRS
from training import (
    standardize_data,
    eigentask_solver,
    pca_solver,
    fft,
    low_pass,
    downsample_data,
    DNNTrain,
    LinearRegression,
)

seed_list = range(5)
power_list = ["high", "low"]
thresholded_list = [True, False]
stdz_list = [True, False]
num_frames_list = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 40, 60, 80, 100]
train_mode_list = [
    "lpfft",
    "eigen",
    "cg",
    "pca",
]
V_from_max_frame = True

array_task_id = int(sys.argv[1])
power, thresholded, num_frames, stdz, seed = list(
    product(power_list, thresholded_list, num_frames_list, stdz_list, list(seed_list))
)[array_task_id]

dataset_dir = MNIST_LENS_DIR[power]
result_dir = RESULT_SUBDIRS["mnist_nn"]
os.makedirs(result_dir, exist_ok=True)

print("dataset_dir = ", dataset_dir)
print("result_dir = ", result_dir)
print(
    f"power = {power}\nthreshold = {thresholded}\nnum_frames = {num_frames}\nseed = {seed}"
)

NTrain, NVal, NTest = 8000, 2000, 2000
N_samp = NTrain + NTest + NVal
pixels = 45
K_max = pixels**2
max_frame = 100

V_frame = max_frame if V_from_max_frame else num_frames

np.random.seed(seed)
perm = np.random.permutation(N_samp)
labels = np.load(dataset_dir / "mnist_metadata.npz")["labels"]
count = {i: 0 for i in range(10)}
train_indices, val_indices, test_indices = [], [], []
for i in range(N_samp):
    count[labels[perm[i]]] += 1
    if count[labels[perm[i]]] <= NTrain / 10:
        train_indices.append(perm[i])
    elif count[labels[perm[i]]] <= (NTrain + NTest) / 10:
        test_indices.append(perm[i])
    else:
        val_indices.append(perm[i])
perm = np.array(test_indices + val_indices + train_indices)
data_labels = labels[perm]

with np.load(dataset_dir / "background.npz") as data:
    params = data["params"]
    offset, sigma = params[:2]
data_images = []
data_train = []
V = np.zeros((K_max, K_max))
for i in tqdm(range(N_samp)):
    frames = np.load(dataset_dir / "frames" / f"{perm[i]}_EM_Gain_200.npz")["data"]
    if thresholded:
        frames = frames - offset
        frames[frames < 3 * sigma] = 0
    data_images.append(np.mean(frames[:num_frames], axis=0))
    XMat = frames[:V_frame].reshape(V_frame, -1).T
    if i >= NTest + NVal:
        V += np.cov(XMat)
        data_train.append(np.mean(frames, axis=0))
V = V / NTrain

V_norm = np.zeros((K_max, K_max))
for i in tqdm(range(len(V))):
    for j in range(i + 1):
        V_norm[i, j] = V[i, j] / np.sqrt(V[i, i] * V[j, j])
        V_norm[j, i] = V[j, i] / np.sqrt(V[i, i] * V[j, j])

print(
    "V diagonalized check:",
    [
        np.max(np.abs([V_norm[i, i + j] for i in range(len(V_norm) - j)]))
        for j in range(5)
    ],
)

data_images = np.array(data_images)
data_train = np.array(data_train).reshape(len(data_train), -1)
data_labels = np.load(dataset_dir / "mnist_metadata.npz")["labels"][perm]
print("V and averaged data obtained!")

# raw data
data_orig = data_images.reshape(len(data_images), -1)
data_orig_norm = standardize_data(
    data_orig, train_indices=np.arange(NTest + NVal, N_samp), zero_center=stdz
)

# eigentasks
_, _, nsr, r_train = eigentask_solver(data_train, V=V)
data_eigen = data_orig @ r_train.T
data_eigen_norm = standardize_data(
    data_eigen, train_indices=np.arange(NTest + NVal, N_samp), zero_center=stdz
)

data_lpfft = fft(data_images)
data_lpfft_norm = standardize_data(
    data_lpfft, train_indices=np.arange(NTest + NVal, N_samp), zero_center=stdz
)

# pca
_, pca = pca_solver(data_train.reshape(NTrain, -1))
data_pca = (
    data_orig - np.tile(np.mean(data_train.reshape(NTrain, -1), axis=0), (N_samp, 1))
) @ (pca).T
data_pca_norm = standardize_data(
    data_pca, train_indices=np.arange(NTest + NVal, N_samp), zero_center=stdz
)

print("Pre-processing finished!")

init_lr, Epochs = 1e-3, 300

Lmin, Lmax = 1, 45
K_first = 10  # K_list for eigen and pca would be 1, 2, ..., K_first, 16, 25, ...

for train_mode in train_mode_list:

    acc_list_nn = []
    acc_list_nn_running = []
    acc_list_nn_loss = []
    acc_list_linear = []
    n_outs = len(np.unique(data_labels))
    file_name = f"power_{power}_thresholded_{thresholded}_shots_{num_frames}_Vmax_{V_from_max_frame}_train_{train_mode}_seed_{seed}_stdz_{stdz}"

    if (train_mode == "lpfft") or (train_mode == "cg") or (train_mode == "lp"):
        K_list = np.arange(Lmin, Lmax + 1) ** 2
    else:
        K_list = np.concatenate(
            [np.arange(K_first) + 1, np.arange(max(4, Lmin), Lmax + 1) ** 2]
        )

    for K in tqdm(K_list):
        L = int(np.sqrt(K))

        if train_mode == "eigen":
            data_training = data_eigen_norm
        elif train_mode == "cg":
            data_training = downsample_data(data_orig_norm.reshape(-1, 45, 45), L=L)
        elif train_mode == "pca":
            data_training = data_pca_norm
        elif train_mode == "lpfft":
            data_training = low_pass(data_lpfft_norm, L=L)

        modelargs = {"Nunits": [400], "batchnorm": False, "nlaf": "relu"}
        acc_nn, acc_nn_running, acc_nn_loss = DNNTrain(
            data_training,
            data_labels,
            modelargs,
            NTrain=NTrain,
            NVal=NVal,
            NTest=NTest,
            rand=False,
            init_lr=init_lr,
            Epochs=Epochs,
            manual_seed=seed,
            K=K,
            runningAccuracy=True,
            lr_scheduler="plateau_reduce",
            justTrain=True,
        )
        acc_list_nn.append(acc_nn)
        acc_list_nn_running.append(acc_nn_running)
        acc_list_nn_loss.append(acc_nn_loss)

        acc_list_linear.append(
            LinearRegression(
                data_training,
                data_labels,
                K=K,
                NTrain=NTrain,
                NVal=NVal,
                NTest=NTest,
                rand=False,
                bias=True,
            )
        )

    acc_list_nn = np.array(acc_list_nn)
    acc_list_linear = np.array(acc_list_linear)

    np.savez(
        result_dir / (file_name + ".npz"),
        Vmax=V_from_max_frame,
        thresholded=thresholded,
        num_frames=num_frames,
        train_mode=train_mode,
        power=power,
        s_train=nsr,
        acc_nn=acc_list_nn,
        acc_nn_running=acc_list_nn_running,
        acc_nn_loss=acc_list_nn_loss,
        acc_linear=acc_list_linear,
        seed=seed,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        K_list=K_list,
    )
