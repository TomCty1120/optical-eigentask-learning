import sys
from pathlib import Path
from itertools import product
import os
import gc

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "code"))

from config import SPDNN_DIR, RESULT_SUBDIRS
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
num_frames_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
N_list = [400, 300, 200, 100, 50]
stdz_list = [True, False]
train_mode_list = [
    "lpfft",
    "eigen",
    "cg",
    "pca",
]  # "coarse grained", "low pass", "eigentask", "pca", "linear regression with MSE"
V_from_max_frame = True

array_task_id = int(sys.argv[1])
num_frames, stdz, seed = list(product(num_frames_list, stdz_list, list(seed_list)))[
    array_task_id
]


dataset_dir = SPDNN_DIR
result_dir = RESULT_SUBDIRS["spdnn_expts_sim_nn"]
os.makedirs(result_dir, exist_ok=True)

print("dataset_dir = ", dataset_dir)
print("result_dir = ", result_dir)
print(f"num_frames = {num_frames}\nseed = {seed}")

for N in N_list:
    print(f"N = {N}")
    with np.load(dataset_dir / f"mnist_spdnn_N{N}_bool.npz") as data:
        train_shots, train_labels = (
            data["train_shots"].astype(int),
            data["train_labels"],
        )
        test_shots, test_labels = data["test_shots"].astype(int), data["test_labels"]
        expt_shots, expt_labels = data["expt_shots"].astype(int), data["expt_labels"]

    NTrain, NVal, NTest = 8000, 2000, 100
    N_samp = NTrain + NTest + NVal
    K_max = N
    max_frame = 30
    V_frame = max_frame if V_from_max_frame else num_frames

    np.random.seed(seed)
    perm = np.random.permutation(len(train_labels))
    train_indices = perm[
        np.concatenate(
            [
                np.where(train_labels[perm] == i)[0][: int(NTrain / 10)]
                for i in range(10)
            ]
        )
    ]
    val_indices = perm[
        np.concatenate(
            [
                np.where(train_labels[perm] == i)[0][
                    int(NTrain / 10) : int((NTrain + NVal) / 10)
                ]
                for i in range(10)
            ]
        )
    ]
    data_labels = np.concatenate(
        (expt_labels, train_labels[val_indices], train_labels[train_indices])
    )
    print([np.sum(train_labels[train_indices] == i) for i in range(10)])
    print([np.sum(train_labels[val_indices] == i) for i in range(10)])

    data_images = list(np.mean(test_shots[:100, :num_frames], axis=1))
    data_train = []
    V = np.zeros((K_max, K_max))

    for i in tqdm(val_indices):
        frames = train_shots[i]
        data_images.append(np.mean(frames[:num_frames], axis=0))

    for i in tqdm(train_indices):
        frames = train_shots[i]
        data_images.append(np.mean(frames[:num_frames], axis=0))
        data_train.append(np.mean(frames[:V_frame], axis=0))
        XMat = frames[:V_frame].reshape(V_frame, -1).T
        V += np.cov(XMat)
    V = V / NTrain
    data_images = np.array(data_images)
    data_train = np.array(data_train)
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

    # low pass filtering
    data_lpfft = fft(data_images)
    data_lpfft_norm = standardize_data(
        data_lpfft, train_indices=np.arange(NTest + NVal, N_samp), zero_center=stdz
    )

    # pca
    _, pca = pca_solver(data_train.reshape(NTrain, -1))
    data_pca = (
        data_orig
        - np.tile(np.mean(data_train.reshape(NTrain, -1), axis=0), (N_samp, 1))
    ) @ (pca).T
    data_pca_norm = standardize_data(
        data_pca, train_indices=np.arange(NTest + NVal, N_samp), zero_center=stdz
    )

    print("Pre-processing finished!")

    init_lr, Epochs = 1e-3, 300

    Lmin, Lmax = 1, int(np.sqrt(N))
    K_first = 10  # K_list for eigen and pca would be 1, 2, ..., K_first, 16, 25, ...

    # Lmin, Lmax = 3, 4
    # K_first = 1 # K_list for eigen and pca would be 1, 2, ..., K_first, 16, 25, ...

    for train_mode in train_mode_list:

        acc_list_nn = []
        acc_list_nn_running = []
        acc_list_nn_loss = []
        acc_list_linear = []
        n_outs = len(np.unique(data_labels))
        file_name = (
            f"{N}_neurons_{num_frames}_shots_{train_mode}_seed_{seed}_stdz_{stdz}"
        )

        if (train_mode == "lpfft") or (train_mode == "cg") or (train_mode == "lp"):
            K_list = np.arange(Lmin, Lmax + 1) ** 2
        else:
            K_list = np.concatenate(
                [np.arange(K_first) + 1, np.arange(max(4, Lmin), Lmax + 1) ** 2]
            )

        for K in tqdm(K_list):
            if train_mode == "eigen":
                data_training = data_eigen_norm
            elif train_mode == "cg":
                data_training = downsample_data(data_orig_norm, L=K, dim="1D")
            elif train_mode == "pca":
                data_training = data_pca_norm
            elif train_mode == "lpfft":
                data_training = low_pass(data_lpfft_norm, L=K)

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
            num_frames=num_frames,
            train_mode=train_mode,
            s_train=nsr,
            acc_nn=acc_list_nn,
            acc_nn_running=acc_list_nn_running,
            acc_nn_loss=acc_list_nn_loss,
            acc_linear=acc_list_linear,
            seed=seed,
            train_indices=train_indices,
            val_indices=val_indices,
            K_list=K_list,
        )
    for var in [
        "data_orig_norm",
        "data_eigen_norm",
        "data_eigenzc_norm",
        "data_lpfft_norm",
        "data_pca_norm",
    ]:
        if var in dir():
            exec(f"del {var}", globals())
    gc.collect()
