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

from config import MPEG7_LENS_DIR, RESULT_SUBDIRS
from training import (
    standardize_data,
    eigentask_solver,
    pca_solver,
    fft,
    low_pass,
    downsample_data,
    LogisticTrain,
    LinearRegression,
)

seed_list = range(5)
N_class_list = [70, 60, 50, 40, 30, 20, 10]
# thresholded_list = [True, False]
stdz_list = [False, True]
num_frames_list = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [15, 20, 25, 30, 50]]
train_mode_list = [
    "eigen",
    "lpfft",
    "cg",
    "pca",
]  # "coarse grained", "low pass", "eigentask", "pca", "linear regression with MSE"
thresholded = False
V_from_max_frame = True

val_fold = 3

array_task_id = int(sys.argv[1])
stdz, N_class, seed, num_frames_div = list(
    product(stdz_list, N_class_list, list(seed_list), num_frames_list)
)[array_task_id]

dataset_dir = MPEG7_LENS_DIR
result_dir = RESULT_SUBDIRS["mpeg7_logistic"]
os.makedirs(result_dir, exist_ok=True)

print("dataset_dir = ", dataset_dir)
print("result_dir = ", result_dir)
print(f"N_class = {N_class}\nseed = {seed}")

for num_frames in num_frames_div:
    print(f"num_frames = {num_frames}")
    np.random.seed(seed)
    perm = np.random.permutation(20)
    pixels = 39
    max_frame = 300
    V_frame = max_frame if V_from_max_frame else num_frames

    cv_acc_list_logistic = {
        train_mode: [[] for _ in range(val_fold)] for train_mode in train_mode_list
    }
    cv_acc_list_logistic_running = {
        train_mode: [[] for _ in range(val_fold)] for train_mode in train_mode_list
    }
    cv_acc_list_logistic_loss = {
        train_mode: [[] for _ in range(val_fold)] for train_mode in train_mode_list
    }
    cv_acc_list_linear = {
        train_mode: [[] for _ in range(val_fold)] for train_mode in train_mode_list
    }

    full_acc_list_logistic = {train_mode: [] for train_mode in train_mode_list}
    full_acc_list_logistic_running = {train_mode: [] for train_mode in train_mode_list}
    full_acc_list_logistic_loss = {train_mode: [] for train_mode in train_mode_list}
    full_acc_list_linear = {train_mode: [] for train_mode in train_mode_list}

    cv_train_indices_list = [[] for _ in range(val_fold)]
    cv_val_indices_list = [[] for _ in range(val_fold)]
    cv_test_indices_list = [[] for _ in range(val_fold)]

    for run in range(val_fold + 1):
        K_max = pixels**2
        isCV = run in range(val_fold)
        if isCV:
            NTrain, NVal, NTest = 10 * N_class, 5 * N_class, 5 * N_class
            cv_test_indices_list[run] = test_indices = np.array(
                [perm[:5] + i * 20 for i in range(N_class)]
            ).flatten()
            cv_val_indices_list[run] = val_indices = np.array(
                [perm[5 * (run + 1) : 5 * (run + 2)] + i * 20 for i in range(N_class)]
            ).flatten()
            cv_train_indices_list[run] = train_indices = np.setdiff1d(
                np.arange(20 * N_class), np.concatenate([test_indices, val_indices])
            )
        else:
            NTrain, NVal, NTest = 15 * N_class, 0, 5 * N_class
            full_test_indices_list = test_indices = np.array(
                [perm[:5] + i * 20 for i in range(N_class)]
            ).flatten()
            full_val_indices_list = val_indices = np.array([], dtype=int)
            full_train_indices_list = train_indices = np.setdiff1d(
                np.arange(20 * N_class), np.concatenate([test_indices, val_indices])
            )
        N_samp = NTrain + NVal + NTest

        indices = np.concatenate((test_indices, val_indices, train_indices)).astype(int)

        data_images = []
        data_train = []
        V = np.zeros((K_max, K_max))
        for i in tqdm(range(N_samp)):
            frames = np.load(dataset_dir / "frames" / f"{indices[i]}_EM_Gain_1000.npz")[
                "data"
            ]
            data_images.append(np.mean(frames[:num_frames], axis=0))
            XMat = frames[:V_frame].reshape(V_frame, -1).T
            if i >= (NTest + NVal):
                V += np.cov(XMat)
                data_train.append(np.mean(frames[:V_frame], axis=0))
        V = V / NTrain
        data_images = np.array(data_images)
        data_train = np.array(data_train).reshape(NTrain, -1)
        data_labels = np.repeat(np.arange(70), 20)[indices]
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

        init_lr, Epochs = (1e-3 if stdz else 0.5), 300

        Lmin, Lmax = 1, pixels
        K_first = (
            10  # K_list for eigen and pca would be 1, 2, ..., K_first, 16, 25, ...
        )

        for train_mode in train_mode_list:

            n_outs = len(np.unique(data_labels))

            if (train_mode == "lpfft") or (train_mode == "cg") or (train_mode == "lp"):
                K_list = np.arange(Lmin, Lmax + 1) ** 2
            else:
                K_max = min(pixels**2, len(train_indices))
                K_list = np.concatenate(
                    [
                        np.arange(K_first) + 1,
                        np.arange(max(4, Lmin), min(Lmax, int(np.sqrt(K_max))) + 1)
                        ** 2,
                    ]
                )
            print(f"K_list for {train_mode}:", K_list)

            for K in tqdm(K_list):
                L = int(np.sqrt(K))

                if train_mode == "eigen":
                    data_training = data_eigen_norm
                elif train_mode == "cg":
                    data_training = downsample_data(
                        data_orig_norm.reshape(-1, pixels, pixels), L=L
                    )
                elif train_mode == "pca":
                    data_training = data_pca_norm
                elif train_mode == "lpfft":
                    data_training = low_pass(data_lpfft_norm, L=L)

                acc_logistic, acc_logistic_running, acc_logistic_loss = LogisticTrain(
                    data_training,
                    data_labels,
                    NTrain=NTrain,
                    NVal=NVal,
                    NTest=NTest,
                    rand=False,
                    init_lr=init_lr,
                    Epochs=Epochs,
                    manual_seed=seed,
                    K=K,
                    runningAccuracy=True,
                    lr_scheduler="step_decay",
                    justTrain=True,
                )

                if isCV:
                    cv_acc_list_logistic[train_mode][run].append(acc_logistic)
                    cv_acc_list_logistic_running[train_mode][run].append(
                        acc_logistic_running
                    )
                    cv_acc_list_logistic_loss[train_mode][run].append(acc_logistic_loss)
                    cv_acc_list_linear[train_mode][run].append(
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
                else:
                    full_acc_list_logistic[train_mode].append(acc_logistic)
                    full_acc_list_logistic_running[train_mode].append(
                        acc_logistic_running
                    )
                    full_acc_list_logistic_loss[train_mode].append(acc_logistic_loss)
                    full_acc_list_linear[train_mode].append(
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

    for train_mode in train_mode_list:

        cv_acc_list_logistic[train_mode] = np.array(cv_acc_list_logistic[train_mode])
        cv_acc_list_logistic_running[train_mode] = np.array(
            cv_acc_list_logistic_running[train_mode]
        )
        cv_acc_list_logistic_loss[train_mode] = np.array(
            cv_acc_list_logistic_loss[train_mode]
        )
        cv_acc_list_linear[train_mode] = np.array(cv_acc_list_linear[train_mode])
        full_acc_list_logistic[train_mode] = np.array(
            full_acc_list_logistic[train_mode]
        )
        full_acc_list_logistic_running[train_mode] = np.array(
            full_acc_list_logistic_running[train_mode]
        )
        full_acc_list_logistic_loss[train_mode] = np.array(
            full_acc_list_logistic_loss[train_mode]
        )
        full_acc_list_linear[train_mode] = np.array(full_acc_list_linear[train_mode])

        if (train_mode == "lpfft") or (train_mode == "cg") or (train_mode == "lp"):
            cv_K_list = full_K_list = np.arange(Lmin, Lmax + 1) ** 2
        else:
            cv_K_max = min(pixels**2, 10 * N_class)
            cv_K_list = np.concatenate(
                [
                    np.arange(K_first) + 1,
                    np.arange(max(4, Lmin), min(Lmax, int(np.sqrt(cv_K_max))) + 1) ** 2,
                ]
            )
            full_K_max = min(pixels**2, 15 * N_class)
            full_K_list = np.concatenate(
                [
                    np.arange(K_first) + 1,
                    np.arange(max(4, Lmin), min(Lmax, int(np.sqrt(full_K_max))) + 1)
                    ** 2,
                ]
            )
        file_name = f"n_class_{N_class}_thresholded_{thresholded}_shots_{num_frames}_Vmax_{V_from_max_frame}_train_{train_mode}_seed_{seed}_stdz_{stdz}"
        np.savez(
            result_dir / (file_name + ".npz"),
            Vmax=V_from_max_frame,
            thresholded=thresholded,
            num_frames=num_frames,
            train_mode=train_mode,
            s_train=nsr,
            cv_acc_logistic=cv_acc_list_logistic[train_mode],
            cv_acc_logistic_running=cv_acc_list_logistic_running[train_mode],
            cv_acc_logistic_loss=cv_acc_list_logistic_loss[train_mode],
            cv_acc_linear=cv_acc_list_linear[train_mode],
            full_acc_logistic=full_acc_list_logistic[train_mode],
            full_acc_logistic_running=full_acc_list_logistic_running[train_mode],
            full_acc_logistic_loss=full_acc_list_logistic_loss[train_mode],
            full_acc_linear=full_acc_list_linear[train_mode],
            seed=seed,
            cv_train_indices=cv_train_indices_list,
            cv_val_indices=cv_val_indices_list,
            cv_test_indices=cv_test_indices_list,
            full_train_indices=full_train_indices_list,
            full_test_indices=full_test_indices_list,
            cv_K_list=cv_K_list,
            full_K_list=full_K_list,
        )
