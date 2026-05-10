"""
training.py
===============================================================================
Training and feature-extraction utilities for the manuscript:

    "Measurement-Adapted Eigentask Representations for Photon-Limited Optical Readout"
    T. Chen, M. M. Sohoni, S. A. Khan, J. Laydevant, S.-Y. Ma, T. Wang,
    P. L. McMahon, H. E. Tureci (2026).

This module implements the core building blocks used to evaluate the four
noise-mitigation representations compared in the manuscript — eigentasks,
PCA, Fourier-domain low-pass filtering, and spatial coarse graining — and
to train the downstream classifiers (multinomial logistic regression or a
multilayer perceptron) on the resulting features.

-------------------------------------------------------------------------------
Contents (high level)
-------------------------------------------------------------------------------
Data utilities
    Torch_Dataset            Wrap a (possibly complex-valued) feature array
                             and its labels as a torch Dataset.
    train_test_generator     Random or sequential split of data into
                             train / (val) / test subsets (Methods,
                             "Dataset-specific data splits").
    standardize_data         Feature-wise standardization OR global
                             non-zero-centered rescaling (the two
                             normalizations described in Methods,
                             "Downstream classifiers, optimization,
                             and reporting").
    set_seed                 Deterministic seeding across numpy / torch /
                             random / CUDA / cuDNN.

Representation transforms  W : R^K -> R^{K_r}  (Eq. (2) in the manuscript)
    eigentask_solver         Construct the eigentask basis by solving the
                             generalized eigenvalue problem V r = (1/alpha^2) G r
                             (Eq. (3)) and project measured features onto it.
    pca_solver               Construct the PCA basis (eigenvectors of the
                             empirical covariance C, Methods Eq. (10))
                             and project data onto it.
    fft + low_pass           Fourier-domain low-pass filtering: retain the
                             real and imaginary parts of the low-frequency
                             components (Methods, "Fourier-domain low-pass
                             filtering and spatial coarse graining").
    downsample_data          Spatial coarse graining by linear-interpolation
                             downsampling (Methods, same section).

Classifiers and trainers    F{.} in Eq. (2) of the manuscript
    LR_classifier            Single-layer multinomial logistic regression
                             (Methods, "Downstream classifiers").
    DNN                      Fully connected MLP with configurable hidden
                             widths. The manuscript uses a single hidden
                             layer of 400 ReLU units.
    LinearRegression         Closed-form least-squares linear readout used
                             for quick baselines.
    LogisticTrain            Full training loop for LR_classifier with
                             AdamW + StepLR / ReduceLROnPlateau scheduling
                             as described in Methods.
    DNNTrain                 Same training loop structure for an MLP.
    get_accuracy             Argmax-based classification accuracy.

-------------------------------------------------------------------------------
Conventions
-------------------------------------------------------------------------------
Throughout the code and the manuscript, for each input u(n) we collect up to
Smax single-shot sensor outputs {X^(s)(u(n))} and form an S-shot mean
    X_bar_S(u(n)) = (1/S) * sum_{s=1..S} X^(s)(u(n))     (manuscript Eq. (11)).
Features have shape N x K where N is the number of inputs and K is the
number of raw readout channels (pixels for the lens-based front end, or
single-photon detectors for the SPDNN front end).

A representation transform W in R^{K_r x K} (with bias b) is applied to give
    Y_bar(u) = W X_bar(u) + b                            (manuscript Eq. (2)),
and the downstream classifier F{.} is trained on Y_bar. For eigentasks b = 0
and for PCA, reproducing the same affine map on unseen data additionally
requires the training mean; this helper returns projected scores and components,
not an explicit bias term.”; see the corresponding solvers below.

Complex-valued inputs (e.g. Fourier features) are handled by concatenating
real and imaginary parts along the feature axis; see `Torch_Dataset`.
===============================================================================
"""

import torch.nn.functional as F
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import numpy.linalg as la
from scipy.ndimage import zoom
from tqdm import tqdm
import random

# =============================================================================
# Data utilities
# =============================================================================


class Torch_Dataset(Dataset):
    """PyTorch Dataset wrapper for (possibly complex-valued) feature arrays.

    A single item is a pair (features, label), where `features` is a 1-D
    float32 tensor living on `dev` (GPU if available, else CPU). Complex
    inputs are converted to real-valued vectors of length 2K by
    concatenating real and imaginary parts; this is the convention used
    by `LR_classifier` and `DNN` when `iscomplex == True`.
    """

    def __init__(self, data, labels, dev):
        # Move the tensors to the target device once during dataset
        # construction so downstream iteration does not repeat host-to-device
        # copies.
        device = _resolve_device(dev)

        # Detect whether the feature array carries non-trivial imaginary
        # content. The 1e-15 tolerance guards against round-off in arrays
        # that are numerically real but stored as complex dtype.
        self.iscomplex = True if (np.max(data.imag) > 1e-15) else False

        self.labels = torch.tensor(labels, dtype=int, device=device)

        if self.iscomplex == False:
            # Real-valued features: take the real part (imag is known to be
            # numerically zero) and store as float32.
            self.data = torch.tensor(data.real, dtype=torch.float32, device=device)
        else:
            # Complex-valued features: stack real and imaginary parts of
            # each sample into a single real vector of length 2K. This is
            # the representation expected by the `iscomplex` branch of
            # LR_classifier / DNN.
            self.data = torch.tensor(
                np.array(
                    [
                        np.concatenate([im.real.flatten(), im.imag.flatten()])
                        for im in data
                    ]
                ),
                dtype=torch.float32,
                device=device,
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class LR_classifier(nn.Module):
    """Single-layer multinomial logistic-regression classifier F{.}.

    Implements the logistic readout used in Figs. 2, 3, 4 of the
    manuscript and in Methods ("Downstream classifiers, optimization,
    and reporting"): a single linear layer mapping the K_r-dimensional
    transformed features to C output logits, trained with cross-entropy
    loss. For complex-valued inputs the effective input width is 2*K_r
    (real and imaginary parts concatenated; see `Torch_Dataset`).
    """

    def __init__(self, input_size: int, n_outs: int, iscomplex: bool):
        super(LR_classifier, self).__init__()
        self.input_size = input_size
        self.n_outputs = n_outs
        self.iscomplex = iscomplex
        #
        if iscomplex:
            # For complex features the classifier sees real/imag concatenated.
            self.classifier = nn.Linear(2 * self.input_size, self.n_outputs)
        else:
            self.classifier = nn.Linear(self.input_size, self.n_outputs)

    def forward(self, X):
        # Reshape is kept explicit to support both (batch, K) and
        # (batch, K, 1, ...) layouts that arise when features carry an
        # auxiliary spatial dimension.
        if self.iscomplex:
            Y = self.classifier(X.reshape(X.shape[0], 2 * self.input_size))
        else:
            Y = self.classifier(X.reshape(X.shape[0], self.input_size))
        return Y


def _resolve_device(dev=None):
    """Return a valid torch device.

    If dev is None, use cuda:0 when CUDA is available, otherwise CPU.
    If a requested CUDA index is unavailable, fall back to cuda:0.
    """
    if dev is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(dev, torch.device):
        dev = str(dev)
    dev_str = str(dev)
    if dev_str.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if ":" in dev_str:
            try:
                idx = int(dev_str.split(":", 1)[1])
            except ValueError:
                return torch.device("cuda:0")
            if idx < 0 or idx >= torch.cuda.device_count():
                return torch.device("cuda:0")
        return torch.device(dev_str)
    return torch.device(dev_str)


def get_accuracy(logit, target):
    """Return the classification accuracy (in percentage points x batch size).

    The caller divides by the number of samples to obtain a percentage.
    Using `.sum()` over the batch (rather than the mean) makes the
    accumulator additive across mini-batches in the training loop.
    """
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects
    return accuracy.item()


def set_seed(seed):
    """Fully deterministic seeding across numpy, python random, and torch.

    Matches the reproducibility protocol described in Methods: random
    seeds are fixed for NumPy, Python, and PyTorch (including CUDA), and
    deterministic cuDNN settings are used. This is important for the
    five-repetition error bars reported in every figure.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_test_generator(
    data_images,
    data_labels,
    NTrain=None,
    NTest=None,
    rand=True,
    rangeTest=None,
    returnIndices=False,
):
    """Split a dataset into train / test subsets.

    A thin helper that produces class-agnostic random (or sequential)
    train/test splits. In the main experiments the calling code
    additionally partitions the training portion into train/val (see
    `LogisticTrain`, `DNNTrain`), yielding the three-way train / val /
    test split described in Methods.

    Parameters
    ----------
    data_images, data_labels : np.ndarray
        Feature array of shape (N, ...) and integer label array of
        shape (N,).
    NTrain, NTest : int, optional
        Requested training / test set sizes. If `NTrain + NTest`
        exceeds the available data, a default 80/20 split is used.
    rand : bool, default True
        If True, permute indices before splitting; otherwise take the
        first NTrain for training and the next NTest for testing.
    rangeTest : iterable of int, optional
        Restrict test indices to this whitelist (used e.g. for the
        SPDNN analysis where the test set is the specific 100
        experimental digits reported in Ref. [46]).
    returnIndices : bool
        If True, additionally return the numeric indices of the
        training and test samples within the original array.
    """
    N_total_samp = len(data_images)
    # Honour the caller's requested sizes when they fit in the data,
    # otherwise fall back to the default 80/20 partition.
    N_train_samp = (
        NTrain
        if (NTrain and NTrain + NTest <= N_total_samp)
        else int(0.8 * N_total_samp)
    )
    N_test_samp = (
        NTest if (NTest and NTrain + NTest <= N_total_samp) else int(0.2 * N_total_samp)
    )

    if rand:
        indices = np.random.permutation(N_total_samp)
    else:
        indices = np.arange(N_total_samp)

    if rangeTest is None:
        # Ordinary random (or sequential) selection of the test set.
        test_indices = indices[:N_test_samp]
    else:
        # Constrained selection: walk the permutation and keep only
        # those indices that lie inside the whitelist `rangeTest`,
        # until we have accumulated N_test_samp test samples.
        i = 0
        test_indices = []
        while len(test_indices) < N_test_samp:
            if indices[i] in rangeTest:
                test_indices.append(indices[i])
            i = i + 1
    test_indices = np.array(test_indices)
    # The training pool is whatever remains in the first
    # (N_train_samp + N_test_samp) positions after removing the test
    # indices. `setdiff1d` also deduplicates.
    train_indices = np.setdiff1d(indices[: (N_train_samp + N_test_samp)], test_indices)

    train_images = data_images[train_indices]
    train_labels = data_labels[train_indices]
    test_images = data_images[test_indices]
    test_labels = data_labels[test_indices]

    if returnIndices:
        return (
            train_images,
            train_labels,
            test_images,
            test_labels,
            train_indices,
            test_indices,
        )
    else:
        return (train_images, train_labels, test_images, test_labels)


def standardize_data(data, train_indices=None, zero_center=True):
    """Normalize feature matrix using statistics computed on the training set.

    The manuscript (Methods, "Downstream classifiers, optimization,
    and reporting") uses two different normalization schemes:

    * `zero_center=True` (standardization): subtract the training-set
      mean and divide by the training-set standard deviation, so each
      feature has zero mean and unit variance over the training set.
      This is applied to PCA, low-pass-filtered, and coarse-grained
      features for logistic regression, and to all methods for the
      neural-network classifier.
    * `zero_center=False` (global rescaling): divide the whole array by a
      single positive scalar based on the total absolute magnitude of the
      training set, preserving relative amplitudes and sign pattern while
      avoiding zero-centering.. This is applied to eigentask features for
      logistic regression.

    Note that `train_std[train_std == 0] = 1` guards against degenerate
    features that are constant across the training set.
    """
    data = np.asarray(data)
    data = data.astype(np.result_type(data.dtype, np.float64), copy=False)
    if train_indices is None:
        train_indices = np.arange(len(data))
    if zero_center:
        # Feature-wise standardization using training statistics.
        train_mean = data[train_indices].mean(axis=0)
        train_std = np.std(data[train_indices], axis=0)
        train_std[train_std == 0] = 1
        return (data - train_mean) / train_std
    else:
        # Global rescaling: a single scalar scales the whole array so
        # that the mean absolute training-set entry becomes unity.
        return data * len(train_indices) / np.sum(np.abs(data[train_indices]))


# =============================================================================
# Data-agnostic (fixed) noise-mitigation transforms: coarse graining and FFT
# =============================================================================


def downsample_data(data, L=None, dim="2D"):
    """Spatial coarse graining by linear-interpolation downsampling.

    Implements the W_CG transform described in Methods under
    "Fourier-domain low-pass filtering and spatial coarse graining":
    a K = L x L (2D) or K = L (1D) feature array is resized to K_r =
    L_r x L_r or K_r = L_r respectively, using `scipy.ndimage.zoom`
    with bilinear interpolation (order=1). This is a fixed,
    data-agnostic transform (Fig. 1(b)).

    Parameters
    ----------
    data : np.ndarray
        Shape (N, L_max, L_max) for 2D images (e.g. EMCCD pixel
        arrays) or (N, L_max) for 1D sequences (e.g. SPDNN detector
        outputs).
    L : int, optional
        Target side length L_r. Capped at L_max; if None, no
        downsampling is performed.
    dim : {"1D", "2D"}
        Dimensionality of the input.

    Returns
    -------
    np.ndarray of shape (N, L_r^2) or (N, L_r).
    """
    if dim == "2D":
        L_max = np.shape(data)[1]  # data assumed N_total_samp x L_max x L_max
        L_samp = L if (L and L <= L_max) else L_max
        # `zoom` performs bilinear interpolation in 2D; the final
        # reshape flattens each coarse-grained image to length L_r^2.
        return np.array([zoom(im, L_samp / L_max, order=1) for im in data]).reshape(
            -1, int(L_samp**2)
        )

    elif dim == "1D":
        L_max = np.shape(data)[1]  # data assumed N_total_samp x L_max for 1D
        L_samp = L if (L and L <= L_max) else L_max
        return np.array([zoom(seq, L_samp / L_max, order=1) for seq in data]).reshape(
            -1, int(L_samp)
        )

    else:
        raise ValueError("dim should be '1D' or '2D'")


# =============================================================================
# Eigentask solver
#
# This is the central construction of the manuscript. Given shot-averaged
# training features X_bar(u) collected at the maximum sampling budget Smax,
# we build a basis of K eigentasks r^(k) (k = 1, ..., K) by solving the
# generalized eigenvalue problem
#
#     V r^(k) = (1 / alpha_k^2) G r^(k),                          Eq. (3)
#
# where V is the (sample-averaged) covariance matrix of the readout noise
# and G is the Gram matrix of the mean input-to-readout map u -> x(u).
# The eigenvalues alpha_k^2 are the signal-to-noise ratios of the
# corresponding eigentask features, and we order them in descending order
# so that r^(1) is the most signal-dominated direction and r^(K) the most
# noise-dominated.
#
# In practice (Methods, "Estimation of eigentasks..."), only finite-sample
# estimates V_tilde and G_tilde are available:
#
#     V_tilde = (1 / Ntrain (Smax - 1)) sum_n sum_s (X^(s)_k - X_bar_k) (X^(s)_k' - X_bar_k')
#                                                                  Eq. (7)
#     G_tilde = (1 / Ntrain) sum_n X_bar_k(u_n) X_bar_k'(u_n)        Eq. (8)
#
# which are related to the population V, G by
#
#     V = V_tilde,    G = G_tilde - (1 / Smax) V_tilde,              Eq. (9)
#
# and the associated SNRs transform as alpha_k^2 = alpha_tilde_k^2 - 1/Smax.
#
# The `eigentask_solver` function below performs the finite-sample eigenvalue
# problem in the (V_tilde, G_tilde) convention and returns the alpha_tilde_k^2
# eigenvalues directly (i.e. WITHOUT applying the 1/Smax correction of
# Eq. (9); see the note in the returned `nsr_nocorrection`). The finite-Smax
# correction should be applied by the caller when the absolute SNR values
# are needed; it is not required to construct the noise-mitigation transform
# W_EGT = (r^(1) ... r^(K_r)), since the eigenvectors r^(k) are invariant
# under the shift alpha_tilde^2 = alpha^2 + 1/Smax.
# =============================================================================


def eigentask_solver(data, V=None):
    """Construct the eigentask basis and project data onto it.

    This implements the finite-sample version of the generalized
    eigenvalue problem in manuscript Eq. (3),
        V_tilde r_tilde = (1 / alpha_tilde^2) G_tilde r_tilde,
    where:
        * `data`  corresponds to an (N x K) matrix of Smax-shot-averaged
                  features X_bar(u), stacked over the Ntrain training
                  inputs; see Methods Eq. (8) for G_tilde.
        * `V`     is the empirical noise covariance matrix V_tilde of
                  shape (K, K), estimated from single-shot residuals
                  (Methods Eq. (7)). If `V is None`, a diagonal
                  Poissonian surrogate V_ii = E[X_i] is used, which is
                  adequate when the readout is photon-counting-limited
                  (e.g. high-gain EMCCD, single-photon detectors) and
                  the shot noise dominates.

    Parameters
    ----------
    data : np.ndarray
        Shape (N, K) or (N, L, L): Smax-shot-averaged training
        features. Higher-dimensional feature shapes are flattened to
        length K before solving.
    V : np.ndarray, optional
        Empirical noise covariance matrix V_tilde of shape (K, K). If
        None, a Poissonian diagonal fallback diag(mean(X)) is used.

    Returns
    -------
    data_orig_basis : np.ndarray, shape (N, K)
        The input features, flattened to 2-D; returned for convenience.
    data_eigen_basis : np.ndarray, shape (N, K_indep)
        Projection of `data` onto the eigentask basis, i.e. the
        eigentask features Y_EGT(u) = W_EGT X_bar(u) with bias b = 0
        (Methods, "Estimation of eigentasks...").
        Columns are ordered by descending alpha_tilde^2 (highest SNR
        first).
    nsr_nocorrection : np.ndarray, shape (K_indep,)
        The (uncorrected) eigenvalues alpha_tilde_k^2 of the
        generalized eigenvalue problem, in descending order.
    r_train.T : np.ndarray, shape (K_indep, K)
        The eigentask masks; row k is the mask r^(k) used to produce
        the k-th eigentask feature as r^(k).T @ X_bar(u) (cf. Fig. 2(b)).
        These are the rows of the transformation matrix W_EGT
        (manuscript Eq. (2)).
    """

    # --- Assemble the Gram matrix G_tilde from training features --------
    # Stack samples as columns: XMat has shape (K, N). Each column is a
    # single input's shot-averaged feature vector X_bar(u_n).
    XMat = data.reshape(len(data), -1).T
    # The generalized eigenvalue problem only has K_indep = min(K, N)
    # non-trivial eigenpairs; all higher modes are artefacts of
    # finite-sample rank deficiency and are discarded below.
    K_indep = np.min(XMat.shape)
    # Empirical Gram matrix G_tilde (Methods Eq. (8)):
    #     G_tilde_{k k'} = (1/N) * sum_n X_bar_k(u_n) X_bar_{k'}(u_n)
    # Note that this uses shot-averaged features, so strictly speaking
    # it equals the population Gram matrix G plus a (1/Smax) V bias
    # (Methods Eq. (9)). In the noise-dominated regimes targeted by
    # this work the bias is absorbed into the eigenvalue shift and
    # does not affect the eigenvectors r^(k) that define W_EGT.
    G = XMat @ XMat.T / XMat.shape[1]

    # Fallback covariance: for photon-counting sensors (EMCCD in the
    # high-gain limit, single-photon detectors), the per-channel noise
    # variance is proportional to the mean signal, yielding a diagonal
    # V with V_kk = E[X_k]. This is only used if the caller does not
    # supply an empirical V_tilde.
    VPoisson = np.diag(np.mean(XMat, axis=1))

    D = V if V is not None else VPoisson

    # --- Solve the generalized eigenvalue problem V r = (1/alpha^2) G r -
    # Recast as an ordinary eigenvalue problem for V^{-1} G:
    #     V^{-1} G r = alpha^2 r
    # The pseudoinverse is used because V may be rank-deficient in the
    # low-shot / low-flux regime (e.g. many camera pixels carry only
    # dark counts and contribute nothing to the covariance).
    RandomWalk = la.pinv(D) @ G
    s, r = la.eig(RandomWalk)
    # Sort eigenpairs by descending SNR so that column k of `r_train`
    # is the k-th eigentask r^(k), with r^(1) the most signal-dominated.
    idx = s.argsort()[::-1]
    s_train = np.real(s[idx])
    r_train = np.real(r[:, idx])

    # `nsr_nocorrection` holds the uncorrected eigenvalues alpha_tilde^2.
    # To obtain the population SNR, apply alpha^2 = alpha_tilde^2 - 1/Smax.
    nsr_nocorrection = s_train
    data_orig_basis = XMat.T
    # Project shot-averaged features onto the eigentask basis: this is
    # the Y_EGT(u) = W_EGT X_bar(u) + b feature vector used downstream,
    # with the bias b = 0 convention.
    data_eigen_basis = XMat.T @ r_train
    # Optional renormalization (commented out) that rescales each input's
    # eigentask features to preserve the total energy of the raw features.
    # Not used in the manuscript; kept here for reference.
    # data_eigen_basis = np.array([data_eigen_basis[i] / np.sum(data_eigen_basis[i]) * np.sum(data_orig_basis[i]) for i in range(len(data_orig_basis))])

    return (
        data_orig_basis,
        data_eigen_basis[:, :K_indep],
        nsr_nocorrection[:K_indep],
        r_train.T[:K_indep],
    )


# =============================================================================
# PCA solver
# =============================================================================


def pca_solver(data, zero_center=True):
    """Principal component analysis of the feature matrix.

    Implements the PCA transform described in Methods ("Estimation
    of principal components..."). The principal components are the
    eigenvectors v^(k) of the empirical covariance matrix C (Methods
    Eq. (10)), ordered by descending variance eigenvalues sigma_1^2 >=
    sigma_2^2 >= ... >= sigma_K^2. The PCA transformation matrix is
        W_PCA = (v^(1) ... v^(K_r)).T
    applied to the zero-centered features, yielding
        Y_PCA(u) = W_PCA (X_bar(u) - X_bar_mean).

    Parameters
    ----------
    data : np.ndarray, shape (N, K) or (N, L, L)
        Shot-averaged training features. Flattened to (N, K) inside.
    zero_center : bool, default True
        * True: use `sklearn.decomposition.PCA`, which subtracts the
          training-set mean (standard PCA; corresponds to the
          manuscript's Eq. (10) formulation).
        * False: diagonalize the raw Gram-like matrix M = X.T @ X
          without zero-centering, yielding a non-standard "PCA" whose
          leading direction is typically the overall intensity.
          Included here for ablation studies.

    Returns
    -------
    data_pca : np.ndarray, shape (N, K_indep)
        Training features projected onto the principal components, in
        descending order of explained variance.
    components : np.ndarray, shape (K_indep, K)
        Rows are the principal components v^(k) (the PCA
        transformation matrix W_PCA).
    """
    if zero_center == True:
        # Standard sklearn PCA: centers the data internally using the
        # training mean. The returned `pca.components_` matrix is
        # already ordered by descending explained variance.
        pca = PCA()
        data_pca = pca.fit_transform(data.reshape(len(data), -1))
        return data_pca, pca.components_
    else:
        # Non-centered variant: eigenvectors of X.T @ X rather than of
        # the covariance matrix. The leading mode then captures the
        # mean direction; subsequent modes differ from standard PCA.
        data = data.reshape(len(data), -1)
        M = data.T @ data
        eigs, pc = np.linalg.eig(M)
        idx = np.argsort(eigs)[::-1]
        pc = (pc.T[idx]).T
        return data @ pc, pc


# =============================================================================
# Fourier-domain low-pass filtering
# =============================================================================


def fft(data):
    """Centered 1D or 2D FFT over a batch of features.

    Applies `fftshift -> fft -> fftshift` so that the zero-frequency
    component lands at the centre of the output array. This is the
    convention expected by `low_pass` below.

    Parameters
    ----------
    data : np.ndarray
        Shape (N, L, L) for 2D (EMCCD images) or (N, L) for 1D (SPDNN
        detector outputs).

    Returns
    -------
    np.ndarray of the same shape as `data`, complex-valued, with DC
    at the centre of the last one or two axes.
    """
    if len(data.shape) == 3:
        return np.array(
            [np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im))) for im in data]
        )
    elif len(data.shape) == 2:
        return np.array(
            [np.fft.fftshift(np.fft.fft(np.fft.fftshift(im))) for im in data]
        )


def low_pass(FFTdata, L=None, type="real"):
    """Fourier-domain low-pass filter: keep the low-frequency real/imag parts.

    Implements the low-pass-filtering transform used as one of the
    four noise-mitigation methods in the manuscript (Methods,
    "Fourier-domain low-pass filtering and spatial coarse graining").
    The input features are Fourier-transformed (see `fft` above); then
    for the retained low-frequency block of side L_r = sqrt(K_r) we
    keep the real and imaginary parts of the complex coefficients as
    separate real-valued features. Because the time-domain features
    are real, the negative-frequency half of the spectrum is redundant
    with the positive-frequency half (f_{i,j} = f*_{-i,-j}), so only
    half of the complex coefficients are retained.

    Parameters
    ----------
    FFTdata : np.ndarray
        Centered Fourier features; shape (N, L, L) (2D) or (N, L)
        (1D), complex-valued.
    L : int, optional
        Retained low-pass window side length L_r. Capped at L_max.
    type : str
        Only "real" is implemented. Included for future extensions.

    Returns
    -------
    np.ndarray of shape (N, K_r), real-valued: real and imaginary
    parts of the retained low-frequency coefficients, concatenated
    along the feature axis.
    """
    L_max = np.shape(FFTdata)[1]
    L_samp = L if (L and L <= L_max) else L_max
    if len(FFTdata.shape) == 3:
        if type == "real":
            # 2D case: retain a centered strip of height L_samp in the
            # vertical (i) direction and the positive-frequency half of
            # the horizontal (j) direction. The precise index arithmetic
            # ensures Hermitian symmetry is exploited without double
            # counting.
            freqs_real = FFTdata[
                :,
                L_max // 2 - L_samp // 2 : L_max // 2 - L_samp // 2 + L_samp,
                L_max // 2 : L_max // 2 + (L_samp + 1) // 2,
            ].real

            freqs_imag = FFTdata[
                :,
                L_max // 2 - L_samp // 2 : L_max // 2 - L_samp // 2 + L_samp,
                L_max // 2 : L_max // 2 + L_samp // 2,
            ].imag

            return np.concatenate(
                (
                    freqs_real.reshape(len(FFTdata), -1),
                    freqs_imag.reshape(len(FFTdata), -1),
                ),
                axis=1,
            )
    elif len(FFTdata.shape) == 2:
        if type == "real":
            # 1D case (used for SPDNN features): retain only the
            # non-negative-frequency half of the spectrum; the negative-
            # frequency half is the complex conjugate.
            freqs_real = FFTdata[:, L_max // 2 : L_max // 2 + L_samp // 2 + 1].real

            freqs_imag = FFTdata[:, L_max // 2 : L_max // 2 - L_samp // 2 + L_samp].imag

            return np.concatenate(
                (
                    freqs_real.reshape(len(FFTdata), -1),
                    freqs_imag.reshape(len(FFTdata), -1),
                ),
                axis=1,
            )


# =============================================================================
# Closed-form linear regression baseline
# =============================================================================


def LinearRegression(
    data_images,
    data_labels,
    K=None,
    NTrain=None,
    NVal=None,
    NTest=None,
    rand=True,
    rangeTest=None,
    bias=True,
    returnW=False,
):
    """Closed-form one-vs-all linear-regression classifier baseline.

    Trains a linear map X -> Y_onehot by solving the ordinary least-
    squares problem `XTrain W = YTrain` (via `numpy.linalg.lstsq`)
    and reports the accuracy obtained by taking argmax over the
    predicted one-hot targets. This is not the main classifier used
    in the manuscript (which is a multinomial logistic regression
    trained by AdamW, see `LogisticTrain`), but it provides a fast,
    deterministic baseline useful for sanity checks and for hyperparameter
    sweeps over K_r.

    Parameters
    ----------
    data_images, data_labels : np.ndarray
        Features and integer (or already one-hot) labels.
    K : int, optional
        Use only the first K features of each sample (i.e. K_r).
    NTrain, NVal, NTest : int, optional
        Sizes of the train, validation, and test sets.
    rand, rangeTest : see `train_test_generator`.
    bias : bool, default True
        If True, append a constant 1 feature to X so the least-squares
        solution absorbs a bias term.
    returnW : bool
        If True, also return the fitted weight matrix W.

    Returns
    -------
    Accuracies (train, [val], test), and optionally `W`.
    """
    # Convert integer class labels to one-hot encoding; passthrough if
    # the caller already supplied one-hot (or soft) targets.
    n_outs = len(np.unique(data_labels))
    data_labels = np.asarray(data_labels)
    if data_labels.ndim == 1 and np.issubdtype(data_labels.dtype, np.integer):
        labels_onehot = np.eye(n_outs, dtype=np.float64)[data_labels]
    else:
        labels_onehot = np.asarray(data_labels)

    X = np.array(data_images).reshape(len(data_images), -1)
    K_samp = X.shape[1] if (K is None or K > X.shape[1]) else K
    X = X[:, :K_samp]

    if bias:
        # Append a constant column of ones so the least-squares fit
        # implicitly estimates an additive bias.
        X = np.concatenate((X, np.ones((len(X), 1))), axis=1)

    Y = labels_onehot

    # Same train / (val) / test partitioning convention as the
    # stochastic-gradient training loops below.
    pool_train_X, pool_train_Y, XTest, YTest = train_test_generator(
        X,
        Y,
        NTrain=(NTrain + NVal if (NVal is not None and NVal > 0) else NTrain),
        NTest=NTest,
        rand=rand,
        rangeTest=rangeTest,
    )

    # Carve off a validation subset from the front of the training
    # pool when requested.
    if NVal is not None and NVal > 0:
        if NVal >= len(pool_train_X):
            raise ValueError(
                f"NVal={NVal} is too large for the available training pool of size {len(pool_train_X)}."
            )
        XVal = pool_train_X[:NVal]
        YVal = pool_train_Y[:NVal]
        XTrain = pool_train_X[NVal:]
        YTrain = pool_train_Y[NVal:]
    else:
        XTrain = pool_train_X
        YTrain = pool_train_Y
        XVal = None
        YVal = None

    if len(XTrain) == 0:
        raise ValueError("Training set is empty in LinearRegression.")
    if len(XTest) == 0:
        raise ValueError("Test set is empty in LinearRegression.")

    # Closed-form least-squares solution. Gracefully handle SVD
    # failures (rare, but possible with very rank-deficient feature
    # matrices) by returning zero accuracies.
    try:
        W, *_ = np.linalg.lstsq(XTrain, YTrain, rcond=None)
    except np.linalg.LinAlgError as e:
        print("Caught SVD error:", e)
        if returnW:
            if NVal is not None and NVal > 0:
                return 0, 0, 0, 0
            else:
                return 0, 0, 0
        else:
            if NVal is not None and NVal > 0:
                return 0, 0, 0
            else:
                return 0, 0

    # Accuracies: argmax over predicted one-hot targets.
    acc_train = (
        np.sum(np.argmax(XTrain @ W, axis=1) == np.argmax(YTrain, axis=1))
        / len(XTrain)
        * 100
    )

    acc_test = (
        np.sum(np.argmax(XTest @ W, axis=1) == np.argmax(YTest, axis=1))
        / len(XTest)
        * 100
    )

    if NVal is not None and NVal > 0:
        if len(XVal) == 0:
            raise ValueError("Validation set is empty in LinearRegression.")
        acc_val = (
            np.sum(np.argmax(XVal @ W, axis=1) == np.argmax(YVal, axis=1))
            / len(XVal)
            * 100
        )
        if returnW:
            return acc_train, acc_val, acc_test, W
        else:
            return acc_train, acc_val, acc_test
    else:
        if returnW:
            return acc_train, acc_test, W
        else:
            return acc_train, acc_test


# =============================================================================
# Logistic-regression training loop
# =============================================================================


def LogisticTrain(
    data_set,
    data_labels,
    dev=None,
    init_lr=1e-3,
    Epochs=300,
    manual_seed=None,
    K=None,
    NTrain=None,
    NVal=None,
    NTest=None,
    verbose=False,
    rand=True,
    justTrain=False,
    rangeTest=None,
    runningAccuracy=False,
    savepath=None,
    batch_size=100,
    lr_scheduler="step_decay",
):
    """Train a multinomial logistic-regression classifier on K features.

    Reproduces the logistic-regression training protocol reported in
    Methods ("Downstream classifiers, optimization, and reporting"):
    AdamW optimizer with betas = (0.9, 0.999), zero weight decay,
    batch size 100, and one of two learning-rate schedulers
    (`StepLR` with factor 0.4 every 50 epochs for MPEG-7;
    `ReduceLROnPlateau` with factor 0.5, patience 10, and min_lr 1e-5
    for MNIST). Cross-entropy loss is used throughout.

    The function optionally returns the per-epoch training,
    validation, and test accuracies, which are needed for the
    "validation-selected model" reporting convention: for each
    method and each (S, K_r) setting the reported test accuracy is
    taken at the epoch that maximizes validation accuracy, with
    ties broken by preferring smaller K_r and earlier epochs.

    Parameters
    ----------
    data_set, data_labels : np.ndarray
        Feature array and integer label array. The features should
        already be normalized (via `standardize_data`) before being
        passed in.
    dev : str
        PyTorch device string (e.g. "cuda:0", "cpu").
    init_lr : float
        Initial learning rate. In the manuscript this is 1e-3 for
        standardized features, and 0.5 for the non-zero-centered
        eigentask features.
    Epochs : int
        Number of training epochs (300 in the manuscript).
    manual_seed : int, optional
        Random seed for reproducibility; passed to `set_seed`.
    K : int, optional
        Number of features K_r to use (leading columns of `data_set`).
    NTrain, NVal, NTest : int, optional
        Train / val / test set sizes.
    verbose, justTrain : bool
        Control the progress-bar and the ten-equally-spaced print
        messages during training.
    rand, rangeTest : see `train_test_generator`.
    runningAccuracy : bool
        If True, record accuracies at every epoch (required for
        validation-based epoch selection).
    savepath : str, optional
        If provided, save the final `state_dict` at this path.
    batch_size : int
        Mini-batch size (100 in the manuscript).
    lr_scheduler : {"step_decay", "plateau_reduce"}
        Learning-rate scheduler choice; see above.
    """
    device = _resolve_device(dev)
    iscomplex = True if (np.max(data_set.imag) > 1e-15) else False

    # Load Data
    if manual_seed is not None:
        set_seed(manual_seed)

    N_total_samp = len(data_set)
    N_train_samp = (
        NTrain
        if (NTrain and NTrain + NTest <= N_total_samp)
        else int(0.8 * N_total_samp)
    )
    N_test_samp = (
        NTest if (NTest and NTrain + NTest <= N_total_samp) else int(0.2 * N_total_samp)
    )
    K_samp = len(data_set[0]) if (K is None or K > len(data_set[0])) else K
    # print(K_samp)

    # Initial split produces train-pool + test. A subsequent step below
    # carves NVal samples off the front of the train pool to form the
    # validation set, reproducing the three-way train / val / test
    # partition used throughout the manuscript.
    train_images, train_labels, test_images, test_labels = train_test_generator(
        data_set,
        data_labels,
        NTrain=(NTrain + NVal if (NVal is not None and NVal > 0) else NTrain),
        NTest=NTest,
        rand=rand,
        rangeTest=rangeTest,
    )
    if NVal is not None and NVal > 0:
        val_images = train_images[:NVal]
        val_labels = train_labels[:NVal]
        train_images = train_images[NVal:]
        train_labels = train_labels[NVal:]
        N_val_samp = NVal
        N_train_samp = NTrain
    else:
        # Without a dedicated validation set we fall back to using the
        # test set as validation. This is only appropriate for
        # sanity-check runs; the main experiments always pass NVal>0.
        val_images = test_images
        val_labels = test_labels
        N_val_samp = N_test_samp

    n_outs = len(np.unique(train_labels))

    # Build PyTorch datasets, each cropped to the first K_samp features
    # (i.e. K_r features for the current sweep point).
    train_data = Torch_Dataset(train_images[:, :K_samp], train_labels, dev=device)
    val_data = Torch_Dataset(val_images[:, :K_samp], val_labels, dev=device)
    test_data = Torch_Dataset(test_images[:, :K_samp], test_labels, dev=device)
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataset = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model_lin = LR_classifier(K_samp, n_outs=n_outs, iscomplex=iscomplex).to(device)

    learn_rate = init_lr
    criterion = nn.CrossEntropyLoss()
    # AdamW with zero weight decay reproduces the optimizer used in
    # the manuscript (Methods). Using AdamW (rather than vanilla Adam)
    # leaves the door open for a non-zero weight-decay ablation.
    optimizer = torch.optim.AdamW(
        model_lin.parameters(), lr=learn_rate, betas=(0.9, 0.999), weight_decay=0.0
    )
    if lr_scheduler == "step_decay":
        # StepLR: multiplicative factor 0.4 every 50 epochs
        # (MPEG-7 protocol).
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.4)
    elif lr_scheduler == "plateau_reduce":
        # ReduceLROnPlateau: factor 0.5, patience 10 epochs, floor
        # 1e-5 (MNIST lens-front-end and SPDNN protocol).
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
        )
    # optimizer.lr = init_lr

    # optimizer = torch.optim.SGD(model_lin.parameters(), lr=learn_rate, weight_decay = 0.0)

    # if (device != "cpu") :
    #     with torch.cuda.device(dev):
    #         torch.cuda.empty_cache()

    loss_train, loss_val, loss_test = [], [], []
    acc_train, acc_val, acc_test = [], [], []

    for epoch in (
        range(Epochs) if (verbose or justTrain) else tqdm(range(Epochs))
    ):  # loop over the dataset multiple times
        train_running_loss = 0.0
        train_acc = 0.0
        model_lin.train()
        for i, data in enumerate(train_dataset):
            # --- Standard PyTorch training step -----------------------
            optimizer.zero_grad()
            inputs, labels = data
            outputs = model_lin(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Accumulate running loss / accuracy only when we actually
            # need them (every epoch in runningAccuracy mode, or only
            # on the last epoch otherwise).
            if runningAccuracy or epoch == Epochs - 1:
                train_running_loss += loss.detach().item()
                train_acc += get_accuracy(outputs, labels)
        model_lin.eval()
        if runningAccuracy:
            acc_train.append(train_acc / N_train_samp)
            loss_train.append(train_running_loss / len(train_dataset))
            if verbose and (epoch % (Epochs / 10) == 0):
                print(
                    "Progress:  %d | Loss: %.4f | Accuracy: %.2f"
                    % (
                        100 * epoch / Epochs,
                        train_running_loss / (N_train_samp),
                        train_acc / N_train_samp,
                    )
                )
        t_loss = 0.0
        test_acc = 0.0
        v_loss = 0.0
        val_acc = 0.0
        if runningAccuracy or epoch == Epochs - 1:
            # --- Validation pass ------------------------------------
            with torch.no_grad():
                for vdata in val_dataset:
                    v_inputs, v_labels = vdata
                    v_outputs = model_lin(v_inputs)
                    v_loss += criterion(v_outputs, v_labels).detach().item()
                    val_acc += get_accuracy(v_outputs, v_labels)
                acc_val.append(val_acc / N_val_samp)
                loss_val.append(v_loss / len(val_dataset))

            # --- Test pass ------------------------------------------
            with torch.no_grad():
                if NVal is not None and NVal > 0:
                    for p, tdata in enumerate(test_dataset):
                        t_inputs, t_labels = tdata
                        t_outputs = model_lin(t_inputs)
                        t_loss += criterion(t_outputs, t_labels).detach().item()
                        test_acc += get_accuracy(t_outputs, t_labels)
                else:
                    # In the no-validation fallback path we already
                    # evaluated on the test set as "val" above; reuse
                    # the numbers here.
                    t_loss = v_loss
                    test_acc = val_acc
                acc_test.append(test_acc / N_test_samp)
                loss_test.append(t_loss / len(test_dataset))
        if verbose or epoch == Epochs - 1:
            if verbose and (epoch % (Epochs / 10) == 0):
                print(
                    "Test Loss: %.6f | Test Accuracy: %.4f"
                    % (t_loss / (N_test_samp), test_acc / N_test_samp)
                )
        if lr_scheduler == "step_decay":
            scheduler.step()
        elif lr_scheduler == "plateau_reduce":
            scheduler.step(v_loss / len(val_dataset))
    if savepath is not None:
        torch.save(model_lin.state_dict(), savepath)

    # Return shape is overloaded to remain backward-compatible:
    #   runningAccuracy = True  -> also return per-epoch curves needed
    #                              for validation-based epoch selection
    #   NVal > 0                -> also return validation accuracy
    if runningAccuracy:
        if NVal is not None and NVal > 0:
            return (
                (
                    train_acc / N_train_samp,
                    val_acc / N_val_samp,
                    test_acc / N_test_samp,
                ),
                (
                    np.array(acc_train),
                    np.array(acc_val),
                    np.array(acc_test),
                ),
                (np.array(loss_train), np.array(loss_val), np.array(loss_test)),
            )
        else:
            return (
                (train_acc / N_train_samp, test_acc / N_test_samp),
                (
                    np.array(acc_train),
                    np.array(acc_test),
                ),
                (np.array(loss_train), np.array(loss_test)),
            )
    else:
        if NVal is not None and NVal > 0:
            return (
                train_acc / N_train_samp,
                val_acc / N_val_samp,
                test_acc / N_test_samp,
            )
        else:
            return (train_acc / N_train_samp, test_acc / N_test_samp)


# =============================================================================
# Multilayer perceptron classifier F{.}
# =============================================================================


class DNN(nn.Module):
    """Configurable fully connected multilayer perceptron.

    Stack of Linear -> (BatchNorm) -> nonlinear-activation blocks,
    terminated by a linear output layer. The manuscript uses a single
    hidden layer of 400 ReLU units, no batch normalization
    (Methods, "Downstream classifiers, optimization, and reporting").

    The module is instantiated with input dimension K_r and output
    dimension C (number of classes).
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        Nunits=None,
        batchnorm=False,
        nlaf="relu",
        final_sigmoid=False,
        **kwargs,
    ):
        """
        Defines configurable deep neural network with fully connected layers and a choice of
        nonlinear activation functions.

        Args:
            input_dim (int): dimension of input layer
            output_dim (int): dimension of output layer
            Nunits (list of int): dimensions of hidden layers. The
                manuscript uses [400] for MNIST and MPEG-7 back ends,
                and [400] for SPDNN back ends in Supplementary Note 4 B.
            batchnorm (bool): determines whether to use batchnorm between each hidden layer.
                The order in which batchnorm is applied is:
                fully connected layer - batchnorm - nonlinear activation function
            nlaf (string): determines the nonlinear activation function. Choices:
                'relu', 'tanh', 'sigmoid'
        """
        super(DNN, self).__init__()

        if Nunits is None:
            # Two-hidden-layer default; not used by the manuscript's
            # single-hidden-layer experiments.
            Nunits = [100, 100]
        else:
            Nunits = list(Nunits)
        self.batchnorm = batchnorm
        # Select nonlinear activation function.
        if nlaf == "relu":
            self.nlaf = torch.relu
        elif nlaf == "tanh":
            self.nlaf = torch.tanh
        elif nlaf == "sigmoid":
            self.nlaf = torch.sigmoid

        # Prepend the input dimension so consecutive layer widths can
        # be read off pair-by-pair.
        Nunits.insert(0, input_dim)

        # Build one Linear per hidden-layer transition; the final
        # output layer (mapping the last hidden width to the number
        # of classes) is stored separately.
        self.layers = nn.ModuleList([])
        for i in range(len(Nunits) - 1):
            self.layers.append(nn.Linear(Nunits[i], Nunits[i + 1]))
        self.outputlayer = nn.Linear(Nunits[-1], output_dim)

        if batchnorm:
            self.batchnorms = nn.ModuleList([])
            for i in range(len(Nunits) - 1):
                self.batchnorms.append(nn.BatchNorm1d(Nunits[i + 1]))

    def forward(self, x):
        """
        Performs the forward pass through the network.

        Args:
            x (float tensor): inputs of dimension [batch_size, input_dim]
        """

        # Apply each hidden block in sequence: Linear, optional
        # BatchNorm, then nonlinearity. The final output layer
        # produces logits (no activation) that are passed to
        # CrossEntropyLoss externally.
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.batchnorm:
                x = self.batchnorms[i](x)
            x = self.nlaf(x)

        return self.outputlayer(x)


def DNNTrain(
    data_set,
    data_labels,
    modelargs,
    dev=None,
    init_lr=1e-3,
    Epochs=300,
    manual_seed=None,
    K=None,
    NTrain=None,
    NVal=None,
    NTest=None,
    verbose=False,
    rand=True,
    justTrain=False,
    rangeTest=None,
    runningAccuracy=False,
    batch_size=100,
    Model=DNN,
    savepath=None,
    lr_scheduler="step_decay",
):
    """Train an MLP classifier on K features.

    Structurally identical to `LogisticTrain` except that the model is
    an instance of `Model` (default: `DNN`) configured by `modelargs`
    instead of a single linear layer. Hyperparameters (AdamW, betas,
    batch size, schedulers) match the manuscript's Methods section.

    Parameters
    ----------
    modelargs : dict
        Keyword arguments forwarded to the model constructor, e.g.
        `{"Nunits": [400], "batchnorm": False, "nlaf": "relu"}`
        to reproduce the 400-unit single-hidden-layer ReLU MLP used
        in the manuscript.

    Other parameters: see `LogisticTrain`.
    """
    device = _resolve_device(dev)
    iscomplex = True if (np.max(data_set.imag) > 1e-15) else False

    if manual_seed is not None:
        set_seed(manual_seed)

    N_total_samp = len(data_set)
    N_train_samp = (
        NTrain
        if (NTrain and NTrain + NTest <= N_total_samp)
        else int(0.8 * N_total_samp)
    )
    N_test_samp = (
        NTest if (NTest and NTrain + NTest <= N_total_samp) else int(0.2 * N_total_samp)
    )
    K_samp = len(data_set[0]) if (K is None or K > len(data_set[0])) else K

    # Same three-way train / val / test split logic as LogisticTrain.
    train_images, train_labels, test_images, test_labels = train_test_generator(
        data_set,
        data_labels,
        NTrain=(NTrain + NVal if (NVal is not None and NVal > 0) else NTrain),
        NTest=NTest,
        rand=rand,
        rangeTest=rangeTest,
    )
    if NVal is not None and NVal > 0:
        val_images = train_images[:NVal]
        val_labels = train_labels[:NVal]
        train_images = train_images[NVal:]
        train_labels = train_labels[NVal:]
        N_val_samp = NVal
        N_train_samp = NTrain
    else:
        val_images = test_images
        val_labels = test_labels
        N_val_samp = N_test_samp

    n_outs = len(np.unique(train_labels))

    train_data = Torch_Dataset(train_images[:, :K_samp], train_labels, dev=device)
    val_data = Torch_Dataset(val_images[:, :K_samp], val_labels, dev=device)
    test_data = Torch_Dataset(test_images[:, :K_samp], test_labels, dev=device)
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataset = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Instantiate the chosen model. For complex-valued inputs the
    # effective feature dimension doubles because real/imag parts are
    # concatenated (see `Torch_Dataset`).
    model = Model(K_samp * 2 if iscomplex else K_samp, n_outs, **modelargs).to(device)

    learn_rate = init_lr
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learn_rate, betas=(0.9, 0.999), weight_decay=0.0
    )
    if lr_scheduler == "step_decay":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.4)
    elif lr_scheduler == "plateau_reduce":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
        )

    loss_train, loss_val, loss_test = [], [], []
    acc_train, acc_val, acc_test = [], [], []

    for epoch in (
        range(Epochs) if (verbose or justTrain) else tqdm(range(Epochs))
    ):  # loop over the dataset multiple times
        train_running_loss = 0.0
        train_acc = 0.0
        model.train()
        for i, data in enumerate(train_dataset):
            # Standard gradient-descent step; identical to LogisticTrain.
            optimizer.zero_grad()
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if runningAccuracy or epoch == Epochs - 1:
                train_running_loss += loss.detach().item()
                train_acc += get_accuracy(outputs, labels)
        model.eval()
        if runningAccuracy:
            acc_train.append(train_acc / N_train_samp)
            loss_train.append(train_running_loss / len(train_dataset))
            if verbose and (epoch % (Epochs / 10) == 0):
                print(
                    "Progress:  %d | Loss: %.4f | Accuracy: %.2f"
                    % (
                        100 * epoch / Epochs,
                        train_running_loss / (N_train_samp),
                        train_acc / N_train_samp,
                    )
                )
        t_loss = 0.0
        test_acc = 0.0
        v_loss = 0.0
        val_acc = 0.0
        if runningAccuracy or epoch == Epochs - 1:
            # Validation pass (in eval mode: BatchNorm/Dropout are
            # frozen if present).
            with torch.no_grad():
                for vdata in val_dataset:
                    v_inputs, v_labels = vdata
                    v_outputs = model(v_inputs)
                    v_loss += criterion(v_outputs, v_labels).detach().item()
                    val_acc += get_accuracy(v_outputs, v_labels)
                acc_val.append(val_acc / N_val_samp)
                loss_val.append(v_loss / len(val_dataset))

            # Test pass.
            with torch.no_grad():
                if NVal is not None and NVal > 0:
                    for p, tdata in enumerate(test_dataset):
                        t_inputs, t_labels = tdata
                        t_outputs = model(t_inputs)
                        t_loss += criterion(t_outputs, t_labels).detach().item()
                        test_acc += get_accuracy(t_outputs, t_labels)
                else:
                    t_loss = v_loss
                    test_acc = val_acc
                acc_test.append(test_acc / N_test_samp)
                loss_test.append(t_loss / len(test_dataset))
        if verbose or epoch == Epochs - 1:
            if verbose and (epoch % (Epochs / 10) == 0):
                print(
                    "Test Loss: %.6f | Test Accuracy: %.4f"
                    % (t_loss / (N_test_samp), test_acc / N_test_samp)
                )
        if lr_scheduler == "step_decay":
            scheduler.step()
        elif lr_scheduler == "plateau_reduce":
            scheduler.step(v_loss / len(val_dataset))

    if savepath is not None:
        torch.save(model.state_dict(), savepath)

    # Same overloaded return shape as LogisticTrain: see there for
    # documentation.
    if runningAccuracy:
        if NVal is not None and NVal > 0:
            return (
                (
                    train_acc / N_train_samp,
                    val_acc / N_val_samp,
                    test_acc / N_test_samp,
                ),
                (
                    np.array(acc_train),
                    np.array(acc_val),
                    np.array(acc_test),
                ),
                (np.array(loss_train), np.array(loss_val), np.array(loss_test)),
            )
        else:
            return (
                (train_acc / N_train_samp, test_acc / N_test_samp),
                (
                    np.array(acc_train),
                    np.array(acc_test),
                ),
                (np.array(loss_train), np.array(loss_test)),
            )
    else:
        if NVal is not None and NVal > 0:
            return (
                train_acc / N_train_samp,
                val_acc / N_val_samp,
                test_acc / N_test_samp,
            )
        else:
            return (train_acc / N_train_samp, test_acc / N_test_samp)
