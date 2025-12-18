# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 13:32:51 2025

@author: 18307
"""
import os
import h5py
import numpy as np

import mne
import time

from utils import utils_feature_loading, utils_eeg_loading
def filter_eeg_seed(identifier, freq=200, save=False, verbose=True, apply_slf='slf', slf_kwargs=None):
    """
    Load, filter, and optionally save SEED dataset EEG data into frequency bands.

    Parameters:
    identifier (str): Identifier for the subject/session.
    freq (int): SEED: 200 Hz. DREAMER: 128 Hz.
    verbose (bool): If True, prints progress messages. Default is True.
    save (bool): If True, saves the filtered EEG data to disk. Default is False.

    Returns:
    dict:
        A dictionary where keys are frequency band names and values are the filtered MNE Raw objects.

    Raises:
    FileNotFoundError: If the SEED data file cannot be found.
    """
    # Load raw EEG data using the provided utility function
    eeg = utils_eeg_loading.read_and_parse_seed(identifier)
    
    # Construct the output folder path for filtered data
    base_path = os.path.abspath(os.path.join(os.getcwd(), "../../Research_Data/SEED/original eeg/Filtered_SLFed_EEG"))
    os.makedirs(base_path, exist_ok=True)
    
    # Filter the EEG data into different frequency bands
    filtered_eeg_dict = filter_eeg(eeg, freq=freq, verbose=verbose, apply_slf=apply_slf, slf_kwargs=slf_kwargs)
    
    # Save filtered EEG data if requested
    if save:
        for band, filtered_eeg in filtered_eeg_dict.items():
            path_file = os.path.join(base_path, f"{identifier}_{band}_eeg.fif")
            filtered_eeg.save(path_file, overwrite=True)
            if verbose:
                print(f"Saved {band} band filtered EEG to {path_file}")
    
    return filtered_eeg_dict

def filter_eeg(eeg, freq=128, verbose=False, apply_slf=False, slf_kwargs=None):
    """
    先（可选）做 SLF，再用 MNE 分段滤波到 Delta/Theta/Alpha/Beta/Gamma。
    """
    eeg = np.asarray(eeg, dtype=float)
    if eeg.ndim != 2:
        raise ValueError(f"eeg must be (n_channels, n_samples), got {eeg.shape}")

    if apply_slf == 'slf':
        slf_kwargs = slf_kwargs or {}
        eeg = surface_laplacian_filtering(
            eeg_data=eeg,
            sampling_rate=freq,   # SLF里不使用，但保留接口
            **slf_kwargs
        )
        if verbose:
            print("Applied SLF on broadband EEG (before bandpass).")
    elif apply_slf == 'ssd':        
        print('Test')

    info = mne.create_info(
        ch_names=[f"Ch{i}" for i in range(eeg.shape[0])],
        sfreq=freq,
        ch_types='eeg'
    )
    mne_eeg = mne.io.RawArray(eeg, info, verbose="ERROR")

    freq_bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 50),
    }

    band_filtered_eeg = {}
    for band, (low_freq, high_freq) in freq_bands.items():
        filtered_raw = mne_eeg.copy().filter(
            l_freq=low_freq, h_freq=high_freq,
            method="fir", phase="zero-double",
            verbose="ERROR"
        )
        band_filtered_eeg[band] = filtered_raw
        if verbose:
            print(f"{band} band filtered: {low_freq}–{high_freq} Hz")

    return band_filtered_eeg

def surface_laplacian_filtering(eeg_data, sampling_rate, dist=None, m=4, _lambda=1e-4):
    """
    Surface Laplacian Filtering (Spherical Spline, Perrin-style).

    Parameters
    ----------
    eeg_data : np.ndarray
        EEG data, shape (channels, time_samples).
    sampling_rate : float
        Sampling rate in Hz. (Not used by SLF; kept for pipeline compatibility.)
    m : int
        Spline order / smoothness parameter. Typical: 3~5 (often 4).
    _lambda : float
        Regularization parameter for numerical stability. Typical: 1e-5 ~ 1e-3.
    dataset : str
        Dataset name used by utils_feature_loading.read_distribution().

    Returns
    -------
    eeg_data_filtered : np.ndarray
        Surface Laplacian (spatial second derivative) of EEG, shape (channels, time_samples).
    """
    # ---- load electrode coordinates ----
    x = np.asarray(dist['x'], dtype=float).ravel()
    y = np.asarray(dist['y'], dtype=float).ravel()
    z = np.asarray(dist['z'], dtype=float).ravel()
    
    r = np.sqrt(x**2 + y**2 + z**2)
    x = x / r
    y = y / r
    z = z / r
    
    eeg_data = np.asarray(eeg_data, dtype=float)
    if eeg_data.ndim != 2:
        raise ValueError(f"eeg_data must be 2D (channels, time_samples), got shape {eeg_data.shape}")

    n_ch, n_t = eeg_data.shape
    if len(x) != n_ch or len(y) != n_ch or len(z) != n_ch:
        raise ValueError(
            f"Coordinate length mismatch: EEG has {n_ch} channels, "
            f"but coords are len(x)={len(x)}, len(y)={len(y)}, len(z)={len(z)}."
        )

    # ---- normalize to unit sphere (standard for spherical spline SLF) ----
    r = np.sqrt(x * x + y * y + z * z)
    if np.any(r == 0):
        raise ValueError("Found a zero-length electrode coordinate vector; cannot normalize.")
    x, y, z = x / r, y / r, z / r

    # ---- compute cosine of inter-electrode angles: cos(gamma_ij) ----
    # cos(gamma_ij) = u_i · u_j for unit vectors u_i
    U = np.stack([x, y, z], axis=1)  # (n_ch, 3)
    cosang = U @ U.T
    cosang = np.clip(cosang, -1.0, 1.0)

    # ---- build spherical spline kernels G (potential) and H (surface Laplacian) ----
    # Series truncation (trade-off accuracy vs speed). Common choices: 20~100.
    n_terms = max(20, min(100, 2 * n_ch))  # adaptive but bounded

    # Legendre recursion P_n(cosang)
    # P_0 = 1, P_1 = x, P_{n} = ((2n-1)x P_{n-1} - (n-1)P_{n-2})/n
    P_nm2 = np.ones_like(cosang)        # P_0
    P_nm1 = cosang.copy()               # P_1

    G = np.zeros((n_ch, n_ch), dtype=float)
    H = np.zeros((n_ch, n_ch), dtype=float)

    # Perrin-style coefficients
    # G_ij = Σ_{n=1..N} (2n+1) / [n^m (n+1)^m] * P_n(cosγ)
    # H_ij = -Σ_{n=1..N} (2n+1) * n(n+1) / [n^m (n+1)^m] * P_n(cosγ)
    # (r is assumed 1 after normalization; otherwise divide H by r^2)
    for n in range(1, n_terms + 1):
        if n == 1:
            Pn = P_nm1
        else:
            Pn = ((2 * n - 1) * cosang * P_nm1 - (n - 1) * P_nm2) / n
            P_nm2, P_nm1 = P_nm1, Pn

        denom = (n ** m) * ((n + 1) ** m)
        cG = (2 * n + 1) / denom
        cH = - (2 * n + 1) * (n * (n + 1)) / denom

        G += cG * Pn
        H += cH * Pn

    # Regularization: add lambda*I to G
    G_reg = G + float(_lambda) * np.eye(n_ch, dtype=float)

    # ---- enforce reference-free constraint via augmentation ----
    # Solve for weights w and constant c:
    # [G_reg  1][w] = [V]
    # [1^T    0][c]   [0]
    ones = np.ones((n_ch, 1), dtype=float)
    K = np.block([
        [G_reg, ones],
        [ones.T, np.zeros((1, 1), dtype=float)]
    ])  # (n_ch+1, n_ch+1)

    RHS = np.vstack([eeg_data, np.zeros((1, n_t), dtype=float)])  # (n_ch+1, n_t)

    try:
        sol = np.linalg.solve(K, RHS)  # (n_ch+1, n_t)
    except np.linalg.LinAlgError:
        # fallback (more stable but slower)
        sol = np.linalg.lstsq(K, RHS, rcond=None)[0]

    w = sol[:n_ch, :]  # (n_ch, n_t)

    # Laplacian at electrodes: H @ w
    eeg_data_filtered = H @ w  # (n_ch, n_t)

    return eeg_data_filtered

dist = utils_feature_loading.read_distribution('seed', 'auto', header=True)
filtered_eeg_dict = filter_eeg_seed('sub1ex1', freq=200, save=True, verbose=True,
                                    apply_slf='slf', slf_kwargs={"dist": dist, "m": 4, "_lambda": 1e-4})




from scipy import signal
def ssd_filtering(eeg_data, sampling_rate, 
                  f_sig=(8.0, 12.0), gap=1.0, bw_noise=2.0, n_components=6, filt_order=4, cov_reg=1e-3,):
    """
    SSD (Spatio-Spectral Decomposition) for EEG data.

    Parameters
    ----------
    eeg_data : np.ndarray
        Shape (channels, time_samples).
    sampling_rate : float
        Sampling rate (Hz).
    f_sig : tuple
        Target band (f1, f2).
    gap : float
        Gap between signal band and noise bands (Hz).
    bw_noise : float
        Bandwidth of each noise flank (Hz).
    n_components : int
        Number of SSD components to return.
    filt_order : int
        Butterworth order.
    cov_reg : float
        Covariance regularization strength.

    Returns
    -------
    S : np.ndarray
        SSD components, shape (n_components, time_samples).
    W : np.ndarray
        Spatial filters, shape (channels, n_components).
    A : np.ndarray
        Spatial patterns (activation maps), shape (channels, n_components).
    evals : np.ndarray
        Generalized eigenvalues, shape (channels,).
    meta : dict
        bands, etc.
    """
    def _bandpass_filtfilt(X, fs, band, order=4, ftype="butter"):
        """Zero-phase bandpass filtering for (channels, time)."""
        low, high = band
        if low <= 0 or high >= fs / 2:
            raise ValueError(f"Invalid band {band} for fs={fs}")
        if ftype == "butter":
            b, a = signal.butter(order, [low, high], btype="bandpass", fs=fs)
            return signal.filtfilt(b, a, X, axis=1)
        else:
            raise NotImplementedError("Only butter is implemented in this template.")


    def _covariance(X, reg=0.0):
        """
        Sample covariance for X: (channels, time).
        reg: add reg * trace(C)/C * I for scale-aware regularization.
        """
        Xc = X - X.mean(axis=1, keepdims=True)
        C = (Xc @ Xc.T) / max(Xc.shape[1] - 1, 1)
        if reg and reg > 0:
            C = C + (reg * np.trace(C) / C.shape[0]) * np.eye(C.shape[0])
        return C


    def _ged(Cs, Cn):
        """
        Solve Cs w = lambda Cn w.
        Returns eigenvalues (desc) and eigenvectors.
        """
        # Whiten Cn: Cn = U diag(d) U^T
        d, U = np.linalg.eigh(Cn)
        eps = 1e-12
        d = np.maximum(d, eps)
        Wn = U @ np.diag(1.0 / np.sqrt(d)) @ U.T  # whitening matrix

        M = Wn @ Cs @ Wn.T
        evals, evecs = np.linalg.eigh(M)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        # back-project to original space
        W = Wn.T @ evecs
        return evals, W
    
    X = np.asarray(eeg_data, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"eeg_data must be (channels, time), got {X.shape}")
    C, T = X.shape

    f1, f2 = f_sig
    noise_left = (max(0.1, f1 - gap - bw_noise), max(0.1, f1 - gap))
    noise_right = (f2 + gap, f2 + gap + bw_noise)

    # Filter to build covariances
    Xs = _bandpass_filtfilt(X, sampling_rate, f_sig, order=filt_order)
    Xn1 = _bandpass_filtfilt(X, sampling_rate, noise_left, order=filt_order)
    Xn2 = _bandpass_filtfilt(X, sampling_rate, noise_right, order=filt_order)

    Cs = _covariance(Xs, reg=cov_reg)
    Cn = 0.5 * (_covariance(Xn1, reg=cov_reg) + _covariance(Xn2, reg=cov_reg))

    evals, W_full = _ged(Cs, Cn)

    k = min(n_components, C)
    W = W_full[:, :k]
    S = W.T @ X

    # Spatial patterns: A = inv(W)^T for square W; for rectangular use pseudo-inverse
    WinvT = np.linalg.pinv(W).T
    A = WinvT  # (channels, k)

    meta = {
        "f_sig": f_sig,
        "noise_left": noise_left,
        "noise_right": noise_right,
        "gap": gap,
        "bw_noise": bw_noise,
        "filt_order": filt_order,
        "cov_reg": cov_reg,
    }
    
    # Reconstruct channel EEG
    eeg_data_filtered = A @ S  # (channels, time)
    
    return eeg_data_filtered, S, W, A, evals, meta

# %% save
import os
import h5py
def save_results(dataset, feature, identifier, data, extension='slf'):
    """Saves functional connectivity matrices to an HDF5 file."""
    path_parent = os.path.dirname(os.getcwd())
    path_parent_parent = os.path.dirname(path_parent)
    base_path = os.path.join(path_parent_parent, 'Research_Data', dataset, f'functional connectivity_{extension}', f'{feature}_h5')
    os.makedirs(base_path, exist_ok=True)
    
    file_path = os.path.join(base_path, f"{identifier}.h5")
    with h5py.File(file_path, 'w') as f:
        if isinstance(data, dict):  # Joint band case
            for band, matrix in data.items():
                f.create_dataset(band, data=matrix, compression="gzip")
        else:  # Single band case
            f.create_dataset("connectivity", data=data, compression="gzip")

    print(f"Data saved to {file_path}")

# %%
from feature_engineering import compute_corr_matrices
def fc_matrices_circle_vc_filtering(dataset, subject_range=range(1, 2), experiment_range=range(1, 2),
                 feature='pcc', band='joint', method='slf', save=False, verbose=True):
    """
    Computes functional connectivity matrices for EEG datasets.

    Features:
    - Computes connectivity matrices based on the selected feature and frequency band.
    - Records total and average computation time.
    - Optionally saves results in HDF5 format.

    Parameters:
    - dataset (str): Dataset name ('SEED' or 'DREAMER').
    - subject_range (range): Range of subject IDs (default: range(1, 2)).
    - experiment_range (range): Range of experiment IDs (default: range(1, 2)).
    - band (str): Frequency band ('delta', 'theta', 'alpha', 'beta', 'gamma', or 'joint').
    - save (bool): Whether to save results (default: False).
    - verbose (bool): Whether to print timing information (default: True).

    Returns:
    - dict: Dictionary containing computed functional connectivity matrices.
    """

    dataset = dataset.upper()
    feature = feature.lower()
    band = band.lower()
    method = method.lower()

    valid_datasets = {'SEED', 'DREAMER'}
    valid_features = {'pcc', 'plv', 'mi', 'pli', 'wpli'}
    valid_bands = {'joint', 'theta', 'delta', 'alpha', 'beta', 'gamma'}
    valid_methods ={'slf', 'ssd', 'ged', }

    if dataset not in valid_datasets:
        raise ValueError(f"Invalid dataset '{dataset}'. Supported datasets: {valid_datasets}")
    if band not in valid_bands:
        raise ValueError(f"Invalid band '{band}'. Supported bands: {valid_bands}")

    def eeg_loader(subject, experiment=None):
        """Loads EEG data for a given subject and experiment."""
        identifier = f"sub{subject}ex{experiment}" if dataset == 'SEED' else f"sub{subject}"
        eeg_data = utils_eeg_loading.read_eeg_filtered(dataset, identifier)
        return identifier, eeg_data

    fc_matrices = {}
    start_time = time.time()
    total_experiment_time = 0
    experiment_count = 0

    if dataset == 'SEED': 
        sampling_rate = 200
        experiments = experiment_range
    elif dataset == 'DREAMER':
        sampling_rate = 128
        experiments = [None]

    for subject in subject_range:
        for experiment in experiments:
            experiment_start = time.time()
            experiment_count += 1

            identifier, eeg_data = eeg_loader(subject, experiment)
            bands_to_process = ['delta', 'theta', 'alpha', 'beta', 'gamma'] if band == 'joint' else [band]

            fc_matrices[identifier] = {} if band == 'joint' else None

            for current_band in bands_to_process:
                data = np.array(eeg_data[current_band])
                
                if method == 'slf':
                    eeg_data_filtered = surface_laplacian_filtering(data, sampling_rate, dataset=dataset)
                elif method == 'ssd':
                    eeg_data_filtered,_,_,_,_,_ = ssd_filtering(data, sampling_rate)
                
                if feature == 'pcc':
                    result = compute_corr_matrices(eeg_data_filtered, sampling_rate)    
                    
                if band == 'joint':
                    fc_matrices[identifier][current_band] = result
                else:
                    fc_matrices[identifier] = result

            experiment_duration = time.time() - experiment_start
            total_experiment_time += experiment_duration

            if verbose:
                print(f"Experiment {identifier} completed in {experiment_duration:.2f} seconds")

            if save:
                save_results(dataset, feature, identifier, fc_matrices[identifier], extension=method)

    total_time = time.time() - start_time
    avg_experiment_time = total_experiment_time / experiment_count if experiment_count else 0

    if verbose:
        print(f"\nTotal time taken: {total_time:.2f} seconds")
        print(f"Average time per experiment: {avg_experiment_time:.2f} seconds")

    return eeg_data_filtered

# dist = utils_feature_loading.read_distribution('seed', 'auto', header=True)
# fc_matrices = fc_matrices_circle_vc_filtering('seed', subject_range=range(1, 16), experiment_range=range(1, 4),
#                                               method='ged', band='joint', save=True, verbose=True)