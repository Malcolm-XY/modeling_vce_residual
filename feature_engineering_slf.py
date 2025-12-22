# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 13:32:51 2025

@author: 18307
"""
import numpy as np
import pandas as pd

import mne
import time

# %% Filtering Raw EEG
from utils import utils_eeg_loading
def filter_eeg_and_save_circle(dataset, subject_range, experiment_range=None,
                               verbose=True, save=False, apply_filter='surface_laplacian_filtering', kwargs=None):
    # Normalize parameters
    apply_filter = apply_filter.lower()
    dataset = dataset.upper()

    valid_dataset = ['SEED', 'DREAMER']
    if dataset not in valid_dataset:
        raise ValueError(f"{dataset} is not a valid dataset. Valid datasets are: {valid_dataset}")

    if dataset == 'SEED' and subject_range is not None and experiment_range is not None:
        for subject in subject_range:
            for experiment in experiment_range:
                identifier = f'sub{subject}ex{experiment}'
                print(f"Processing: {identifier}.")
                filtered_eeg = filter_eeg_seed(identifier, save=save, verbose=verbose, apply_filter=apply_filter, kwargs=kwargs)
    else:
        raise ValueError("Error of unexpected subject or experiment range designation.")
    
    return filtered_eeg
    
def filter_eeg_seed(identifier, freq=200, save=False, verbose=True, apply_filter='Surface_Laplacian_Filtering', kwargs=None):
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
    apply_filter = apply_filter.lower()
    if apply_filter == 'surface_laplacian_filtering':
        base_path = os.path.abspath(os.path.join(os.getcwd(), "../../Research_Data/SEED/original eeg/Filtered_SLFed_EEG"))
    elif apply_filter == 'spatio_spectral_decomposition':
        base_path = os.path.abspath(os.path.join(os.getcwd(), "../../Research_Data/SEED/original eeg/Filtered_SSDed_EEG"))
    os.makedirs(base_path, exist_ok=True)
    
    # Filter the EEG data into different frequency bands
    filtered_eeg_dict = filter_eeg(eeg, freq=freq, verbose=verbose, apply_filter=apply_filter, kwargs=kwargs)
    
    # Save filtered EEG data if requested
    if save:
        for band, filtered_eeg in filtered_eeg_dict.items():
            path_file = os.path.join(base_path, f"{identifier}_{band}_eeg.fif")
            filtered_eeg.save(path_file, overwrite=True)
            if verbose:
                print(f"Saved {band} band filtered EEG to {path_file}")
    
    return filtered_eeg_dict

def filter_eeg(eeg, freq=128, verbose=False, apply_filter='surface_laplacian_filtering', kwargs=None):
    """
    先（可选）做 SLF / SSD（二选一），再用 MNE 分段滤波到 Delta/Theta/Alpha/Beta/Gamma。

    Parameters
    ----------
    eeg : np.ndarray
        (n_channels, n_samples)
    freq : float
        Sampling rate (Hz)
    apply_filter : str
        'slf' | 'ssd' | None/'none'
    kwargs : dict | None
        - if 'slf': kwargs -> surface_laplacian_filtering(...)
        - if 'ssd': kwargs -> ssd_filtering(...), but f_sig will be overridden per band

    Returns
    -------
    dict[str, mne.io.Raw]
        band name -> MNE Raw object (bandpassed)
    """
    apply_filter = apply_filter.lower()
    eeg = np.asarray(eeg, dtype=float)
    if eeg.ndim != 2:
        raise ValueError(f"eeg must be (n_channels, n_samples), got {eeg.shape}")

    kwargs = kwargs or {}

    freq_bands = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta":  (13, 30),
        "Gamma": (30, 50),
    }

    def _make_raw(X):
        info = mne.create_info(
            ch_names=[f"Ch{i}" for i in range(X.shape[0])],
            sfreq=freq,
            ch_types='eeg'
        )
        return mne.io.RawArray(X, info, verbose="ERROR")

    def _mne_bandpass(raw, low, high):
        return raw.copy().filter(
            l_freq=low, h_freq=high,
            method="fir", phase="zero-double",
            verbose="ERROR"
        )

    apply_filter = (apply_filter or "none").lower()
    band_filtered_eeg = {}

    # ---------- Case 1: SLF on broadband, then bandpass ----------
    if apply_filter == "surface_laplacian_filtering":
        eeg_slf = surface_laplacian_filtering(
            eeg_data=eeg,
            sampling_rate=freq,  # 你说 SLF 内部不使用，但保留接口
            **kwargs
        )
        if verbose:
            print("Applied SLF on broadband EEG (before bandpass).")

        raw = _make_raw(eeg_slf)
        for band, (low, high) in freq_bands.items():
            band_filtered_eeg[band] = _mne_bandpass(raw, low, high)
            if verbose:
                print(f"{band} band filtered: {low}–{high} Hz")

        return band_filtered_eeg

    # ---------- Case 2: SSD per-band, then bandpass ----------
    if apply_filter == "spatio_spectral_decomposition":
        # 复制一份 kwargs，避免污染外部
        ssd_kwargs = dict(kwargs)
        # f_sig 由每个 band 决定，这里强制移除用户传入的 f_sig，避免冲突/歧义
        ssd_kwargs.pop("f_sig", None)

        for band, (low, high) in freq_bands.items():
            eeg_ssd, *_ = ssd_filtering(
                eeg_data=eeg,
                sampling_rate=freq,
                f_sig=(low, high),
                **ssd_kwargs
            )
            if verbose:
                print(f"Applied SSD for {band} target band: {low}–{high} Hz")

            raw = _make_raw(eeg_ssd)
            band_filtered_eeg[band] = _mne_bandpass(raw, low, high)
            if verbose:
                print(f"{band} band filtered (post-SSD): {low}–{high} Hz")

        return band_filtered_eeg

    # ---------- Case 3: no spatial filter, only bandpass ----------
    if apply_filter in ("none", "no", "false", "0"):
        raw = _make_raw(eeg)
        for band, (low, high) in freq_bands.items():
            band_filtered_eeg[band] = _mne_bandpass(raw, low, high)
            if verbose:
                print(f"{band} band filtered: {low}–{high} Hz")
        return band_filtered_eeg

    raise ValueError("apply_filter must be one of: 'slf', 'ssd', None/'none'")

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
    def _sanitize_band(band, fs, fmin=0.1, margin=1e-6):
        """Clip band into (0, fs/2) and ensure low < high."""
        low, high = band
        nyq = fs / 2.0
        low = max(fmin, float(low))
        high = min(nyq - fmin, float(high))  # 留一点余量避免贴边
        if not (low + margin < high):
            return None
        return (low, high)

    def _bandpass_filtfilt(X, fs, band, order=4):
        band = _sanitize_band(band, fs)
        if band is None:
            return None
        low, high = band
        b, a = signal.butter(order, [low, high], btype="bandpass", fs=fs)
        return signal.filtfilt(b, a, X, axis=1)

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

    noise_left_raw  = (f1 - gap - bw_noise, f1 - gap)
    noise_right_raw = (f2 + gap, f2 + gap + bw_noise)
    
    Xs  = _bandpass_filtfilt(X, sampling_rate, f_sig, order=filt_order)
    Xn1 = _bandpass_filtfilt(X, sampling_rate, noise_left_raw, order=filt_order)
    Xn2 = _bandpass_filtfilt(X, sampling_rate, noise_right_raw, order=filt_order)
    
    if Xs is None:
        raise ValueError(f"Signal band {f_sig} invalid for fs={sampling_rate}")
    
    Cs = _covariance(Xs, reg=cov_reg)
    
    # 允许单侧噪声
    noise_covs = []
    if Xn1 is not None:
        noise_covs.append(_covariance(Xn1, reg=cov_reg))
    if Xn2 is not None:
        noise_covs.append(_covariance(Xn2, reg=cov_reg))
    
    if len(noise_covs) == 0:
        raise ValueError(
            f"Both noise flanks invalid for f_sig={f_sig} with gap={gap}, bw_noise={bw_noise}, fs={sampling_rate}"
        )
    
    Cn = sum(noise_covs) / len(noise_covs)

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

# %% Functional Networks
import os
import h5py
from utils import utils_basic_reading
def read_eeg_filtered(dataset, identifier, freq_band='joint', object_type='pandas_dataframe', folder_name='Filtered_SLFed_EEG'):
    """
    Read filtered EEG data for the specified experiment and frequency band.

    Parameters:
    dataset (str): Dataset name (e.g., 'SEED', 'DREAMER').
    identifier (str): Identifier for the subject/session.
    freq_band (str): Frequency band to load ("alpha", "beta", "gamma", "delta", "theta", or "joint").
                     Default is "joint", which loads all bands.
    object_type (str): Desired output format: 'pandas_dataframe', 'numpy_array', or 'mne'.

    Returns:
    mne.io.Raw | dict | pandas.DataFrame | numpy.ndarray:
        - If 'mne', returns the MNE Raw object (or a dictionary of them for 'joint').
        - If 'pandas_dataframe', returns a DataFrame with EEG data.
        - If 'numpy_array', returns a NumPy array with EEG data.

    Raises:
    ValueError: If the specified frequency band is not valid.
    FileNotFoundError: If the expected file does not exist.
    """
    # Valid options
    valid_datasets = ['SEED', 'DREAMER']
    valid_bands = ['alpha', 'beta', 'gamma', 'delta', 'theta', 'joint']
    valid_object_types = ['pandas_dataframe', 'numpy_array', 'mne', 'fif']
    
    # Normalize inputs
    dataset = dataset.upper()
    identifier = identifier.lower()
    freq_band = freq_band.lower()
    object_type = object_type.lower()
    
    # Validate inputs
    if dataset not in valid_datasets:
        raise ValueError(f"Invalid dataset: {dataset}. Choose from {', '.join(valid_datasets)}.")
    
    if freq_band not in valid_bands:
        raise ValueError(f"Invalid frequency band: {freq_band}. Choose from {', '.join(valid_bands)}.")
    
    if object_type not in valid_object_types:
        raise ValueError(f"Invalid object type: {object_type}. Choose from {', '.join(valid_object_types)}.")
    
    # Construct base path
    path_parent_parent = os.path.dirname(os.path.dirname(os.getcwd()))
    base_path = os.path.join(path_parent_parent, 'Research_Data', dataset, 'original eeg', folder_name)
    
    # Function to process a single frequency band
    def process_band(band):
        file_path = os.path.join(base_path, f'{identifier}_{band.capitalize()}_eeg.fif')
        try:
            raw_data = mne.io.read_raw_fif(file_path, preload=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}. Check the path and file existence.")
            
        if object_type == 'pandas_dataframe':
            return pd.DataFrame(raw_data.get_data(), index=raw_data.ch_names)
        elif object_type == 'numpy_array':
            return raw_data.get_data()
        else:  # Default to MNE Raw object / .fif object
            return raw_data
    
    # Handle joint vs. single band request
    if freq_band == 'joint':
        result = {}
        for band in ['alpha', 'beta', 'gamma', 'delta', 'theta']:
            result[band] = process_band(band)
        return result
    else:
        return process_band(freq_band)

def read_fcs(dataset, identifier, feature, band='joint', file_type='.h5', folder_name='functional connectivity_slf'):
    """
    Reads functional connectivity (FCS) data from a file (HDF5 or MAT format).

    Parameters:
    - dataset (str): Dataset name (e.g., 'SEED').
    - identifier (str): Subject or experiment identifier.
    - feature (str): Feature type (e.g., 'pcc', 'pli').
    - band (str): Frequency band to extract (default: 'joint').
    - file_type (str): File extension indicating format, either '.h5' or '.mat'.

    Returns:
    - dict: FCS data for the specified band. If the band is not found, returns an empty dict.
    
    Raises:
    - ValueError: If the specified file_type is unsupported.
    - FileNotFoundError: If the corresponding file does not exist.
    """
    dataset = dataset.upper()
    identifier = identifier.lower()
    feature = feature.lower()
    band = band.lower()

    base_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    base_dir = os.path.join(base_path, 'Research_Data', dataset, folder_name)

    if file_type == '.h5':
        path_file = os.path.join(base_dir, f'{feature}_h5', f"{identifier}.h5")
        fcs_data = utils_basic_reading.read_hdf5(path_file)
    elif file_type == '.mat':
        path_file = os.path.join(base_dir, f'{feature}_mat', f"{identifier}.mat")
        fcs_data = utils_basic_reading.read_mat(path_file)
    else:
        raise ValueError(f"Unsupported file_type: {file_type}. Supported types are '.h5' and '.mat'.")

    return fcs_data if band == 'joint' else fcs_data.get(band, {})

def read_fcs_global_average(dataset, feature, band='joint', sub_range=range(1, 16), folder_name='functional connectivity_slf'):
    dataset, feature, band = dataset.upper(), feature.lower(), band.lower()
    path_parent_parent = os.path.dirname(os.path.dirname(os.getcwd()))
    path_file = os.path.join(path_parent_parent, 'Research_Data', dataset, folder_name, 
                             f'{feature}_h5', f'global_averaged_{sub_range.stop-1}_15.h5')
    fcs_temp = utils_basic_reading.read_hdf5(path_file)
    return fcs_temp if band == 'joint' else fcs_temp.get(band, {})

def save_results(dataset, feature, identifier, data, folder_name='functional connectivity_slf'):
    """Saves functional connectivity matrices to an HDF5 file."""
    path_parent = os.path.dirname(os.getcwd())
    path_parent_parent = os.path.dirname(path_parent)
    base_path = os.path.join(path_parent_parent, 'Research_Data', dataset, folder_name, f'{feature}_h5')
    os.makedirs(base_path, exist_ok=True)
    
    file_path = os.path.join(base_path, f"{identifier}.h5")
    with h5py.File(file_path, 'w') as f:
        if isinstance(data, dict):  # Joint band case
            for band, matrix in data.items():
                f.create_dataset(band, data=matrix, compression="gzip")
        else:  # Single band case
            f.create_dataset("connectivity", data=data, compression="gzip")

    print(f"Data saved to {file_path}")

from feature_engineering import compute_corr_matrices, compute_plv_matrices
def fc_matrices_circle_vc_filtered(dataset, subject_range=range(1, 2), experiment_range=range(1, 2),
                                   feature='pcc', band='joint', method='surface_laplacian_filtering', save=False, verbose=True):
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
    valid_methods = {'surface_laplacian_filtering', 'spatio_spectral_decomposition'}
    
    if dataset not in valid_datasets:
        raise ValueError(f"Invalid dataset '{dataset}'. Supported datasets: {valid_datasets}")
    if feature not in valid_features:
        raise ValueError(f"Invalid feature '{feature}'. Supported features: {valid_features}")
    if band not in valid_bands:
        raise ValueError(f"Invalid band '{band}'. Supported bands: {valid_bands}")
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Supported methods: {valid_methods}")

    if method == 'surface_laplacian_filtering':
        folder_name = 'Filtered_SLFed_EEG'
        save_folder = 'functional_connectivity_slfed'
    elif method == 'spatio_spectral_decomposition':
        folder_name = 'Filtered_SSDed_EEG'
        save_folder = 'functional_connectivity_ssded'

    def eeg_loader(subject, experiment=None, folder_name='Filtered_SLFed_EEG'):
        """Loads EEG data for a given subject and experiment."""
        identifier = f"sub{subject}ex{experiment}" if dataset == 'SEED' else f"sub{subject}"
        eeg_data = read_eeg_filtered(dataset, identifier, folder_name=folder_name)
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

            identifier, eeg_data_filtered = eeg_loader(subject, experiment, folder_name)
            bands_to_process = ['delta', 'theta', 'alpha', 'beta', 'gamma'] if band == 'joint' else [band]

            fc_matrices[identifier] = {} if band == 'joint' else None

            for current_band in bands_to_process:
                data = np.array(eeg_data_filtered[current_band])
                
                if feature == 'pcc':
                    result = compute_corr_matrices(data, sampling_rate)    
                elif feature == 'plv':
                    result = compute_plv_matrices(data, sampling_rate)
                
                if band == 'joint':
                    fc_matrices[identifier][current_band] = result
                else:
                    fc_matrices[identifier] = result

            experiment_duration = time.time() - experiment_start
            total_experiment_time += experiment_duration

            if verbose:
                print(f"Experiment {identifier} completed in {experiment_duration:.2f} seconds")

            if save:
                save_results(dataset, feature, identifier, fc_matrices[identifier], folder_name=save_folder)

    total_time = time.time() - start_time
    avg_experiment_time = total_experiment_time / experiment_count if experiment_count else 0

    if verbose:
        print(f"\nTotal time taken: {total_time:.2f} seconds")
        print(f"Average time per experiment: {avg_experiment_time:.2f} seconds")

    return None

from utils import utils_visualization
def compute_average_fcs(dataset, subjects=range(1, 2), experiments=range(1, 2), 
                        feature='pcc', band='joint', filtering_method='surface_laplacian_filtering', # 'spatio_spectral_decomposition'
                        in_file_type='.h5', 
                        save=False, verbose=False, visualization=False):
    """
    Computes and optionally saves or visualizes the averaged functional connectivity matrices.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g., 'seed').
    subjects : iterable
        List or range of subject indices.
    experiments : iterable
        List or range of experiment indices.
    feature : str
        Feature type, e.g., 'pcc', 'plv', 'pli'.
    band : str
        Frequency band or 'joint' for all bands.
    in_file_type : str
        Input file type, '.h5' or '.mat'.
    out_file_type : str
        Output file type, '.h5' or '.mat'.
    save : bool
        Whether to save the result.
    verbose : bool
        Whether to print verbose output.
    visualization : bool
        Whether to visualize the global averaged matrix.

    Returns
    -------
    np.ndarray
        The global averaged functional connectivity matrix.
    """
    assert filtering_method.lower() in {'surface_laplacian_filtering', 'spatio_spectral_decomposition'}, "Unsupported filtering method."
    assert dataset.lower() in {'seed'}, "Unsupported dataset."
    assert feature.lower() in {'pcc', 'plv', 'mi', 'pli', 'wpli'}, "Invalid feature."
    assert band.lower() in {'joint', 'alpha', 'beta', 'gamma', 'delta', 'theta'}, "Invalid band."
    assert in_file_type in {'.h5', '.mat'}, "Unsupported input file type."

    if filtering_method.lower() == 'surface_laplacian_filtering':
        read_folder = 'functional_connectivity_SLFed'
        save_folder = 'functional_connectivity_SLFed'
    elif filtering_method.lower() == 'spatio_spectral_decomposition':
        read_folder = 'functional_connectivity_SSDed'
        save_folder = 'functional_connectivity_SSDed'
    
    fcs_averaged_dict, fcs_averaged_dict_ = [], {'alpha': [], 'beta': [], 'gamma': [], 'delta': [], 'theta': []}
    
    for subject in subjects:
        for experiment in experiments:
            identifier = f"sub{subject}ex{experiment}"
            if verbose:
                print(f"Processing: {identifier}")
            
            features = read_fcs(dataset, identifier, feature, band, in_file_type, folder_name=read_folder)
            
            if band == 'joint':
                try:
                    avg_bands = [{"average": np.mean(features[b], axis=0), 
                                  "band": b, "subject": subject, "experiment": experiment} 
                                 for b in ['alpha', 'beta', 'gamma', 'delta', 'theta']]
                    
                    fcs_averaged_dict.append(avg_bands)
                    for entry in avg_bands:                        
                        fcs_averaged_dict_[entry["band"]].append(entry["average"])
                    
                    # Correct theta/delta swap if necessary
                except KeyError:
                    avg_bands = [{"average": np.mean(features[b], axis=0), 
                                  "band": b, "subject": subject, "experiment": experiment} 
                                 for b in ['alpha', 'beta', 'gamma']]
                    
                    fcs_averaged_dict.append(avg_bands)
                    for entry in avg_bands:                        
                        fcs_averaged_dict_[entry["band"]].append(entry["average"])

    # Compute global average
    try:
        fcs_global_averaged = {b: np.mean(fcs_averaged_dict_[b], axis=0)
                               for b in ['alpha', 'beta', 'gamma', 'delta', 'theta']}
    except KeyError:
        fcs_global_averaged = {b: np.mean(fcs_averaged_dict_[b], axis=0)
                               for b in ['alpha', 'beta', 'gamma']}
        
    if visualization:
        for fc in fcs_global_averaged.values():
            utils_visualization.draw_projection(fc)

    if save:
        save_results(dataset, feature, f'global_averaged_{subject}_15', fcs_global_averaged, folder_name=save_folder)
        
        if verbose:
            print("Results saved to .h5 and .mat")

    return fcs_global_averaged, fcs_averaged_dict_

# %%
if __name__ == '__main__':
    # %% Surface Laplacian Filtering
    # slf eeg
    # from utils import utils_feature_loading
    # dist = utils_feature_loading.read_distribution('seed', 'auto', header=True)
    # filter_eeg_and_save_circle('seed', subject_range=range(1,2), experiment_range=range(1,2), 
    #                            verbose=True, save=False, apply_filter='surface_laplacian_filtering',
    #                            kwargs={"dist": dist, "m": 4, "_lambda": 1e-4})
    
    # slf fns
    # fc_matrices_circle_vc_filtered('seed', subject_range=range(1, 2), experiment_range=range(1, 4),
    #                                 feature='plv', band='joint', method='surface_laplacian_filtering', save=False, verbose=True)
    
    # averaged fn
    # compute_average_fcs('seed', range(1, 6), range(1, 4), 
    #                     feature='plv', band='joint', filtering_method='surface_laplacian_filtering',
    #                     in_file_type='.h5', 
    #                     save=False, verbose=True, visualization=True)
    
    # %% Spatio Spectral Decomposition
    # ssd eeg
    # filter_eeg_and_save_circle('seed', subject_range=range(1,2), experiment_range=range(1,4), 
    #                            verbose=True, save=False, apply_filter='spatio_spectral_decomposition')
    
    # ssd fns
    fc_matrices_circle_vc_filtered('seed', subject_range=range(1, 16), experiment_range=range(1, 4),
                                    feature='plv', band='joint', method='spatio_spectral_decomposition', save=True, verbose=True)
    
    # averaged fn
    compute_average_fcs('seed', range(1, 6), range(1, 4), 
                        feature='plv', band='joint', filtering_method='spatio_spectral_decomposition',
                        in_file_type='.h5',
                        save=True, verbose=True, visualization=True)
    
    compute_average_fcs('seed', range(1, 11), range(1, 4), 
                        feature='plv', band='joint', filtering_method='spatio_spectral_decomposition',
                        in_file_type='.h5',
                        save=True, verbose=True, visualization=True)
    
    compute_average_fcs('seed', range(1, 16), range(1, 4), 
                        feature='plv', band='joint', filtering_method='spatio_spectral_decomposition',
                        in_file_type='.h5',
                        save=True, verbose=True, visualization=True)
    
    # %% End Program
    from cnn_val_circle import end_program_actions
    end_program_actions(play_sound=True, shutdown=True, countdown_seconds=120)