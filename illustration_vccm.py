# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 03:43:41 2026

@author: 18307
"""

# %% montage
import numpy as np
import matplotlib.pyplot as plt
import mne
from adjustText import adjust_text

def get_montage_xy_reference_style(channels):
    """
    Get 2D electrode positions using the same montage style as the reference file:
    MNE standard_1005 montage, using raw x-y coordinates.
    """

    montage = mne.channels.make_standard_montage("standard_1005")
    pos3d = montage.get_positions()["ch_pos"]

    # Convert your uppercase channel labels to MNE-compatible labels
    name_map = {
        ch: ch for ch in channels
    }
    
    name_map.update({
    "FP1": "Fp1",
    "FPZ": "Fpz",
    "FP2": "Fp2",

    "AF3": "AF3",
    "AF4": "AF4",

    "FZ": "Fz",
    "FCZ": "FCz",
    "CZ": "Cz",
    "CPZ": "CPz",
    "PZ": "Pz",
    "POZ": "POz",
    "OZ": "Oz",

    # These two are not always available in standard EEG montages.
    # Approximate them around left/right inferior occipital positions.
    "CB1": "PO9",
    "CB2": "PO10",
})
    
    xy = {}

    missing = []
    for ch in channels:
        mne_name = name_map.get(ch, ch)

        if mne_name in pos3d:
            xy[ch] = np.array([pos3d[mne_name][0], pos3d[mne_name][1]])
        else:
            missing.append((ch, mne_name))

    if missing:
        print("Missing channels in montage:")
        for ch, mne_name in missing:
            print(f"{ch} -> {mne_name}")

    return xy

def plot_vc_scalp_connectivity_reference_style(
    VC,
    channels,
    percentile=95,
    title="Scalp connectivity map of estimated VC components",
    node_size=45,
    show_labels=True,
    label_fontsize=9,
    adjust_labels=True,
    save_path=None,
    title_fontsize=12,
    color="blue"
):
    """
    Plot VC components as a scalp topographic connectivity graph.

    Parameters
    ----------
    VC : ndarray, shape (N, N)
        Volume-conduction component matrix.
    channels : list of str
        Electrode names in the same order as VC.
    percentile : float
        Keep only connections above this percentile.
    title : str
        Figure title.
    node_size : int
        Electrode marker size.
    show_labels : bool
        Whether to show electrode labels.
    save_path : str or None
        If given, save the figure to this path.
    """
    channels = list(channels)
    VC = np.asarray(VC).copy()

    if VC.shape[0] != VC.shape[1]:
        raise ValueError("VC must be a square NxN matrix.")

    if VC.shape[0] != len(channels):
        raise ValueError("Number of channels must match VC matrix size.")

    xy = get_montage_xy_reference_style(channels)

    valid_channels = [ch for ch in channels if ch in xy]
    valid_idx = [channels.index(ch) for ch in valid_channels]

    A = VC[np.ix_(valid_idx, valid_idx)].copy()

    # Remove self-connections
    np.fill_diagonal(A, 0)

    # Symmetrize if needed
    A = (A + A.T) / 2

    # Use absolute values if VC can contain negative values
    A_plot = np.abs(A)

    upper_vals = A_plot[np.triu_indices_from(A_plot, k=1)]
    upper_vals = upper_vals[upper_vals > 0]

    if len(upper_vals) == 0:
        raise ValueError("No positive off-diagonal VC values to plot.")

    threshold = np.percentile(upper_vals, percentile)

    vmax = upper_vals.max()

    fig, ax = plt.subplots(figsize=(8, 9), dpi=220)

    # ============================
    # Head outline, same style as reference
    # ============================
    theta = np.linspace(0, 2 * np.pi, 600)
    radius = 0.105

    ax.plot(
        radius * np.cos(theta),
        radius * np.sin(theta),
        linewidth=1.6,
        color="black"
    )

    # Nose
    ax.plot(
        [-0.018, 0, 0.018],
        [radius * 0.98, radius * 1.14, radius * 0.98],
        linewidth=1.6,
        color="black"
    )

    # Ears
    ear_h = 0.046
    ear_w = 0.018
    t = np.linspace(-np.pi / 2, np.pi / 2, 160)

    for s in [-1, 1]:
        ax.plot(
            s * (radius + ear_w * np.cos(t)),
            ear_h * np.sin(t),
            linewidth=1.3,
            color="black"
        )

    # ============================
    # Draw VC connections
    # ============================
    coords = np.array([xy[ch] for ch in valid_channels])
    
    # ============================
    # Fit electrodes inside scalp outline
    # ============================
    max_r = np.max(np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2))
    
    # Option A: adaptive head radius
    radius = max_r * 1.10
    
    # If you prefer fixed reference radius, use this instead:
    # radius = 0.105
    # coords = coords * (radius * 0.92 / max_r)
    
    y_offset = 0.015
    
    for i in range(len(valid_channels)):
        for j in range(i + 1, len(valid_channels)):
            w = A_plot[i, j]

            if w >= threshold:
                # Normalize edge width
                lw = 0.4 + 3.8 * (w - threshold) / (vmax - threshold + 1e-12)

                ax.plot(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1]+y_offset, coords[j, 1]+y_offset],
                    linewidth=lw,
                    alpha=0.42,
                    color=f"tab:{color}",
                    zorder=1
                )

    # ============================
    # Draw electrodes
    # ============================
    ax.scatter(
        coords[:, 0],
        coords[:, 1] + y_offset,
        s=node_size,
        color="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=3
    )

    # ============================
    # Draw labels using adjustText
    # ============================
    if show_labels:
        texts = []

        for ch, (x, y) in zip(valid_channels, coords):
            texts.append(
                ax.text(
                    x,
                    y + y_offset,
                    ch,
                    ha="center",
                    va="center",
                    fontsize=10,
                    zorder=4
                )
            )

        adjust_text(
            texts,
            ax=ax,
            force_text=(0.3, 0.5),
            force_static=(0.2, 0.4),
            only_move={
                "text": "xy",
                "static": "xy",
                "explode": "xy",
                "pull": "xy"
            },
            arrowprops=dict(
                arrowstyle="-",
                linewidth=0.5,
                color="gray",
                alpha=0.7
            )
        )

    # ax.set_title(title, fontsize=title_fontsize, pad=18)

    ax.text(
        0,
        -0.125,
        f"{title} \n \n Top {100 - percentile:.1f}% strongest VC connections",
        ha="center",
        va="center",
        fontsize=title_fontsize
    )

    ax.set_aspect("equal")
    ax.set_xlim(-0.135, 0.135)
    ax.set_ylim(-0.145, 0.135)
    ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

# %% VCCM
from cnn_subnetworks_val_circl_new import read_params
from connectivity_matrix_rebuilding import cm_vc
from utils import utils_visualization
from utils import utils_feature_loading

# channels
channels = utils_feature_loading.read_distribution("seed")["channel"]

# distance matrix
import feature_engineering
_, dm = feature_engineering.compute_distance_matrix(dataset="seed", projection_params={"type": "3d_euclidean"}, visualize=False)
dm = feature_engineering.normalize_matrix(dm)

# feature
# %%
feature = "pcc"
model = "Rational_Quadratic" # "exponential", "rational_quadratic", "generalized_gaussian", "sigmoid"

param = read_params(model.lower(), model_fm="basic", model_rcm="linear", feature=feature)
vccm = cm_vc(dm, param, model.lower(), model_fm='basic', fm_normalization=False)
utils_visualization.draw_projection(vccm, f"VC-CM, {model}", channels, channels, show_colorbar=False, max_labels=64,
                                    title_position="upper", cmap="viridis",
                                    figsize=(10, 10), label_fontsize=12, title_fontsize=24)

# %% 
from utils import utils_feature_loading
cm = utils_feature_loading.read_fcs_global_average("seed", feature)["alpha"]
utils_visualization.draw_projection(cm, "Observed Functional Network", channels, channels, show_colorbar=False, max_labels=64,
                                    title_position="upper", cmap="viridis",
                                    figsize=(10, 10), label_fontsize=12, title_fontsize=24)

rcm = cm-vccm
utils_visualization.draw_projection(rcm, "Corrected Functional Network", channels, channels, show_colorbar=False, max_labels=64,
                                    title_position="upper", cmap="viridis",
                                    figsize=(10, 10), label_fontsize=12, title_fontsize=24)

plot_vc_scalp_connectivity_reference_style(
    cm,
    channels,
    percentile=70,
    title="Scalp connectivity map",
    node_size=90,
    label_fontsize=16,
    title_fontsize=16
)

plot_vc_scalp_connectivity_reference_style(
    vccm,
    channels,
    percentile=70,
    title="Scalp connectivity map of estimated VC components",
    node_size=90,
    label_fontsize=9,
    title_fontsize=16,
    color="orange"
)

plot_vc_scalp_connectivity_reference_style(
    rcm,
    channels,
    percentile=70,
    title="Scalp connectivity map",
    node_size=90,
    label_fontsize=9,
    title_fontsize=16
)