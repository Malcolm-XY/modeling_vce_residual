# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:02:23 2025

@author: usouu
"""

import os
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# %% Visualization
def draw_heatmap_1d(data, yticklabels=None, figsize=(2, 10), title=None):
    """
    Plots a heatmap for an Nx1 array (vertical orientation).

    Parameters:
        data (numpy.ndarray): Nx1 array for visualization.
        yticklabels (list, optional): Labels for the y-axis. If None, indices will be used.
    """
    if yticklabels is None:
        yticklabels = list(range(data.shape[0]))  # Automatically generate indices as labels
    
    if title == None:
        title = "Vertical Heatmap of Nx1 Array"
    
    if len(data.shape) == 1:
        data = np.reshape(data, (-1, 1))
    
    data = np.array(data, dtype=float)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        data, 
        cmap='Blues',
        annot=False,
        linewidths=0.5, 
        xticklabels=False, 
        yticklabels=yticklabels
    )
    plt.title(title)
    plt.show()

def draw_joint_heatmap_1d(data_dict, xticklabels=None, title="Heatmap of Channel Importances", cmap="viridis", xrot=60, yrot=0):
    # --- Validate input ---
    if not data_dict:
        raise ValueError("data_dict cannot be empty.")
    
    # --- Prepare data ---
    labels = list(data_dict.keys())
    data = np.vstack(list(data_dict.values()))
    
    # --- Plot ---
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        data,
        cmap=cmap,
        cbar=True,
        xticklabels=xticklabels if xticklabels is not None else False,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="gray"
    )
    
    # --- Rotation settings ---
    plt.xticks(rotation=xrot)
    plt.yticks(rotation=yrot)
    
    # --- Labels and layout ---
    plt.title(title, fontsize=14)
    plt.xlabel("Channel Index", fontsize=12)
    plt.ylabel("Model", fontsize=12)
    plt.tight_layout()
    plt.show()

def draw_projection(sample_projection, title=None,
                    xticklabels=None, yticklabels=None,
                    show_colorbar=True, max_labels=20,
                    title_position="upper", cmap="viridis",
                    figsize=(6, 5),
                    label_fontsize=10,
                    title_fontsize=12):
    """
    Visualizes 2D or 3D data projections.

    Parameters:
        sample_projection (np.ndarray): 2D matrix, or 3D array where each slice
            along axis 0 is visualized separately.
        title (str): Optional plot title for 2D input.
        xticklabels (list): Optional x-axis labels.
        yticklabels (list): Optional y-axis labels.
        show_colorbar (bool): Whether to display the color bar.
        max_labels (int): Maximum number of displayed labels before sparsifying
            with omission markers.
        title_position (str): Position of the title, either "upper" or "lower".
        cmap (str): Colormap used by imshow.
        figsize (tuple): Figure size, e.g., (6, 5).
        label_fontsize (int or float): Font size of x/y tick labels.
        title_fontsize (int or float): Font size of the plot title.
    """
    if title is None:
        title = "2D Matrix Visualization"

    if title_position not in ["upper", "lower"]:
        raise ValueError("title_position must be either 'upper' or 'lower'")

    def sparsify_labels_with_ellipsis(labels, max_labels):
        """
        Return sparse tick positions and labels with '…' inserted to indicate
        omitted regions.
        """
        n = len(labels)

        if n == 0:
            return [], []

        if n <= max_labels:
            return list(range(n)), list(labels)

        # Keep evenly spaced labels, always including first and last
        core_count = max(2, max_labels)
        idx = np.linspace(0, n - 1, num=core_count, dtype=int)
        idx = np.unique(idx).tolist()

        tick_positions = []
        tick_labels = []

        prev = None
        for current in idx:
            if prev is not None and current - prev > 1:
                ellipsis_pos = (prev + current) / 2
                tick_positions.append(ellipsis_pos)
                tick_labels.append("…")

            tick_positions.append(current)
            tick_labels.append(labels[current])
            prev = current

        return tick_positions, tick_labels

    def apply_axis_labels(ax, xticks, yticks):
        if xticks is not None:
            x_pos, x_lab = sparsify_labels_with_ellipsis(xticks, max_labels)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                x_lab,
                rotation=90,
                fontsize=label_fontsize
            )

        if yticks is not None:
            y_pos, y_lab = sparsify_labels_with_ellipsis(yticks, max_labels)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(
                y_lab,
                fontsize=label_fontsize
            )

    def apply_title(ax, plot_title):
        if title_position == "upper":
            ax.set_title(
                plot_title,
                pad=10,
                fontsize=title_fontsize
            )
        elif title_position == "lower":
            ax.set_xlabel(
                plot_title,
                labelpad=20,
                fontsize=title_fontsize
            )

    def plot_single(matrix, plot_title):
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix, cmap=cmap)

        if show_colorbar:
            plt.colorbar(im, ax=ax)

        apply_title(ax, plot_title)
        apply_axis_labels(ax, xticklabels, yticklabels)

        plt.tight_layout()
        plt.show()

    if sample_projection.ndim == 2:
        plot_single(sample_projection, title)

    elif sample_projection.ndim == 3 and sample_projection.shape[0] <= 100:
        for i in range(sample_projection.shape[0]):
            plot_single(sample_projection[i], f"Channel {i + 1} Visualization")

    else:
        raise ValueError(
            f"The dimension of sample matrix for drawing is wrong, "
            f"shape of sample: {sample_projection.shape}"
        )

def draw_projection_(sample_projection, title=None):
    """
    Visualizes data projections (common for both datasets).
    """
    if title == None:
        title = "2D Matrix Visualization"
    
    if sample_projection.ndim == 2:
        plt.imshow(sample_projection, cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.show()
    elif sample_projection.ndim == 3 and sample_projection.shape[0] <= 100:
        for i in range(sample_projection.shape[0]):
            plt.imshow(sample_projection[i], cmap='viridis')
            plt.colorbar()
            plt.title(f"Channel {i + 1} Visualization")
            plt.show()
    else:
        raise ValueError(f"the dimension of sample matrix for drawing is wrong, shape of sample: {sample_projection.shape}")

# %% End Program Actions
import time
import threading
def shutdown_with_countdown(countdown_seconds=30):
    """
    Initiates a shutdown countdown, allowing the user to cancel shutdown within the given time.

    Args:
        countdown_seconds (int): The number of seconds to wait before shutting down.
    """
    def cancel_shutdown():
        nonlocal shutdown_flag
        user_input = input("\nPress 'c' and Enter to cancel shutdown: ").strip().lower()
        if user_input == 'c':
            shutdown_flag = False
            print("Shutdown cancelled.")

    # Flag to determine whether to proceed with shutdown
    shutdown_flag = True

    # Start a thread to listen for user input
    input_thread = threading.Thread(target=cancel_shutdown, daemon=True)
    input_thread.start()

    # Countdown timer
    print(f"Shutdown scheduled in {countdown_seconds} seconds. Press 'c' to cancel.")
    for i in range(countdown_seconds, 0, -1):
        print(f"Time remaining: {i} seconds", end="\r")
        time.sleep(1)

    # Check the flag after countdown
    if shutdown_flag:
        print("\nShutdown proceeding...")
        os.system("shutdown /s /t 1")  # Execute shutdown command
    else:
        print("\nShutdown aborted.")

def end_program_actions(play_sound=True, shutdown=False, countdown_seconds=120):
    """
    Performs actions at the end of the program, such as playing a sound or shutting down the system.

    Args:
        play_sound (bool): If True, plays a notification sound.
        shutdown (bool): If True, initiates shutdown with a countdown.
        countdown_seconds (int): Countdown time for shutdown confirmation.
    """
    if play_sound:
        try:
            import winsound
            print("Playing notification sound...")
            winsound.Beep(1000, 500)  # Frequency: 1000Hz, Duration: 500ms
        except ImportError:
            print("winsound module not available. Skipping sound playback.")

    if shutdown:
        shutdown_with_countdown(countdown_seconds)

# %% Example Usage
# if __name__ == '__main__':
