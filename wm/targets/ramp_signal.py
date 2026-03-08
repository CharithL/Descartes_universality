"""
DESCARTES WM — Ramp Signal Computation

SECONDARY probe target: gradually increasing firing rates through the
delay period, ramping toward a motor-preparation threshold.
"""

import numpy as np


def compute_ramp_signal(Y):
    """Extract the temporal ramp component from thalamic population.

    Parameters
    ----------
    Y : ndarray, (n_trials, n_timesteps, n_neurons)

    Returns
    -------
    ramp_signal : ndarray, (n_trials, n_timesteps)
    ramp_axis : ndarray, (n_neurons,)
    """
    # Average across trials
    Y_mean = Y.mean(axis=0)  # (n_timesteps, n_neurons)

    # Fit linear slope to each neuron's temporal profile
    n_t = Y_mean.shape[0]
    time = np.arange(n_t, dtype=float)
    time = (time - time.mean()) / (time.std() + 1e-8)

    slopes = np.array([
        np.polyfit(time, Y_mean[:, n], 1)[0]
        for n in range(Y_mean.shape[1])
    ])

    # Ramp axis = neurons weighted by their slope
    norm = np.linalg.norm(slopes)
    if norm < 1e-8:
        ramp_axis = slopes
    else:
        ramp_axis = slopes / norm

    # Project each trial onto ramp axis
    ramp_signal = np.einsum('tbn,n->tb', Y, ramp_axis)

    return ramp_signal, ramp_axis


def trial_average_ramp_signal(ramp_signal):
    """Average ramp signal over timesteps for each trial.

    Returns
    -------
    ndarray, (n_trials,)
    """
    return ramp_signal.mean(axis=1)
