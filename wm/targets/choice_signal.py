"""
DESCARTES WM — Choice Signal Computation

The PRIMARY probe target: persistent choice-selective activity during
the delay period. Computed as the projection of thalamic population
activity onto the choice-coding axis (difference of condition means).
"""

import numpy as np


def compute_choice_axis(Y, trial_types):
    """Compute the population-level choice signal in thalamus.

    Parameters
    ----------
    Y : ndarray, (n_trials, n_timesteps, n_neurons)
        Thalamic population activity (binned spike counts).
    trial_types : ndarray, (n_trials,)
        Binary trial type labels (0=left, 1=right).

    Returns
    -------
    choice_signal : ndarray, (n_trials, n_timesteps)
        Projection onto the choice axis at each timestep.
    choice_axis : ndarray, (n_neurons,)
        The normalised difference-of-means axis.
    """
    conditions = np.unique(trial_types)
    if len(conditions) < 2:
        raise ValueError(
            f"Need at least 2 trial types, got {len(conditions)}: {conditions}"
        )

    # Use first two conditions as left/right
    c0, c1 = conditions[0], conditions[1]

    # Mean population vector for each condition, averaged over time
    mean_0 = Y[trial_types == c0].mean(axis=(0, 1))  # (n_neurons,)
    mean_1 = Y[trial_types == c1].mean(axis=(0, 1))  # (n_neurons,)

    # Choice axis = difference of means, normalised
    choice_axis = mean_1 - mean_0
    norm = np.linalg.norm(choice_axis)
    if norm < 1e-8:
        raise ValueError("Choice axis has zero norm — conditions are identical")
    choice_axis = choice_axis / norm

    # Project each trial's population vector onto this axis at each timestep
    # Y: (n_trials, n_timesteps, n_neurons) @ choice_axis: (n_neurons,)
    choice_signal = np.einsum('tbn,n->tb', Y, choice_axis)

    return choice_signal, choice_axis


def compute_choice_magnitude(choice_signal):
    """Strength of choice encoding: absolute value of choice signal.

    Parameters
    ----------
    choice_signal : ndarray, (n_trials, n_timesteps)

    Returns
    -------
    choice_magnitude : ndarray, (n_trials, n_timesteps)
    """
    return np.abs(choice_signal)


def compute_delay_stability(choice_signal):
    """How constant is choice coding through the delay?

    Computed as the temporal autocorrelation of the choice signal
    across the delay period within each trial.

    Parameters
    ----------
    choice_signal : ndarray, (n_trials, n_timesteps)

    Returns
    -------
    stability : ndarray, (n_trials,)
        Per-trial autocorrelation (high = persistent, low = drifting).
    """
    n_trials, n_t = choice_signal.shape
    stability = np.zeros(n_trials)

    for i in range(n_trials):
        sig = choice_signal[i]
        if np.std(sig) < 1e-10:
            stability[i] = 0.0
            continue
        # Autocorrelation at lag 1
        autocorr = np.corrcoef(sig[:-1], sig[1:])[0, 1]
        stability[i] = float(autocorr)

    return stability


def trial_average_choice_signal(choice_signal):
    """Average choice signal over timesteps for each trial.

    This is the scalar probe target for Ridge regression.

    Returns
    -------
    ndarray, (n_trials,)
    """
    return choice_signal.mean(axis=1)
