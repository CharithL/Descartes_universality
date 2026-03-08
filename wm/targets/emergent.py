"""
DESCARTES WM — Emergent Dynamics Computation (Level C)

Population-level dynamical properties of the thalamic population
that emerge from the interaction of many neurons. These are the
gamma_amp equivalents for the working memory experiment.
"""

import numpy as np
from scipy import signal as sp_signal


def compute_population_rate(Y):
    """Total thalamic firing rate per trial per timestep.

    Parameters
    ----------
    Y : ndarray, (n_trials, n_timesteps, n_neurons)

    Returns
    -------
    pop_rate : ndarray, (n_trials, n_timesteps)
    """
    return Y.sum(axis=2)


def compute_theta_modulation(Y, bin_size_s=0.01, theta_band=(4, 8)):
    """Extract theta-band (4-8 Hz) oscillatory amplitude.

    Parameters
    ----------
    Y : ndarray, (n_trials, n_timesteps, n_neurons)
    bin_size_s : float
    theta_band : tuple of float

    Returns
    -------
    theta_amp : ndarray, (n_trials,)
        Mean theta amplitude per trial.
    """
    fs = 1.0 / bin_size_s  # Sampling frequency

    # Population rate
    pop_rate = Y.sum(axis=2)  # (n_trials, n_timesteps)

    n_trials = pop_rate.shape[0]
    theta_amp = np.zeros(n_trials)

    for i in range(n_trials):
        sig = pop_rate[i]
        if len(sig) < 16:
            continue

        # Bandpass filter
        nyq = fs / 2
        low = theta_band[0] / nyq
        high = theta_band[1] / nyq

        if high >= 1.0:
            continue

        try:
            b, a = sp_signal.butter(3, [low, high], btype='band')
            filtered = sp_signal.filtfilt(b, a, sig)
            # Hilbert envelope
            analytic = sp_signal.hilbert(filtered)
            envelope = np.abs(analytic)
            theta_amp[i] = float(np.mean(envelope))
        except Exception:
            theta_amp[i] = 0.0

    return theta_amp


def compute_population_synchrony(Y):
    """Mean pairwise correlation magnitude across thalamic neurons.

    Parameters
    ----------
    Y : ndarray, (n_trials, n_timesteps, n_neurons)

    Returns
    -------
    sync : ndarray, (n_trials,)
    """
    n_trials, n_t, n_neurons = Y.shape
    sync = np.zeros(n_trials)

    if n_neurons < 2:
        return sync

    for i in range(n_trials):
        # Correlation matrix across neurons for this trial
        activity = Y[i].T  # (n_neurons, n_timesteps)

        # Skip if any neuron has zero variance
        stds = activity.std(axis=1)
        active = stds > 1e-10
        if active.sum() < 2:
            continue

        corr_mat = np.corrcoef(activity[active])
        # Mean absolute off-diagonal correlation
        n = corr_mat.shape[0]
        mask = ~np.eye(n, dtype=bool)
        sync[i] = float(np.mean(np.abs(corr_mat[mask])))

    return sync


def compute_all_level_c(Y, trial_types, choice_signal, bin_size_s=0.01):
    """Compute all Level C probe targets.

    Parameters
    ----------
    Y : ndarray, (n_trials, n_timesteps, n_neurons)
    trial_types : ndarray, (n_trials,)
    choice_signal : ndarray, (n_trials, n_timesteps)
    bin_size_s : float

    Returns
    -------
    targets : dict mapping target_name -> ndarray (n_trials,)
    """
    from wm.targets.choice_signal import compute_delay_stability

    return {
        'delay_stability': compute_delay_stability(choice_signal),
        'theta_modulation': compute_theta_modulation(Y, bin_size_s),
        'population_synchrony': compute_population_synchrony(Y),
    }
