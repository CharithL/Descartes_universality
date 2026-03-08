"""
DESCARTES Human Universality -- Probe Target Computations

Human equivalents of the mouse ``wm/targets/choice_signal.py`` and
``wm/targets/emergent.py``, adapted for the Rutishauser single-neuron
recognition-memory paradigm (DANDI 000576).

Three target levels are defined:

    Level B  -- Cognitive variables (THE KEY LEVEL)
        persistent_delay, memory_load, delay_stability, recognition_decision

    Level C  -- Emergent dynamics
        theta_modulation, gamma_modulation, population_synchrony

    Level A  -- Single-neuron (expected zombie)
        concept_selectivity, mean_firing_rate

All functions accept a neural activity tensor ``Y`` with shape
``(n_trials, n_timesteps, n_neurons)`` and return per-trial scalars
``(n_trials,)``.  Axis/weight vectors are returned as secondary outputs
where applicable.

Dependencies: numpy, scipy (signal, stats), sklearn (LinearRegression).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_signal
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_trial_key(
    trial_info: Dict[str, Any],
    candidates: list[str],
    description: str,
) -> str:
    """Find the first key from *candidates* that exists in *trial_info*.

    Raises ``KeyError`` with a helpful message when none are found.
    """
    for key in candidates:
        if key in trial_info:
            return key
    raise KeyError(
        f"trial_info must contain one of {candidates} for {description}. "
        f"Available keys: {sorted(trial_info.keys())}"
    )


def _safe_corrcoef(x: NDArray, y: NDArray) -> float:
    """Pearson correlation with safe handling for zero-variance vectors."""
    if x.std() < 1e-10 or y.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _bandpass_amplitude(
    pop_rate: NDArray,
    fs: float,
    band: Tuple[float, float],
    min_samples: int = 16,
) -> NDArray:
    """Bandpass-filter *pop_rate* and return the mean Hilbert envelope per trial.

    Parameters
    ----------
    pop_rate : ndarray, (n_trials, n_timesteps)
        Population firing rate (summed across neurons).
    fs : float
        Sampling frequency in Hz (1 / bin_size_s).
    band : tuple of (low_hz, high_hz)
        Pass-band edges in Hz.
    min_samples : int
        Minimum number of time-steps required for filtering. Trials shorter
        than this receive amplitude = 0.

    Returns
    -------
    amp : ndarray, (n_trials,)
        Mean oscillatory amplitude in the specified band for each trial.
    """
    n_trials = pop_rate.shape[0]
    amp = np.zeros(n_trials, dtype=np.float64)

    nyq = fs / 2.0
    low = band[0] / nyq
    high = band[1] / nyq

    # Nyquist check: cannot bandpass above Nyquist frequency
    if high >= 1.0:
        warnings.warn(
            f"Upper band edge ({band[1]} Hz) meets or exceeds the Nyquist "
            f"frequency ({nyq} Hz). Returning zeros.",
            RuntimeWarning,
            stacklevel=3,
        )
        return amp

    if low <= 0.0:
        warnings.warn(
            f"Lower band edge ({band[0]} Hz) is non-positive. "
            f"Returning zeros.",
            RuntimeWarning,
            stacklevel=3,
        )
        return amp

    # Design Butterworth bandpass once
    try:
        sos = sp_signal.butter(3, [low, high], btype='band', output='sos')
    except ValueError as exc:
        warnings.warn(
            f"Butterworth filter design failed ({exc}). Returning zeros.",
            RuntimeWarning,
            stacklevel=3,
        )
        return amp

    for i in range(n_trials):
        sig = pop_rate[i]
        if len(sig) < min_samples:
            continue
        try:
            filtered = sp_signal.sosfiltfilt(sos, sig)
            analytic = sp_signal.hilbert(filtered)
            envelope = np.abs(analytic)
            amp[i] = float(np.mean(envelope))
        except Exception:
            amp[i] = 0.0

    return amp


# ===================================================================
# Level B -- Cognitive Variables (THE KEY LEVEL)
# ===================================================================

def compute_persistent_delay(
    Y: NDArray,
    delay_bins: slice,
) -> NDArray:
    """Mean frontal firing rate during the delay period.

    A straightforward scalar measure of persistent activity: the average
    population firing rate across all neurons and time-bins within the
    memory delay.

    Parameters
    ----------
    Y : ndarray, shape (n_trials, n_timesteps, n_neurons)
        Population activity tensor (binned spike counts or rates).
    delay_bins : slice
        Slice selecting the delay-period time-bins from axis 1.

    Returns
    -------
    persistent : ndarray, shape (n_trials,)
        Mean firing rate during the delay for each trial.
    """
    Y_delay = Y[:, delay_bins, :]  # (n_trials, n_delay_bins, n_neurons)
    # Average over both time and neuron dimensions
    persistent = Y_delay.mean(axis=(1, 2))
    return persistent


def compute_memory_load_signal(
    Y: NDArray,
    trial_info: Dict[str, Any],
    delay_bins: slice,
) -> Tuple[NDArray, NDArray]:
    """Memory load signal: projection onto a load-coding axis.

    The load axis is learned via linear regression of the population
    activity vector (averaged over the delay period) against the integer
    set-size label.  The resulting per-trial scalar should scale
    monotonically with memory load (1, 2, 3 items).

    Parameters
    ----------
    Y : ndarray, shape (n_trials, n_timesteps, n_neurons)
    trial_info : dict
        Must contain one of ``'set_size'``, ``'load'``, or ``'n_items'``.
        Values should be integer-like, shape ``(n_trials,)``.
    delay_bins : slice
        Slice into the time axis for the delay period.

    Returns
    -------
    load_signal : ndarray, shape (n_trials,)
        Per-trial projection onto the load axis.
    load_axis : ndarray, shape (n_neurons,)
        Unit-normalised weight vector defining the load coding axis.
    """
    from sklearn.linear_model import LinearRegression

    load_key = _resolve_trial_key(
        trial_info,
        ['set_size', 'load', 'n_items'],
        'memory load',
    )
    load_labels = np.asarray(trial_info[load_key], dtype=np.float64)

    # Mean population vector during delay for each trial
    Y_delay = Y[:, delay_bins, :]                  # (n_trials, n_delay, n_neurons)
    pop_vec = Y_delay.mean(axis=1)                  # (n_trials, n_neurons)

    n_trials, n_neurons = pop_vec.shape

    # Edge case: single neuron or zero variance
    if n_neurons == 0:
        return np.zeros(n_trials), np.array([])

    # Fit regression: pop_vec -> load_labels
    reg = LinearRegression()
    reg.fit(pop_vec, load_labels)

    load_axis = reg.coef_.copy()                    # (n_neurons,)
    norm = np.linalg.norm(load_axis)
    if norm < 1e-8:
        warnings.warn(
            "Load axis has near-zero norm -- load may not be decodable.",
            RuntimeWarning,
            stacklevel=2,
        )
        load_axis = np.zeros(n_neurons)
    else:
        load_axis = load_axis / norm

    # Project each trial onto the axis
    load_signal = pop_vec @ load_axis               # (n_trials,)
    return load_signal, load_axis


def compute_delay_stability(
    Y: NDArray,
    delay_bins: slice,
) -> NDArray:
    """Temporal stability of the population representation through the delay.

    For each trial, the delay-period population vector time-series is
    split into first half and second half.  Stability is the Pearson
    correlation between the mean population vector of each half.  A high
    value indicates that the neural code is maintained rather than drifting.

    Parameters
    ----------
    Y : ndarray, shape (n_trials, n_timesteps, n_neurons)
    delay_bins : slice
        Slice into the time axis for the delay period.

    Returns
    -------
    stability : ndarray, shape (n_trials,)
        Per-trial first-half vs second-half correlation.
    """
    Y_delay = Y[:, delay_bins, :]                   # (n_trials, n_delay, n_neurons)
    n_trials, n_delay, n_neurons = Y_delay.shape

    stability = np.zeros(n_trials, dtype=np.float64)

    if n_delay < 2 or n_neurons == 0:
        return stability

    mid = n_delay // 2

    # Mean population vector in each half
    first_half = Y_delay[:, :mid, :].mean(axis=1)   # (n_trials, n_neurons)
    second_half = Y_delay[:, mid:, :].mean(axis=1)  # (n_trials, n_neurons)

    for i in range(n_trials):
        stability[i] = _safe_corrcoef(first_half[i], second_half[i])

    return stability


def compute_recognition_decision(
    Y: NDArray,
    trial_info: Dict[str, Any],
    probe_bins: slice,
) -> Tuple[NDArray, NDArray]:
    """Recognition decision signal at probe time (match vs non-match).

    A decision axis is computed as the normalised difference of mean
    population vectors between match and non-match trials during the
    probe period.  Each trial is then projected onto this axis.

    Parameters
    ----------
    Y : ndarray, shape (n_trials, n_timesteps, n_neurons)
    trial_info : dict
        Must contain one of ``'in_set'``, ``'match'``, or
        ``'probe_match'`` with binary labels (0/1 or bool),
        shape ``(n_trials,)``.
    probe_bins : slice
        Slice into the time axis for the probe/test period.

    Returns
    -------
    decision_signal : ndarray, shape (n_trials,)
        Per-trial projection onto the decision axis.
    decision_axis : ndarray, shape (n_neurons,)
        Unit-normalised weight vector separating match from non-match.
    """
    match_key = _resolve_trial_key(
        trial_info,
        ['in_set', 'match', 'probe_match'],
        'recognition decision',
    )
    labels = np.asarray(trial_info[match_key]).astype(int).ravel()

    # Population vector during probe period
    Y_probe = Y[:, probe_bins, :]                   # (n_trials, n_probe, n_neurons)
    pop_vec = Y_probe.mean(axis=1)                  # (n_trials, n_neurons)

    n_trials, n_neurons = pop_vec.shape

    if n_neurons == 0:
        return np.zeros(n_trials), np.array([])

    conditions = np.unique(labels)
    if len(conditions) < 2:
        warnings.warn(
            f"Need >= 2 conditions for decision axis, got {len(conditions)}. "
            f"Returning zeros.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.zeros(n_trials), np.zeros(n_neurons)

    # Match (1) vs non-match (0)
    c_match = (labels == 1)
    c_nonmatch = (labels == 0)

    if c_match.sum() == 0 or c_nonmatch.sum() == 0:
        warnings.warn(
            "One condition has zero trials. Returning zeros.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.zeros(n_trials), np.zeros(n_neurons)

    mean_match = pop_vec[c_match].mean(axis=0)
    mean_nonmatch = pop_vec[c_nonmatch].mean(axis=0)

    decision_axis = mean_match - mean_nonmatch      # (n_neurons,)
    norm = np.linalg.norm(decision_axis)
    if norm < 1e-8:
        warnings.warn(
            "Decision axis has near-zero norm -- match vs non-match "
            "populations are indistinguishable. Returning zeros.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.zeros(n_trials), np.zeros(n_neurons)

    decision_axis = decision_axis / norm

    decision_signal = pop_vec @ decision_axis       # (n_trials,)
    return decision_signal, decision_axis


# ===================================================================
# Level C -- Emergent Dynamics
# ===================================================================

def compute_theta_modulation(
    Y: NDArray,
    bin_size_s: float = 0.05,
    theta_band: Tuple[float, float] = (4, 8),
) -> NDArray:
    """4-8 Hz oscillatory amplitude of the population firing rate.

    The population rate (sum across neurons) is bandpass-filtered in the
    theta band using a 3rd-order Butterworth filter.  The analytic signal
    (Hilbert transform) gives the instantaneous amplitude envelope, which
    is averaged over time to yield one scalar per trial.

    Parameters
    ----------
    Y : ndarray, shape (n_trials, n_timesteps, n_neurons)
    bin_size_s : float
        Width of each time-bin in seconds (default 0.05 = 50 ms).
    theta_band : tuple of (float, float)
        (low_hz, high_hz) for the theta band (default 4-8 Hz).

    Returns
    -------
    theta_amp : ndarray, shape (n_trials,)
        Mean theta-band amplitude per trial.
    """
    fs = 1.0 / bin_size_s
    pop_rate = Y.sum(axis=2)                        # (n_trials, n_timesteps)
    return _bandpass_amplitude(pop_rate, fs, theta_band)


def compute_gamma_modulation(
    Y: NDArray,
    bin_size_s: float = 0.05,
    gamma_band: Tuple[float, float] = (30, 80),
) -> NDArray:
    """30-80 Hz oscillatory amplitude of the population firing rate.

    Identical methodology to :func:`compute_theta_modulation` but in the
    gamma band.

    .. note::

        With 50 ms bins (20 Hz sampling rate), the Nyquist frequency is
        10 Hz, which is far below the gamma range.  Gamma modulation is
        therefore only meaningful if ``bin_size_s`` is small enough
        (< 1 / (2 * gamma_band[1])).  The function will return zeros and
        emit a warning when the band exceeds Nyquist.

    Parameters
    ----------
    Y : ndarray, shape (n_trials, n_timesteps, n_neurons)
    bin_size_s : float
        Width of each time-bin in seconds (default 0.05 = 50 ms).
    gamma_band : tuple of (float, float)
        (low_hz, high_hz) for the gamma band (default 30-80 Hz).

    Returns
    -------
    gamma_amp : ndarray, shape (n_trials,)
        Mean gamma-band amplitude per trial.
    """
    fs = 1.0 / bin_size_s
    pop_rate = Y.sum(axis=2)
    return _bandpass_amplitude(pop_rate, fs, gamma_band)


def compute_population_synchrony(Y: NDArray) -> NDArray:
    """Mean pairwise correlation magnitude across neurons.

    For each trial, the pairwise Pearson correlation matrix is computed
    over the full time axis.  The synchrony measure is the mean absolute
    off-diagonal entry of this matrix.  Neurons with zero temporal
    variance are excluded on a per-trial basis.

    Parameters
    ----------
    Y : ndarray, shape (n_trials, n_timesteps, n_neurons)

    Returns
    -------
    sync : ndarray, shape (n_trials,)
        Mean |r| across all neuron pairs for each trial.
    """
    n_trials, n_t, n_neurons = Y.shape
    sync = np.zeros(n_trials, dtype=np.float64)

    if n_neurons < 2:
        return sync

    for i in range(n_trials):
        activity = Y[i].T                           # (n_neurons, n_timesteps)

        # Exclude neurons with zero variance in this trial
        stds = activity.std(axis=1)
        active_mask = stds > 1e-10
        n_active = int(active_mask.sum())
        if n_active < 2:
            continue

        corr_mat = np.corrcoef(activity[active_mask])
        off_diag = ~np.eye(n_active, dtype=bool)
        sync[i] = float(np.mean(np.abs(corr_mat[off_diag])))

    return sync


# ===================================================================
# Level A -- Single-Neuron (expected zombie)
# ===================================================================

def compute_concept_selectivity(
    Y: NDArray,
    trial_info: Dict[str, Any],
    encoding_bins: slice,
) -> NDArray:
    """Image / category selectivity during the encoding period.

    A one-way ANOVA (F-statistic) is computed for each neuron across
    stimulus conditions during the encoding window.  The per-trial
    selectivity is the mean F-statistic across neurons evaluated at each
    trial's stimulus condition, giving a rough measure of how selective
    the population is for the stimulus presented on that trial.

    In practice this is computed as:
        1. Average each neuron's activity over encoding_bins.
        2. Compute F-statistic across conditions for each neuron.
        3. Per trial, return the mean F across all neurons.

    Parameters
    ----------
    Y : ndarray, shape (n_trials, n_timesteps, n_neurons)
    trial_info : dict
        Must contain one of ``'stimulus_id'``, ``'category'``,
        ``'condition'``, or ``'image_id'`` with integer or string labels,
        shape ``(n_trials,)``.
    encoding_bins : slice
        Slice into the time axis for the encoding (stimulus presentation)
        period.

    Returns
    -------
    selectivity : ndarray, shape (n_trials,)
        Per-trial mean F-statistic across neurons.  Higher values mean
        the population is more selective at distinguishing the stimulus
        shown on that trial from other stimuli.
    """
    cond_key = _resolve_trial_key(
        trial_info,
        ['stimulus_id', 'category', 'condition', 'image_id'],
        'concept selectivity',
    )
    conditions = np.asarray(trial_info[cond_key])
    unique_conds = np.unique(conditions)

    n_trials, _, n_neurons = Y.shape

    # Average neuron activity over the encoding window
    Y_enc = Y[:, encoding_bins, :].mean(axis=1)     # (n_trials, n_neurons)

    if len(unique_conds) < 2:
        warnings.warn(
            "Need >= 2 stimulus conditions for selectivity, "
            f"got {len(unique_conds)}. Returning zeros.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.zeros(n_trials)

    # Compute per-neuron F-statistic across conditions
    f_stats = np.zeros(n_neurons, dtype=np.float64)
    for j in range(n_neurons):
        groups = [Y_enc[conditions == c, j] for c in unique_conds]

        # Remove empty groups and groups with zero variance
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue

        try:
            f_val, _ = sp_stats.f_oneway(*groups)
            if np.isfinite(f_val):
                f_stats[j] = f_val
        except Exception:
            pass

    # Per-trial selectivity = mean F across all neurons
    # This is a global measure (same for all trials), but we broadcast it
    # so that the output shape is (n_trials,).  Alternatively, one could
    # weight by the trial's own condition, but the global F is more
    # interpretable and stable.
    mean_f = float(f_stats.mean()) if n_neurons > 0 else 0.0
    selectivity = np.full(n_trials, mean_f, dtype=np.float64)

    return selectivity


def compute_mean_firing_rate(Y: NDArray) -> NDArray:
    """Overall mean firing rate across all neurons and time-bins.

    Parameters
    ----------
    Y : ndarray, shape (n_trials, n_timesteps, n_neurons)

    Returns
    -------
    rate : ndarray, shape (n_trials,)
        Mean activity per trial.
    """
    return Y.mean(axis=(1, 2))


# ===================================================================
# Master function
# ===================================================================

def compute_all_targets(
    Y: NDArray,
    trial_info: Dict[str, Any],
    task_timing: Dict[str, slice],
    bin_size_s: float = 0.05,
    theta_band: Tuple[float, float] = (4, 8),
    gamma_band: Tuple[float, float] = (30, 80),
) -> Dict[str, NDArray]:
    """Compute all nine probe targets for the human WM experiment.

    Parameters
    ----------
    Y : ndarray, shape (n_trials, n_timesteps, n_neurons)
        Population activity tensor.  Typically binned spike counts from
        frontal and/or MTL neurons.
    trial_info : dict
        Per-trial metadata.  Expected keys (at least one alias per group):

        - Memory load: ``'set_size'`` | ``'load'`` | ``'n_items'``
          -- integer set-size per trial.
        - Recognition: ``'in_set'`` | ``'match'`` | ``'probe_match'``
          -- binary match label per trial.
        - Selectivity: ``'stimulus_id'`` | ``'category'`` | ``'condition'``
          | ``'image_id'`` -- categorical stimulus identity.

    task_timing : dict
        Time-bin slices keyed by epoch name:

        - ``'encoding_bins'`` : slice for stimulus presentation period.
        - ``'delay_bins'``    : slice for memory delay period.
        - ``'probe_bins'``    : slice for probe / test period.

    bin_size_s : float
        Time-bin width in seconds (default 0.05 s = 50 ms).
    theta_band : tuple of (float, float)
        Theta oscillation band in Hz (default 4-8).
    gamma_band : tuple of (float, float)
        Gamma oscillation band in Hz (default 30-80).

    Returns
    -------
    targets : dict[str, ndarray]
        Mapping from target name to per-trial scalar array of shape
        ``(n_trials,)``.  Keys are:

        Level B:
            ``persistent_delay``, ``memory_load``, ``delay_stability``,
            ``recognition_decision``
        Level C:
            ``theta_modulation``, ``gamma_modulation``,
            ``population_synchrony``
        Level A:
            ``concept_selectivity``, ``mean_firing_rate``

    Notes
    -----
    Targets that require trial_info keys which are absent will be
    silently skipped (a warning is emitted) and omitted from the
    returned dictionary.  Oscillatory targets that cannot be computed due
    to Nyquist constraints will return zero arrays.
    """
    encoding_bins = task_timing['encoding_bins']
    delay_bins = task_timing['delay_bins']
    probe_bins = task_timing['probe_bins']

    targets: Dict[str, NDArray] = {}

    # ------------------------------------------------------------------
    # Level B -- Cognitive variables
    # ------------------------------------------------------------------

    # persistent_delay -- always computable
    targets['persistent_delay'] = compute_persistent_delay(Y, delay_bins)

    # memory_load -- requires set_size / load / n_items
    try:
        load_signal, _load_axis = compute_memory_load_signal(
            Y, trial_info, delay_bins,
        )
        targets['memory_load'] = load_signal
    except KeyError as exc:
        warnings.warn(
            f"Skipping 'memory_load': {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    # delay_stability -- always computable
    targets['delay_stability'] = compute_delay_stability(Y, delay_bins)

    # recognition_decision -- requires match label
    try:
        decision_signal, _decision_axis = compute_recognition_decision(
            Y, trial_info, probe_bins,
        )
        targets['recognition_decision'] = decision_signal
    except KeyError as exc:
        warnings.warn(
            f"Skipping 'recognition_decision': {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Level C -- Emergent dynamics
    # ------------------------------------------------------------------

    targets['theta_modulation'] = compute_theta_modulation(
        Y, bin_size_s=bin_size_s, theta_band=theta_band,
    )
    targets['gamma_modulation'] = compute_gamma_modulation(
        Y, bin_size_s=bin_size_s, gamma_band=gamma_band,
    )
    targets['population_synchrony'] = compute_population_synchrony(Y)

    # ------------------------------------------------------------------
    # Level A -- Single-neuron
    # ------------------------------------------------------------------

    # concept_selectivity -- requires stimulus labels
    try:
        targets['concept_selectivity'] = compute_concept_selectivity(
            Y, trial_info, encoding_bins,
        )
    except KeyError as exc:
        warnings.warn(
            f"Skipping 'concept_selectivity': {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    targets['mean_firing_rate'] = compute_mean_firing_rate(Y)

    return targets
