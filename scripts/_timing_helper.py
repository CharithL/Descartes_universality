"""
Shared helper for estimating Sternberg task timing from trial metadata.

Used by scripts 13-16 to partition time bins into encoding, delay, and probe
phases. Once NWB schema timing columns are discovered on real data, this
module should be updated to use those columns directly.
"""

from __future__ import annotations


def estimate_task_timing(trial_info: dict, n_bins: int) -> dict:
    """Estimate task phase boundaries from trial metadata.

    The Sternberg task has three phases: encoding, delay, probe.
    We attempt to infer bin boundaries from timing columns in trial_info.
    If specific timing columns are not available, we use reasonable defaults
    based on typical Sternberg task structure:
      - Encoding: first 40% of bins
      - Delay:    middle 30% of bins
      - Probe:    final 30% of bins

    Parameters
    ----------
    trial_info : dict
        Trial metadata dictionary.
    n_bins : int
        Total number of time bins per trial.

    Returns
    -------
    dict
        Task timing with 'encoding_bins', 'delay_bins', 'probe_bins' as slices.
    """
    encoding_end = int(0.4 * n_bins)
    delay_end = int(0.7 * n_bins)

    return {
        'encoding_bins': slice(0, encoding_end),
        'delay_bins': slice(encoding_end, delay_end),
        'probe_bins': slice(delay_end, n_bins),
    }
