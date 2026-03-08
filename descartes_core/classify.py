"""
DESCARTES Core — Final Variable Classification

Combines evidence from probing stages:
  1. Ridge ΔR²      — is it decodable beyond chance?
  2. Causal ablation — does the network actually use it?

Final categories (ordered by scientific importance):
  ZOMBIE:                 ΔR² < 0.1
  LEARNED_BYPRODUCT:      ΔR² > 0.1, above baseline, but ablation z > -2
  MANDATORY_CONCENTRATED: Causal (z < -2), breaks at ≤ 10%
  MANDATORY_DISTRIBUTED:  Causal, breaks at 10-60%
  MANDATORY_REDUNDANT:    Causal, breaks at > 70%
"""

import logging
from collections import Counter

from descartes_core.config import (
    CAUSAL_Z_THRESHOLD,
    DELTA_THRESHOLD_LEARNED,
)

logger = logging.getLogger(__name__)

CATEGORIES = [
    'ZOMBIE',
    'LEARNED_BYPRODUCT',
    'MANDATORY_CONCENTRATED',
    'MANDATORY_DISTRIBUTED',
    'MANDATORY_REDUNDANT',
]


def classify_variable(ridge_result, ablation_result=None):
    """Classify a single variable through the probing cascade.

    Parameters
    ----------
    ridge_result : dict
        From probe_single_variable(): must contain 'delta_R2', 'R2_trained',
        'R2_untrained', 'p_value'.
    ablation_result : dict, optional
        From resample_ablation() or causal_ablation(): must contain
        'classification', 'breaking_point'.

    Returns
    -------
    classification : dict
    """
    delta_r2 = ridge_result.get('delta_R2', 0.0)
    r2_trained = ridge_result.get('R2_trained', 0.0)
    r2_untrained = ridge_result.get('R2_untrained', 0.0)
    p_value = ridge_result.get('p_value', 1.0)
    var_name = ridge_result.get('var_name', 'unknown')

    evidence = {
        'var_name': var_name,
        'delta_R2': delta_r2,
        'R2_trained': r2_trained,
        'R2_untrained': r2_untrained,
        'p_value': p_value,
    }

    # Stage 1: Zombie test
    if delta_r2 < DELTA_THRESHOLD_LEARNED:
        evidence['stage_reached'] = 'ridge'
        return {
            'final_category': 'ZOMBIE',
            'evidence': evidence,
            'stage_reached': 'ridge',
        }

    # Stage 2: Causal ablation
    if ablation_result is not None:
        abl_class = ablation_result.get('classification', 'NON_CAUSAL')
        breaking_point = ablation_result.get('breaking_point', None)

        evidence['ablation_classification'] = abl_class
        evidence['breaking_point'] = breaking_point
        evidence['baseline_cc'] = ablation_result.get('baseline_cc', None)

        ablation_steps = ablation_result.get('ablation_steps', [])
        if ablation_steps:
            min_z = min(s.get('z_score', 0.0) for s in ablation_steps)
            evidence['min_z_score'] = min_z

        evidence['stage_reached'] = 'ablation'

        if abl_class == 'NON_CAUSAL':
            return {
                'final_category': 'LEARNED_BYPRODUCT',
                'evidence': evidence,
                'stage_reached': 'ablation',
            }

        if abl_class in ('MANDATORY_CONCENTRATED', 'MANDATORY_DISTRIBUTED',
                         'MANDATORY_REDUNDANT'):
            return {
                'final_category': abl_class,
                'evidence': evidence,
                'stage_reached': 'ablation',
            }

    # If ablation was not run, classify as LEARNED (pending ablation)
    evidence['stage_reached'] = 'ridge_only'
    return {
        'final_category': 'LEARNED_BYPRODUCT',
        'evidence': evidence,
        'stage_reached': 'ridge_only',
        'note': 'ablation_not_run',
    }


def print_classification_summary(classifications):
    """Print formatted summary of zombie test results.

    Parameters
    ----------
    classifications : list of dict
        Each from classify_variable().
    """
    total = len(classifications)
    counts = Counter(c['final_category'] for c in classifications)

    print("\n" + "=" * 70)
    print("  DESCARTES — Variable Classification Summary")
    print("=" * 70)
    print(f"\n  Total variables probed: {total}\n")

    print("  Category Breakdown:")
    print("  " + "-" * 50)
    for cat in CATEGORIES:
        n = counts.get(cat, 0)
        pct = (n / max(total, 1)) * 100
        bar = "#" * int(pct / 2)
        print(f"    {cat:<28s}  {n:4d}  ({pct:5.1f}%)  {bar}")

    n_mandatory = sum(counts.get(c, 0) for c in CATEGORIES
                      if c.startswith('MANDATORY'))
    n_zombie = counts.get('ZOMBIE', 0)

    print(f"\n  {n_zombie} zombie  |  {n_mandatory} mandatory")
    print("=" * 70 + "\n")
