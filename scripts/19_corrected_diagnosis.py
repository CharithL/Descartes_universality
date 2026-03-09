#!/usr/bin/env python
"""
Script 19 -- Corrected Diagnosis & Co-occurrence Analysis

Loads existing results (NO re-training) and computes:
  1. Per-patient mandatory count distribution
  2. Variable co-occurrence matrix
  3. Corrected diagnosis (Scenario B)
  4. Cross-circuit comparison table
  5. Saves corrected_diagnosis.json and variable_cooccurrence.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from human_wm.config import RESULTS_DIR

CC_GOOD = 0.5


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    # Load existing results
    audit = load_json(RESULTS_DIR / 'quality_audit.json')
    corrected = load_json(RESULTS_DIR / 'corrected_universality.json')
    matrix = load_json(RESULTS_DIR / 'patient_variable_matrix.json')

    # Key variables (exclude gamma_modulation which is always ZERO_VARIANCE)
    key_vars = ['theta_modulation', 'population_synchrony', 'persistent_delay',
                'delay_stability', 'mean_firing_rate']
    short = {
        'theta_modulation': 'theta',
        'population_synchrony': 'sync',
        'persistent_delay': 'persist',
        'delay_stability': 'delay_st',
        'mean_firing_rate': 'firing',
    }

    # Filter to GOOD patients only
    good_patients = [p for p in matrix if p.get('cc', 0) > CC_GOOD]
    all_patients = matrix

    # =====================================================================
    # 1. Per-patient mandatory count distribution
    # =====================================================================
    print('=' * 75)
    print('1. PER-PATIENT MANDATORY COUNT (GOOD patients, CC > 0.5)')
    print('=' * 75)
    print()

    mand_counts = []
    for p in good_patients:
        n_mand = sum(1 for v in key_vars if p.get(v) == 'MANDATORY')
        mand_counts.append(n_mand)

    # Distribution
    from collections import Counter
    count_dist = Counter(mand_counts)
    max_count = max(mand_counts) if mand_counts else 0

    print(f'{"Mandatory count":<18} {"N patients":>11} {"Percentage":>11}')
    print('-' * 45)
    for i in range(max_count + 1):
        n = count_dist.get(i, 0)
        pct = n / len(good_patients) * 100 if good_patients else 0
        label = f'{i} variables'
        if i == 0:
            label += ' (true zombie)'
        print(f'{label:<18} {n:>11} {pct:>10.0f}%')

    n_with_any = sum(1 for c in mand_counts if c > 0)
    pct_with_any = n_with_any / len(good_patients) * 100 if good_patients else 0
    print()
    print(f'{n_with_any}/{len(good_patients)} good patients '
          f'({pct_with_any:.0f}%) have at least one mandatory variable')

    # =====================================================================
    # 2. Variable co-occurrence matrix
    # =====================================================================
    print()
    print('=' * 75)
    print('2. VARIABLE CO-OCCURRENCE MATRIX (among GOOD patients)')
    print('=' * 75)
    print()

    # Build co-occurrence
    cooccurrence = {}
    for v1 in key_vars:
        cooccurrence[v1] = {}
        for v2 in key_vars:
            count = sum(
                1 for p in good_patients
                if p.get(v1) == 'MANDATORY' and p.get(v2) == 'MANDATORY'
            )
            cooccurrence[v1][v2] = count

    # Print matrix
    header = f'{"":>12}'
    for v in key_vars:
        header += f' {short[v]:>8}'
    print(header)
    print('-' * len(header))
    for v1 in key_vars:
        row = f'{short[v1]:>12}'
        for v2 in key_vars:
            row += f' {cooccurrence[v1][v2]:>8}'
        print(row)

    # =====================================================================
    # 3. Corrected diagnosis
    # =====================================================================
    print()
    print('=' * 75)
    print('3. CORRECTED DIAGNOSIS')
    print('=' * 75)
    print()

    # Gather stats
    n_good = len(good_patients)
    n_total = len(all_patients)
    pct_good = n_good / n_total * 100 if n_total else 0

    # Per-variable mandatory rates among good patients
    var_rates = {}
    for v in key_vars:
        n_mand = sum(1 for p in good_patients if p.get(v) == 'MANDATORY')
        var_rates[v] = {
            'n': n_mand,
            'total': n_good,
            'pct': n_mand / n_good * 100 if n_good else 0,
        }

    max_var = max(var_rates.items(), key=lambda x: x[1]['pct'])
    max_var_name = short[max_var[0]]
    max_var_pct = max_var[1]['pct']

    # Check overlap between top two variables
    theta_ids = set(p['patient_id'] for p in good_patients
                    if p.get('theta_modulation') == 'MANDATORY')
    sync_ids = set(p['patient_id'] for p in good_patients
                   if p.get('population_synchrony') == 'MANDATORY')
    overlap_ids = theta_ids & sync_ids

    print('SCENARIO B: Individual Computational Variability')
    print()
    print(f'  - {n_good}/{n_total} patients had GOOD model quality (CC > {CC_GOOD})')

    # Cross-arch CCs
    cross_arch_dir = RESULTS_DIR / 'cross_architecture'
    arch_ccs = {}
    if cross_arch_dir.exists():
        for arch_dir in sorted(cross_arch_dir.iterdir()):
            if arch_dir.is_dir():
                for rf in arch_dir.glob('results_*.json'):
                    d = load_json(rf)
                    arch_ccs[arch_dir.name] = d.get('cc', 0)

    if arch_ccs:
        min_arch_cc = min(arch_ccs.values())
        print(f'  - All {len(arch_ccs)} architectures learned the task '
              f'(CC > {min_arch_cc:.2f})')

    print(f'  - No single variable is universal '
          f'(max {max_var_pct:.0f}% across good patients: {max_var_name})')
    print(f'  - But {n_with_any}/{n_good} patients ({pct_with_any:.0f}%) '
          f'have at least one mandatory variable')
    print(f'  - Different patients require different mandatory variables')

    for v in key_vars:
        r = var_rates[v]
        if r['n'] > 0:
            print(f'  - {short[v]}: {r["n"]} patients ({r["pct"]:.0f}%)')

    if overlap_ids:
        print(f'  - {len(overlap_ids)} patients overlap on both theta and sync')

    print()
    print('  INTERPRETATION: Working memory computation is individually')
    print('  variable in humans. There is no single "thought" every brain')
    print('  must think. But almost every brain has SOME mandatory')
    print('  computational intermediate -- just a different one.')

    # =====================================================================
    # 4. Cross-circuit comparison table
    # =====================================================================
    print()
    print('=' * 75)
    print('4. CROSS-CIRCUIT COMPARISON TABLE')
    print('=' * 75)
    print()

    fmt = f'{"Circuit":<22} {"Scale":<10} {"Function":<17} ' \
          f'{"% with >=1 mand":>16} {"Universal var?":<20}'
    print(fmt)
    print('-' * 90)
    print(f'{"L5PC":<22} {"Neuron":<10} {"Dendritic":<17} '
          f'{"0% (0/1)":>16} {"No (univ. zombie)":<20}')
    print(f'{"Hippocampal CA3->CA1":<22} {"Circuit":<10} {"Memory encoding":<17} '
          f'{"100% (1/1)*":>16} {"Yes: gamma_amp":<20}')
    print(f'{"Mouse ALM->Thal":<22} {"Circuit":<10} {"Working memory":<17} '
          f'{"100% (2/2)*":>16} {"Yes: theta, choice":<20}')
    print(f'{"Human MTL->Frontal":<22} {"Cognition":<10} {"Working memory":<17} '
          f'{f"{pct_with_any:.0f}% ({n_with_any}/{n_good})":>16} '
          f'{"No: indiv. variable":<20}')
    print()
    print('  * Only tested on 1-2 sessions, not cross-patient')

    # =====================================================================
    # 5. Save results
    # =====================================================================
    print()
    print('=' * 75)
    print('5. SAVING RESULTS')
    print('=' * 75)

    # Corrected diagnosis
    diagnosis_data = {
        'scenario': 'B',
        'scenario_name': 'Individual Computational Variability',
        'n_good_patients': n_good,
        'n_total_patients': n_total,
        'pct_good': pct_good,
        'n_with_any_mandatory': n_with_any,
        'pct_with_any_mandatory': pct_with_any,
        'mandatory_count_distribution': {
            str(k): count_dist.get(k, 0) for k in range(max_count + 1)
        },
        'per_variable_rates_good': {
            short[v]: var_rates[v] for v in key_vars
        },
        'arch_ccs': arch_ccs,
        'theta_mandatory_patients': sorted(theta_ids),
        'sync_mandatory_patients': sorted(sync_ids),
        'theta_sync_overlap': sorted(overlap_ids),
        'interpretation': (
            'Working memory computation is individually variable in humans. '
            'There is no single "thought" every brain must think. But almost '
            'every brain has SOME mandatory computational intermediate -- '
            'just a different one.'
        ),
    }
    with open(RESULTS_DIR / 'corrected_diagnosis.json', 'w') as f:
        json.dump(diagnosis_data, f, indent=2, default=str)
    print(f'  Saved: {RESULTS_DIR / "corrected_diagnosis.json"}')

    # Co-occurrence
    cooccurrence_short = {
        short[v1]: {short[v2]: cooccurrence[v1][v2] for v2 in key_vars}
        for v1 in key_vars
    }
    with open(RESULTS_DIR / 'variable_cooccurrence.json', 'w') as f:
        json.dump(cooccurrence_short, f, indent=2)
    print(f'  Saved: {RESULTS_DIR / "variable_cooccurrence.json"}')

    print()
    print('Done.')


if __name__ == '__main__':
    main()
