#!/usr/bin/env python
"""
Script 18 -- Quality Audit & Corrected Universality Report

Diagnoses whether universality failure is due to bad models or real variability.

For every patient tested in cross-patient:
  1. Reports model quality (CC) and data characteristics
  2. Filters to GOOD patients (CC > 0.5) and recomputes universality
  3. Analyzes what predicts which variables are mandatory
  4. Checks cross-architecture CCs
  5. Prints corrected universality table

Saves:
  data/results/quality_audit.json
  data/results/corrected_universality.json
  data/results/patient_variable_matrix.json
"""

import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from human_wm.config import RAW_NWB_DIR, RESULTS_DIR, load_nwb_schema
from human_wm.data.patient_inventory import build_inventory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s  %(message)s',
)
logger = logging.getLogger(__name__)

# Quality thresholds
CC_GOOD = 0.5
CC_MARGINAL = 0.3


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def find_per_patient_results(results_dir):
    """Find all per-patient result JSONs from cross_patient run."""
    cross_patient_dir = Path(results_dir) / 'cross_patient'
    if not cross_patient_dir.exists():
        logger.error('No cross_patient directory found at %s', cross_patient_dir)
        return {}

    patient_results = {}
    for patient_dir in sorted(cross_patient_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        # Find the result JSON (e.g., results_lstm_h128_s0.json)
        result_files = list(patient_dir.glob('results_*.json'))
        if result_files:
            data = load_json(result_files[0])
            if data:
                patient_results[patient_dir.name] = data

    return patient_results


def find_cross_arch_results(results_dir):
    """Find per-architecture result JSONs from cross_architecture run."""
    cross_arch_dir = Path(results_dir) / 'cross_architecture'
    if not cross_arch_dir.exists():
        return {}

    arch_results = {}
    for arch_dir in sorted(cross_arch_dir.iterdir()):
        if not arch_dir.is_dir():
            continue
        result_files = list(arch_dir.glob('results_*.json'))
        if result_files:
            data = load_json(result_files[0])
            if data:
                arch_results[arch_dir.name] = data

    return arch_results


def find_cross_seed_results(results_dir):
    """Find per-seed result JSONs from cross_seed run."""
    cross_seed_dir = Path(results_dir) / 'cross_seed'
    if not cross_seed_dir.exists():
        return {}

    seed_results = {}
    for result_file in sorted(cross_seed_dir.glob('results_*.json')):
        data = load_json(result_file)
        if data:
            seed_key = f"seed_{data.get('seed', result_file.stem)}"
            seed_results[seed_key] = data

    return seed_results


def build_patient_inventory_map(nwb_dir, schema):
    """Build a map from patient_id -> inventory info."""
    nwb_paths = sorted(Path(nwb_dir).glob('**/*.nwb'))
    if not nwb_paths:
        return {}
    inventory = build_inventory(nwb_paths, schema)
    inv_map = {}
    for item in inventory:
        pid = item['patient_id']
        inv_map[pid] = item
    return inv_map


def main():
    schema = load_nwb_schema()

    # =====================================================================
    # PRIORITY 1: Model Quality Audit
    # =====================================================================
    print('=' * 80)
    print('PRIORITY 1: MODEL QUALITY AUDIT')
    print('=' * 80)

    patient_results = find_per_patient_results(RESULTS_DIR)
    if not patient_results:
        print('ERROR: No per-patient results found.')
        print(f'Expected at: {RESULTS_DIR / "cross_patient" / "<patient_id>" / "results_*.json"}')
        sys.exit(1)

    # Build inventory map for neuron counts
    inv_map = build_patient_inventory_map(RAW_NWB_DIR, schema)

    # Collect per-patient data
    audit_rows = []
    for patient_id, result in sorted(patient_results.items()):
        cc = result.get('cc', 0.0)
        if cc > CC_GOOD:
            quality = 'GOOD'
        elif cc > CC_MARGINAL:
            quality = 'MARGINAL'
        else:
            quality = 'FAILED'

        inv = inv_map.get(patient_id, {})
        mtl = inv.get('n_mtl', '?')
        frontal = inv.get('n_frontal', '?')
        trials = inv.get('n_trials', '?')

        # Per-variable classifications
        var_classes = {}
        for var_name, var_data in result.get('variables', {}).items():
            var_classes[var_name] = var_data.get('classification', 'UNKNOWN')

        audit_rows.append({
            'patient_id': patient_id,
            'cc': cc,
            'quality': quality,
            'mtl': mtl,
            'frontal': frontal,
            'trials': trials,
            'variables': var_classes,
        })

    # Print audit table
    print()
    print(f'{"Patient":<25} {"MTL":>5} {"Front":>6} {"Trials":>7} '
          f'{"CC":>7} {"Quality":<10}')
    print('-' * 70)
    for row in audit_rows:
        print(f'{row["patient_id"]:<25} {str(row["mtl"]):>5} '
              f'{str(row["frontal"]):>6} {str(row["trials"]):>7} '
              f'{row["cc"]:>7.3f} {row["quality"]:<10}')

    # Summary
    good_patients = [r for r in audit_rows if r['quality'] == 'GOOD']
    marginal_patients = [r for r in audit_rows if r['quality'] == 'MARGINAL']
    failed_patients = [r for r in audit_rows if r['quality'] == 'FAILED']

    print()
    print(f'Total patients tested: {len(audit_rows)}')
    print(f'  GOOD (CC > {CC_GOOD}):     {len(good_patients):>3}  '
          f'{[r["patient_id"] for r in good_patients]}')
    print(f'  MARGINAL ({CC_MARGINAL}-{CC_GOOD}): {len(marginal_patients):>3}  '
          f'{[r["patient_id"] for r in marginal_patients]}')
    print(f'  FAILED (CC < {CC_MARGINAL}):  {len(failed_patients):>3}  '
          f'{[r["patient_id"] for r in failed_patients]}')

    # =====================================================================
    # PRIORITY 2: Recompute Universality on GOOD Patients Only
    # =====================================================================
    print()
    print('=' * 80)
    print('PRIORITY 2: CORRECTED UNIVERSALITY (GOOD patients only, CC > 0.5)')
    print('=' * 80)

    # Collect all variable names
    all_vars = set()
    for row in audit_rows:
        all_vars.update(row['variables'].keys())
    all_vars = sorted(all_vars)

    # Original counts (all patients)
    print()
    print(f'{"Variable":<25} {"All patients":<18} {"GOOD only":<18} '
          f'{"Change"}')
    print('-' * 75)

    corrected = {}
    for var_name in all_vars:
        n_all = sum(1 for r in audit_rows
                    if r['variables'].get(var_name) == 'MANDATORY')
        total_all = sum(1 for r in audit_rows
                        if var_name in r['variables'])
        pct_all = (n_all / total_all * 100) if total_all > 0 else 0

        n_good = sum(1 for r in good_patients
                     if r['variables'].get(var_name) == 'MANDATORY')
        total_good = sum(1 for r in good_patients
                         if var_name in r['variables'])
        pct_good = (n_good / total_good * 100) if total_good > 0 else 0

        change = pct_good - pct_all
        change_str = f'+{change:.0f}%' if change > 0 else f'{change:.0f}%'

        all_str = f'{n_all}/{total_all} ({pct_all:.0f}%)'
        good_str = f'{n_good}/{total_good} ({pct_good:.0f}%)'

        corrected[var_name] = {
            'n_mandatory_all': n_all,
            'n_total_all': total_all,
            'pct_all': pct_all,
            'n_mandatory_good': n_good,
            'n_total_good': total_good,
            'pct_good': pct_good,
        }

        print(f'{var_name:<25} {all_str:<18} {good_str:<18} {change_str}')

    # =====================================================================
    # PRIORITY 3: What Predicts Mandatory Variables?
    # =====================================================================
    print()
    print('=' * 80)
    print('PRIORITY 3: WHAT PREDICTS MANDATORY VARIABLES?')
    print('=' * 80)

    # 3a: MTL neuron count
    print()
    print('--- 3a: MTL neuron count vs mandatory count ---')
    high_mtl = [r for r in audit_rows
                if isinstance(r['mtl'], (int, float)) and r['mtl'] > 30]
    low_mtl = [r for r in audit_rows
               if isinstance(r['mtl'], (int, float)) and r['mtl'] <= 30]
    for label, group in [('MTL > 30', high_mtl), ('MTL <= 30', low_mtl)]:
        n_mand = sum(
            sum(1 for v in r['variables'].values() if v == 'MANDATORY')
            for r in group
        )
        avg = n_mand / len(group) if group else 0
        avg_cc = np.mean([r['cc'] for r in group]) if group else 0
        print(f'  {label}: {len(group)} patients, '
              f'{n_mand} total mandatory vars, '
              f'avg {avg:.1f} per patient, avg CC={avg_cc:.3f}')

    # 3b: Frontal neuron count
    print()
    print('--- 3b: Frontal neuron count vs mandatory count ---')
    high_frontal = [r for r in audit_rows
                    if isinstance(r['frontal'], (int, float)) and r['frontal'] > 20]
    low_frontal = [r for r in audit_rows
                   if isinstance(r['frontal'], (int, float)) and r['frontal'] <= 20]
    for label, group in [('Frontal > 20', high_frontal), ('Frontal <= 20', low_frontal)]:
        n_mand = sum(
            sum(1 for v in r['variables'].values() if v == 'MANDATORY')
            for r in group
        )
        avg = n_mand / len(group) if group else 0
        avg_cc = np.mean([r['cc'] for r in group]) if group else 0
        print(f'  {label}: {len(group)} patients, '
              f'{n_mand} total mandatory vars, '
              f'avg {avg:.1f} per patient, avg CC={avg_cc:.3f}')

    # 3c: Trial count
    print()
    print('--- 3c: Trial count vs mandatory count ---')
    high_trials = [r for r in audit_rows
                   if isinstance(r['trials'], (int, float)) and r['trials'] > 100]
    low_trials = [r for r in audit_rows
                  if isinstance(r['trials'], (int, float)) and r['trials'] <= 100]
    for label, group in [('Trials > 100', high_trials), ('Trials <= 100', low_trials)]:
        n_mand = sum(
            sum(1 for v in r['variables'].values() if v == 'MANDATORY')
            for r in group
        )
        avg = n_mand / len(group) if group else 0
        avg_cc = np.mean([r['cc'] for r in group]) if group else 0
        print(f'  {label}: {len(group)} patients, '
              f'{n_mand} total mandatory vars, '
              f'avg {avg:.1f} per patient, avg CC={avg_cc:.3f}')

    # 3d: Theta-mandatory patients
    print()
    print('--- 3d: Theta-mandatory patients ---')
    theta_mand = [r for r in audit_rows
                  if r['variables'].get('theta_modulation') == 'MANDATORY']
    theta_not = [r for r in audit_rows
                 if r['variables'].get('theta_modulation') != 'MANDATORY'
                 and 'theta_modulation' in r['variables']]
    print(f'  Theta MANDATORY ({len(theta_mand)} patients):')
    for r in theta_mand:
        print(f'    {r["patient_id"]:<20} MTL={str(r["mtl"]):>3}  '
              f'Frontal={str(r["frontal"]):>3}  Trials={str(r["trials"]):>4}  '
              f'CC={r["cc"]:.3f}')
    if theta_mand:
        avg_cc_mand = np.mean([r['cc'] for r in theta_mand])
        avg_mtl_mand = np.mean([r['mtl'] for r in theta_mand
                                if isinstance(r['mtl'], (int, float))])
        print(f'    Average: CC={avg_cc_mand:.3f}, MTL={avg_mtl_mand:.1f}')
    print(f'  Theta NOT mandatory ({len(theta_not)} patients):')
    if theta_not:
        avg_cc_not = np.mean([r['cc'] for r in theta_not])
        avg_mtl_not = np.mean([r['mtl'] for r in theta_not
                               if isinstance(r['mtl'], (int, float))])
        print(f'    Average: CC={avg_cc_not:.3f}, MTL={avg_mtl_not:.1f}')

    # 3e: Patient x Variable mandatory matrix
    print()
    print('--- 3e: Patient x Variable Mandatory Matrix ---')
    # Determine key variables to show
    key_vars = ['theta_modulation', 'population_synchrony', 'persistent_delay',
                'delay_stability', 'mean_firing_rate', 'gamma_modulation']
    # Only show variables that exist in results
    key_vars = [v for v in key_vars if v in all_vars]

    # Short names for display
    short_names = {
        'theta_modulation': 'theta',
        'population_synchrony': 'sync',
        'persistent_delay': 'persist',
        'delay_stability': 'delay_st',
        'mean_firing_rate': 'firing',
        'gamma_modulation': 'gamma',
    }

    # Header
    header = f'{"Patient":<20} {"CC":>5}'
    for v in key_vars:
        header += f' {short_names.get(v, v[:8]):>8}'
    print(header)
    print('-' * len(header))

    matrix_data = []
    for row in sorted(audit_rows, key=lambda r: -r['cc']):
        line = f'{row["patient_id"]:<20} {row["cc"]:>5.3f}'
        patient_row = {'patient_id': row['patient_id'], 'cc': row['cc']}
        for v in key_vars:
            cls = row['variables'].get(v, '?')
            if cls == 'MANDATORY':
                symbol = 'MAND'
            elif cls == 'LEARNED_ZOMBIE':
                symbol = 'L_ZMB'
            elif cls == 'ZOMBIE':
                symbol = 'zmb'
            elif cls == 'ZERO_VARIANCE':
                symbol = 'ZV'
            elif cls == 'SKIPPED':
                symbol = 'skip'
            else:
                symbol = cls[:5]
            line += f' {symbol:>8}'
            patient_row[v] = cls
        print(line)
        matrix_data.append(patient_row)

    # Overlap analysis
    print()
    sync_mand = set(r['patient_id'] for r in audit_rows
                    if r['variables'].get('population_synchrony') == 'MANDATORY')
    theta_mand_ids = set(r['patient_id'] for r in theta_mand)
    overlap = theta_mand_ids & sync_mand
    print(f'Theta-mandatory patients:  {theta_mand_ids}')
    print(f'Sync-mandatory patients:   {sync_mand}')
    print(f'Overlap:                   {overlap if overlap else "NONE (disjoint)"}')

    # =====================================================================
    # PRIORITY 4: Cross-Architecture CC Comparison
    # =====================================================================
    print()
    print('=' * 80)
    print('PRIORITY 4: CROSS-ARCHITECTURE CC COMPARISON')
    print('=' * 80)

    arch_results = find_cross_arch_results(RESULTS_DIR)
    if arch_results:
        print()
        print(f'{"Architecture":<15} {"CC":>7} {"Can learn?":>12}')
        print('-' * 40)
        good_archs = []
        for arch_name, result in sorted(arch_results.items()):
            cc = result.get('cc', 0.0)
            can_learn = 'YES' if cc > CC_MARGINAL else 'NO'
            if cc > CC_MARGINAL:
                good_archs.append(arch_name)
            print(f'{arch_name:<15} {cc:>7.3f} {can_learn:>12}')

        # Recompute cross-arch for valid architectures only
        print()
        if len(good_archs) < len(arch_results):
            print(f'Valid architectures (CC > {CC_MARGINAL}): {good_archs}')
            print(f'Invalid (never learned): '
                  f'{[a for a in arch_results if a not in good_archs]}')
            print()
            print('Corrected cross-architecture (valid archs only):')
            for var_name in sorted(all_vars):
                n_mand = sum(
                    1 for arch in good_archs
                    if arch_results[arch].get('variables', {}).get(var_name, {}).get(
                        'classification') == 'MANDATORY'
                )
                print(f'  {var_name:<25} {n_mand}/{len(good_archs)}')
        else:
            print(f'All architectures learned the task (CC > {CC_MARGINAL})')
    else:
        print('No cross-architecture results found.')

    # =====================================================================
    # PRIORITY 5: Corrected Universality Table
    # =====================================================================
    print()
    print('=' * 80)
    print('CORRECTED UNIVERSALITY REPORT (GOOD patients only, CC > 0.5)')
    print('=' * 80)

    # Load cross-seed summary
    cross_seed_summary = load_json(
        RESULTS_DIR / 'cross_seed' / 'cross_seed_summary.json')

    # Determine good architecture count
    n_good_arch = len(good_archs) if arch_results else 0
    n_total_arch = len(arch_results) if arch_results else 0

    n_good_patients = len(good_patients)

    print()
    header = (f'{"Variable":<25} | {"Cross-Seed":<14} | '
              f'{"Cross-Patient":<18} | {"Cross-Arch":<14} | VERDICT')
    print(header)
    n_seeds = 10
    if cross_seed_summary:
        n_seeds = cross_seed_summary.get('successful_seeds', 10)
    subheader = (f'{"":25} | {"(N/" + str(n_seeds) + ")":<14} | '
                 f'{"(N/" + str(n_good_patients) + " good)":<18} | '
                 f'{"(N/" + str(n_good_arch) + " good)":<14} |')
    print(subheader)
    print('-' * 85)

    final_verdicts = {}
    for var_name in sorted(all_vars):
        # Cross-seed
        if cross_seed_summary and var_name in cross_seed_summary.get('variables', {}):
            cs = cross_seed_summary['variables'][var_name]
            n_seed_mand = cs['n_mandatory']
            n_seed_total = cs['n_total']
            seed_str = f'{n_seed_mand}/{n_seed_total}'
            seed_robust = n_seed_mand >= 8
        else:
            seed_str = '?'
            seed_robust = False

        # Cross-patient (GOOD only)
        n_good_mand = corrected[var_name]['n_mandatory_good'] if var_name in corrected else 0
        n_good_total = corrected[var_name]['n_total_good'] if var_name in corrected else 0
        pct_good = corrected[var_name]['pct_good'] if var_name in corrected else 0
        patient_str = f'{n_good_mand}/{n_good_total} ({pct_good:.0f}%)'
        patient_universal = pct_good >= 80
        patient_robust = pct_good >= 50

        # Cross-architecture (good archs only)
        if arch_results and good_archs:
            n_arch_mand = sum(
                1 for arch in good_archs
                if arch_results[arch].get('variables', {}).get(var_name, {}).get(
                    'classification') == 'MANDATORY'
            )
            arch_str = f'{n_arch_mand}/{len(good_archs)}'
            arch_universal = n_arch_mand >= len(good_archs) - 1 and len(good_archs) >= 3
        else:
            arch_str = '?'
            arch_universal = False

        # Determine verdict
        if seed_robust and patient_universal and arch_universal:
            verdict = 'UNIVERSAL'
        elif seed_robust and patient_robust:
            verdict = 'ROBUST'
        elif seed_robust or patient_robust:
            verdict = 'PARTIAL'
        elif pct_good < 50 and n_good_total >= 5:
            verdict = 'INDIVIDUAL'
        else:
            verdict = 'ZOMBIE'

        final_verdicts[var_name] = verdict

        line = (f'{var_name:<25} | {seed_str:<14} | {patient_str:<18} | '
                f'{arch_str:<14} | {verdict}')
        print(line)

    print()
    print('VERDICT KEY (among GOOD models only):')
    print('  UNIVERSAL:   >=8/10 seeds AND >=80% good patients AND >=3/K good architectures')
    print('  ROBUST:      >=8/10 seeds AND >=50% good patients')
    print('  PARTIAL:     Some evidence but not consistent across tests')
    print('  INDIVIDUAL:  <50% good patients despite good model quality')
    print('  ZOMBIE:      Not mandatory anywhere')

    # =====================================================================
    # Diagnosis: Which Scenario?
    # =====================================================================
    print()
    print('=' * 80)
    print('DIAGNOSIS')
    print('=' * 80)
    print()

    pct_failed = len(failed_patients) / len(audit_rows) * 100 if audit_rows else 0
    pct_good_ratio = len(good_patients) / len(audit_rows) * 100 if audit_rows else 0

    if pct_failed > 60:
        print('SCENARIO A: Most patients had bad models')
        print(f'  {pct_failed:.0f}% of patients had CC < {CC_MARGINAL} (FAILED)')
        print('  The universality results are dominated by noise from bad models.')
        print('  Look at the GOOD-only rates above for the real picture.')
        scenario = 'A'
    elif pct_good_ratio > 60:
        # Check if any variable is universal among good patients
        any_robust = any(v for v in final_verdicts.values()
                         if v in ('UNIVERSAL', 'ROBUST'))
        if any_robust:
            print('SCENARIO B: Models are fine, real variability exists')
            print(f'  {pct_good_ratio:.0f}% of patients had CC > {CC_GOOD} (GOOD)')
            print('  Some variables are robust but not all, suggesting individual variability.')
            scenario = 'B'
        else:
            print('SCENARIO C: Nothing is mandatory even in good patients')
            print(f'  {pct_good_ratio:.0f}% of patients had good models')
            print('  But zero variables reach 50% mandatory rate among good patients.')
            print('  The human MTL->frontal transformation may be a universal learned zombie.')
            scenario = 'C'
    else:
        print('MIXED: Moderate model quality with mixed results')
        print(f'  GOOD: {len(good_patients)}, MARGINAL: {len(marginal_patients)}, '
              f'FAILED: {len(failed_patients)}')
        scenario = 'MIXED'

    # =====================================================================
    # Save Results
    # =====================================================================
    save_dir = RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # Quality audit
    audit_data = {
        'total_patients': len(audit_rows),
        'n_good': len(good_patients),
        'n_marginal': len(marginal_patients),
        'n_failed': len(failed_patients),
        'cc_threshold_good': CC_GOOD,
        'cc_threshold_marginal': CC_MARGINAL,
        'patients': [{
            'patient_id': r['patient_id'],
            'cc': r['cc'],
            'quality': r['quality'],
            'mtl': r['mtl'],
            'frontal': r['frontal'],
            'trials': r['trials'],
        } for r in audit_rows],
        'scenario': scenario,
    }
    with open(save_dir / 'quality_audit.json', 'w') as f:
        json.dump(audit_data, f, indent=2, default=str)

    # Corrected universality
    corrected_data = {
        'n_good_patients': n_good_patients,
        'n_good_architectures': n_good_arch,
        'variables': {},
    }
    for var_name in sorted(all_vars):
        c = corrected.get(var_name, {})
        corrected_data['variables'][var_name] = {
            'all_patients': f'{c.get("n_mandatory_all", 0)}/{c.get("n_total_all", 0)}',
            'pct_all': c.get('pct_all', 0),
            'good_patients': f'{c.get("n_mandatory_good", 0)}/{c.get("n_total_good", 0)}',
            'pct_good': c.get('pct_good', 0),
            'verdict': final_verdicts.get(var_name, 'UNKNOWN'),
        }
    with open(save_dir / 'corrected_universality.json', 'w') as f:
        json.dump(corrected_data, f, indent=2)

    # Patient x Variable matrix
    with open(save_dir / 'patient_variable_matrix.json', 'w') as f:
        json.dump(matrix_data, f, indent=2, default=str)

    print()
    print(f'Saved: {save_dir / "quality_audit.json"}')
    print(f'Saved: {save_dir / "corrected_universality.json"}')
    print(f'Saved: {save_dir / "patient_variable_matrix.json"}')


if __name__ == '__main__':
    main()
