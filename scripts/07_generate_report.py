"""
DESCARTES WM — Step 7: Generate Summary Report

Aggregates probing and ablation results across sessions
and hidden sizes into a final classification table.

Usage:
    python scripts/07_generate_report.py [--probe-dir results/probing] \
        [--ablation-dir results/ablation] [--out-dir results]
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict

from descartes_core import classify_variable, print_classification_summary

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_all_results(probe_dir, ablation_dir):
    """Load all probing and ablation results across sessions."""
    all_probe = []
    all_ablation = []

    probe_dir = Path(probe_dir)
    ablation_dir = Path(ablation_dir)

    if probe_dir.exists():
        for f in probe_dir.rglob('probe_level*.json'):
            with open(f) as fh:
                all_probe.extend(json.load(fh))

    if ablation_dir.exists():
        for f in ablation_dir.rglob('ablation_results.json'):
            with open(f) as fh:
                data = json.load(fh)
                # Data is a dict keyed by "level_varname_hN"
                if isinstance(data, dict):
                    for key, val in data.items():
                        all_ablation.append(val)
                elif isinstance(data, list):
                    all_ablation.extend(data)

    return all_probe, all_ablation


def build_classification_table(probe_results, ablation_results):
    """Build final classification for each variable."""
    # Index ablation results by variable name
    ablation_by_var = defaultdict(list)
    for r in ablation_results:
        ablation_by_var[r.get('var_name', '')].append(r)

    classifications = []
    for pr in probe_results:
        var_name = pr.get('var_name', '')

        # Find matching ablation result
        abl = ablation_by_var.get(var_name, [None])[0]

        classification = classify_variable(
            ridge_result=pr,
            ablation_result=abl,
        )
        classification['session'] = pr.get('session', 'unknown')
        classifications.append(classification)

    return classifications


def main():
    parser = argparse.ArgumentParser(description="Generate analysis report")
    parser.add_argument('--probe-dir', type=str, default='results/probing')
    parser.add_argument('--ablation-dir', type=str, default='results/ablation')
    parser.add_argument('--out-dir', type=str, default='results')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_results, ablation_results = load_all_results(
        args.probe_dir, args.ablation_dir,
    )

    logger.info("Loaded %d probe results, %d ablation results",
                len(probe_results), len(ablation_results))

    if not probe_results:
        logger.error("No probe results found. Run 05_run_probing.py first.")
        return

    classifications = build_classification_table(probe_results, ablation_results)

    # Save classifications
    with open(out_dir / 'classifications.json', 'w') as f:
        json.dump(classifications, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print("DESCARTES WM — Classification Summary")
    print("=" * 70)

    print_classification_summary(classifications)

    # Expected outcomes check
    print("\n--- Expected Outcome Checks ---")
    for c in classifications:
        var = c.get('evidence', {}).get('var_name', '')
        cat = c.get('final_category', '')
        if var == 'choice_signal':
            expected = 'MANDATORY'
            status = 'PASS' if 'MANDATORY' in cat else 'FAIL'
            print(f"  choice_signal: {cat} (expected {expected}) [{status}]")
        elif var == 'ramp_signal':
            expected = 'MANDATORY or LEARNED'
            status = 'PASS' if cat != 'ZOMBIE' else 'CHECK'
            print(f"  ramp_signal: {cat} (expected {expected}) [{status}]")

    print(f"\nFull results saved to {out_dir / 'classifications.json'}")


if __name__ == '__main__':
    main()
