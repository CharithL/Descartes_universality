# Design: DESCARTES Human Universality Pipeline

**Date:** 2026-03-07
**Source:** DESCARTES_UNIVERSALITY_GUIDE.md
**Dataset:** Rutishauser Human Single-Neuron WM (DANDI 000576)
**Goal:** Test whether mandatory variables from mouse WM are universal across
human patients, random seeds, and model architectures.

---

## Decisions

1. **Location:** `human_wm/` package inside `Working memory/`, alongside `wm/`.
   Shares `descartes_core` and `pyproject.toml`.

2. **Architectures:** All 4 upfront (LSTM, GRU, Transformer, Linear).

3. **Dataset:** DANDI 000576 only (spikes). LFP from 000673 deferred.
   Theta/gamma targets use spike-derived population rate oscillations.

4. **Ablation strategy:** Architecture-specific forward passes, NOT a
   unified generic function. Rationale: recurrent models require per-timestep
   intervention during manual unrolling to preserve the causal chain.
   Transformers use layer-wise post-hoc intervention. Linear uses direct
   intermediate intervention.
   - `forward_with_resample_recurrent()` — LSTM and GRU (shared, 95% identical)
   - `forward_with_resample_transformer()` — per-layer intervention
   - `forward_with_resample_linear()` — direct projection intervention

5. **NWB column discovery:** `10_human_explore_nwb.py` is a hard prerequisite.
   It outputs a JSON schema file (`nwb_schema.json`) listing exact column names
   for units, trials, brain regions, timing, and trial conditions. All downstream
   code reads from this schema — NO hardcoded column names anywhere.

---

## Package Structure

```
Working memory/
  human_wm/
    __init__.py
    config.py                    # Rutishauser constants + schema loader
    data/
      __init__.py
      nwb_explorer.py            # Discover NWB structure -> nwb_schema.json
      nwb_loader.py              # Extract patient data (MTL/Frontal binning)
      patient_inventory.py       # Build usability inventory (21 patients)
    surrogate/
      __init__.py
      architectures.py           # LSTM, GRU, Transformer, Linear
      train.py                   # Architecture-agnostic training loop
      extract_hidden.py          # Hidden state extraction (all 4 archs)
    targets/
      __init__.py
      persistent_delay.py        # Mean frontal firing during delay
      memory_load.py             # Load-dependent activity (1/2/3 items)
      delay_stability.py         # Temporal autocorrelation through delay
      recognition_decision.py    # Match vs non-match divergence at probe
      concept_selectivity.py     # Image-specific neuron tuning (F-stat)
      oscillatory.py             # Theta/gamma from population rate
      population_sync.py         # Pairwise correlation during delay
    analysis/
      __init__.py
      single_patient.py          # Full pipeline for one patient
      cross_seed.py              # 10 seeds x 1 patient
      cross_patient.py           # All usable patients
      cross_architecture.py      # 4 architectures x 1 patient
      universality_report.py     # Final summary table
    ablation/
      __init__.py
      recurrent.py               # forward_with_resample for LSTM/GRU
      transformer.py             # forward_with_resample for Transformer
      linear.py                  # forward_with_resample for Linear
      dispatch.py                # Route model -> correct ablation fn
  scripts/
    10_human_explore_nwb.py      # HARD PREREQUISITE: discover NWB structure
    11_human_download.py         # Download from DANDI 000576
    12_human_inventory.py        # Build patient inventory
    13_human_single_patient.py   # Proof of concept on best patient
    14_human_cross_seed.py       # Cross-seed test (10 seeds)
    15_human_cross_patient.py    # Cross-patient universality test
    16_human_cross_architecture.py  # Cross-architecture test (4 archs)
    17_human_report.py           # Generate final universality table
```

---

## Data Flow

```
DANDI 000576 NWB files
  |
  v
10_human_explore_nwb.py --> data/nwb_schema.json (column names, regions)
  |
  v
nwb_loader.extract_patient_data(nwb_path, schema)
  --> X (n_trials, n_bins, n_mtl_neurons)    [hippocampus + amygdala]
  --> Y (n_trials, n_bins, n_frontal_neurons) [dACC + pre-SMA + vmPFC]
  --> trial_info dict (load, correct, in_set, timing)
  |
  v
train/val/test split
  |
  v
train_surrogate(model, ...) --> checkpoint .pt
  |
  v
extract_hidden_states(model, X_test) --> hidden_states .npz
  |
  v
For each target in [persistent_delay, memory_load, ...]:
  probe_single_variable(trained_H, untrained_H, target) --> delta_R2
  |
  if delta_R2 > 0.1:
    resample_ablation(model, ..., target) --> z-scores
    classify_variable() --> ZOMBIE / LEARNED_BYPRODUCT / MANDATORY_*
  |
  v
Universality aggregation:
  cross_seed:    N/10 seeds mandatory
  cross_patient: N/21 patients mandatory
  cross_arch:    N/4 architectures mandatory
  |
  v
Final universality table
```

---

## Ablation Architecture

### Recurrent (LSTM/GRU) — Manual unrolling with per-timestep intervention

```python
def forward_with_resample_recurrent(model, test_inputs, clamp_dims,
                                     hidden_states, rng, arch='lstm'):
    # Manual unrolling: at each timestep t
    #   1. Run one step of LSTM/GRU
    #   2. Replace clamp_dims in h_t with random samples
    #   3. h_t propagates to t+1 (causal chain preserved)
    #   4. Readout from intervened h_t
```

### Transformer — Layer-wise post-hoc intervention

```python
def forward_with_resample_transformer(model, test_inputs, clamp_dims,
                                       hidden_states, rng):
    # Full sequence processed, but intervene between layers:
    #   1. input_proj(x) + pos_enc
    #   2. For each transformer layer:
    #      a. Run layer
    #      b. Replace clamp_dims in output
    #   3. Final readout from intervened representation
```

### Linear — Direct intermediate intervention

```python
def forward_with_resample_linear(model, test_inputs, clamp_dims,
                                  hidden_states, rng):
    # No temporal state:
    #   1. h = proj_in(x)
    #   2. Replace clamp_dims in h
    #   3. y = proj_out(h)
```

### Dispatch

```python
def forward_with_resample(model, test_inputs, clamp_dims,
                           hidden_states, rng=None):
    if hasattr(model, 'lstm'):
        return forward_with_resample_recurrent(model, ...)
    elif hasattr(model, 'gru'):
        return forward_with_resample_recurrent(model, ..., arch='gru')
    elif hasattr(model, 'transformer'):
        return forward_with_resample_transformer(model, ...)
    else:
        return forward_with_resample_linear(model, ...)
```

---

## Probe Targets (9 total)

| Target | Level | Source | Returns |
|--------|-------|--------|---------|
| persistent_delay | B | Y delay mean | (n_trials,) |
| memory_load | B | Y delay projected onto load axis | (n_trials,) |
| delay_stability | B | Half-delay correlation | (n_trials,) |
| recognition_decision | B | Match vs non-match divergence | (n_trials,) |
| theta_modulation | C | 4-8 Hz pop-rate amplitude | (n_trials,) |
| gamma_modulation | C | 30-80 Hz pop-rate amplitude | (n_trials,) |
| population_synchrony | C | Pairwise correlation magnitude | (n_trials,) |
| concept_selectivity | A | F-stat across image conditions | (n_trials,) |
| mean_firing_rate | A | Y overall mean | (n_trials,) |

---

## Universality Thresholds

| Test | Metric | Universal | Robust | Fragile |
|------|--------|-----------|--------|---------|
| Cross-Seed | N/10 seeds mandatory | >= 8/10 | >= 5/10 | < 5/10 |
| Cross-Patient | N/21 patients mandatory | >= 80% | >= 50% | < 50% |
| Cross-Architecture | N/4 archs mandatory | 4/4 | >= 2/4 | 1/4 |

---

## Dependencies on descartes_core

- `descartes_core.config` — RIDGE_ALPHAS, CV_FOLDS, thresholds
- `descartes_core.ridge_probe` — probe_single_variable()
- `descartes_core.ablation` — resample_ablation() (for LSTM only; extended
  in human_wm/ablation/ for other architectures)
- `descartes_core.classify` — classify_variable(), print_classification_summary()
- `descartes_core.metrics` — cross_condition_correlation_grouped()
