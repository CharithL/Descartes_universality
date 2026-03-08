# DESCARTES Working Memory Guide: Chen/Svoboda Thalamocortical Loop

## Comprehensive Guide for Applying the Zombie Test to Cognitive-Level Computation

**Target:** Chen, Svoboda et al. 2024 — Brain-wide memory-guided movement
**Dataset:** DANDI 000363
**Question:** Is persistent delay-period activity a mandatory computational intermediate for working memory, or can an LSTM replicate the ALM→thalamus transformation without it?

---

## 0. Why This Dataset

The Chen/Svoboda dataset is the single best candidate for extending DESCARTES from circuit-level oscillations to cognitive-level computation. Here is why.

The hippocampal experiment tested a population-level transformation (CA3→CA1 memory encoding) and found gamma_amp as the sole mandatory variable. But gamma_amp is an oscillatory property — a rhythm, not a thought. To test whether cognitive variables resist zombification, we need a dataset where the transformation serves a genuine cognitive function with a well-characterised cognitive intermediate.

Working memory is that function. The animal hears a tone, must remember which direction to lick, waits through a delay period, then executes the lick. During the delay, anterior lateral motor cortex (ALM) and thalamus (VM/VAL) maintain persistent activity that encodes the upcoming choice. This persistent activity IS the working memory — it bridges the temporal gap between stimulus and response. If we can show that the LSTM surrogate needs persistent-attractor-like dynamics in its hidden states to replicate this transformation, we've demonstrated that a cognitive computation (memory maintenance) is mathematically irreducible.

The dataset provides five simultaneous Neuropixels probes covering ALM, contralateral ALM, thalamus, striatum, midbrain, cerebellum, and brainstem. The original authors performed ALM photoinhibition showing that silencing ALM collapses thalamic choice coding and vice versa — providing ground-truth causal structure. The data is in NWB format on DANDI with no access restrictions.

The transformation chain is:

```
Auditory tone → ALM encoding → ALM↔thalamus maintenance (DELAY) → ALM motor command → lick
```

We model the ALM→thalamus link during the delay period. The mandatory variable candidate is persistent choice-selective activity. The zombie alternative is that the LSTM finds a non-persistent shortcut — perhaps encoding the choice in a transient burst at delay onset and reading it out at delay offset without maintaining it continuously.

---

## 1. Dataset Specification

**Full name:** Brain-wide neural activity during memory-guided movement
**Authors:** Chen, Hou, Bhatt, McKenzie, Li, Mathis, Svoboda et al. (2024)
**Publication:** Nature (2024) — "Brain-wide neural activity underlying memory-guided movement"
**DANDI URL:** https://dandiarchive.org/dandiset/000363
**License:** CC-BY 4.0
**Format:** Neurodata Without Borders (NWB) via PyNWB/DANDI CLI

### What is recorded

Each session contains simultaneous Neuropixels recordings from up to 5 probes targeting different brain regions. The key regions for our purposes are:

**ALM (Anterior Lateral Motor cortex):** The primary cortical node for memory-guided licking. During the delay period, ALM neurons show persistent choice-selective activity — some fire more for "lick left" trials, others for "lick right." This IS the cortical working memory signal.

**Thalamus (VM, VAL, PO nuclei):** The thalamic relay that cooperates with ALM to maintain persistent activity. Guo et al. (2017) showed that ALM and thalamus form a mutually dependent loop — silencing either one collapses the other's choice coding. The thalamus is not a passive relay; it is an active computational partner.

**Additional regions** (available but not primary targets): contralateral ALM, striatum (caudate, putamen), midbrain (superior colliculus, red nucleus, SNr), cerebellum (dentate, fastigial), brainstem motor nuclei.

### Task structure

```
Time:    0s          0.5s              1.3s        1.6s          2.5s+
         |-----------|-----------------|-----------|-------------|
         | Auditory  |    DELAY        |    Go     |   Response  |
         | tone (L/R)|  (no stimulus)  |   cue     |   (lick)    |
         |-----------|-----------------|-----------|-------------|
                     ↑                              ↑
              Memory formed                    Memory read out
              (ALM encodes choice)             (ALM drives motor)
```

The DELAY period (0.5s to 1.3s) is the critical window. During this 800ms, there is no external stimulus. The animal must maintain the choice representation internally. ALM and thalamus show persistent choice-selective firing throughout this period. This persistent firing is what we probe for.

### Scale

The dataset contains 28 mice, 173 sessions. Each session has hundreds of trials (typically 200-400 correct trials). Neurons per region per session vary but typically yield 50-200+ units in ALM and 30-100+ in thalamus, well above the 20-neuron minimum. Total across all sessions: tens of thousands of neurons.

---

## 2. Data Access and Preprocessing

### 2.1 Download

```bash
pip install dandi pynwb numpy scipy h5py
dandi download https://dandiarchive.org/dandiset/000363

# This downloads ALL sessions (~100+ GB). For initial work, download specific sessions:
dandi download https://dandiarchive.org/dandiset/000363 --glob "sub-*/sub-*_ses-*_ecephys.nwb" --limit 5
```

Alternatively, stream specific sessions without full download:

```python
from dandi.dandiapi import DandiAPIClient
import pynwb
import h5py
from fsspec.implementations.http import HTTPFileSystem

client = DandiAPIClient()
dandiset = client.get_dandiset("000363")
assets = list(dandiset.get_assets())

# List all available sessions
for asset in assets[:20]:
    print(asset.path, f"({asset.size / 1e9:.1f} GB)")
```

### 2.2 Session Selection

Not all sessions contain both ALM and thalamus recordings. We must filter for sessions with simultaneous coverage of both regions. The NWB files contain electrode metadata with brain region annotations tied to the Allen Common Coordinate Framework (CCF).

```python
import pynwb
import numpy as np

def get_regions_in_session(nwb_path):
    """Extract unique brain regions recorded in this session."""
    with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
        nwb = io.read()
        units = nwb.units
        # Region info is in electrode metadata
        # Exact column name may vary — check:
        # units.colnames to see available columns
        regions = set()
        if 'brain_region' in units.colnames:
            regions = set(units['brain_region'].data[:])
        elif 'location' in units.colnames:
            regions = set(units['location'].data[:])
        return regions

def find_alm_thalamus_sessions(data_dir):
    """Find sessions with both ALM and thalamic recordings."""
    import glob
    good_sessions = []
    for nwb_path in glob.glob(f"{data_dir}/**/*.nwb", recursive=True):
        regions = get_regions_in_session(nwb_path)
        has_alm = any('ALM' in r or 'MOs' in r for r in regions)
        has_thal = any('VM' in r or 'VAL' in r or 'VPM' in r or 'PO' in r
                       for r in regions)
        if has_alm and has_thal:
            good_sessions.append((nwb_path, regions))
            print(f"GOOD: {nwb_path}")
            print(f"  Regions: {regions}")
    return good_sessions
```

**Target:** Find at least 10 sessions with ≥30 ALM neurons and ≥20 thalamus neurons simultaneously. This gives us enough data for robust LSTM training with proper train/val/test splits.

### 2.3 Spike Extraction and Trial Alignment

```python
def extract_session_data(nwb_path, region_input='ALM', region_output='thalamus',
                         bin_size_ms=10, delay_start_s=0.5, delay_end_s=1.3):
    """
    Extract binned spike counts for input and output regions,
    aligned to trial events.

    Returns:
        X: (n_trials, n_timesteps, n_input_neurons)  — ALM during delay
        Y: (n_trials, n_timesteps, n_output_neurons)  — Thalamus during delay
        trial_types: (n_trials,)  — 0=lick_left, 1=lick_right
    """
    with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
        nwb = io.read()

        # Get unit spike times and region labels
        units = nwb.units
        n_units = len(units)
        spike_times = [units['spike_times'][i] for i in range(n_units)]
        regions = [units['brain_region'][i] for i in range(n_units)]

        # Separate input (ALM) and output (thalamus) neurons
        alm_idx = [i for i, r in enumerate(regions)
                    if 'ALM' in r or 'MOs' in r]
        thal_idx = [i for i, r in enumerate(regions)
                    if any(t in r for t in ['VM', 'VAL', 'VPM', 'PO'])]

        # Get trial information
        # NWB stores trials as an intervals table
        trials = nwb.trials
        n_trials = len(trials)

        # Extract trial timing — column names depend on the specific NWB file
        # Common columns: 'start_time', 'stop_time', 'stimulus', 'response'
        # Check trials.colnames for exact available columns
        trial_starts = trials['start_time'].data[:]
        trial_types_raw = trials['trial_type'].data[:]  # or similar column

        # Bin spikes during delay period
        delay_duration = delay_end_s - delay_start_s
        n_bins = int(delay_duration / (bin_size_ms / 1000))

        X = np.zeros((n_trials, n_bins, len(alm_idx)))
        Y = np.zeros((n_trials, n_bins, len(thal_idx)))
        trial_types = np.zeros(n_trials, dtype=int)

        for t in range(n_trials):
            t_start = trial_starts[t] + delay_start_s
            for b in range(n_bins):
                bin_start = t_start + b * (bin_size_ms / 1000)
                bin_end = bin_start + (bin_size_ms / 1000)

                for j, idx in enumerate(alm_idx):
                    spikes = spike_times[idx]
                    X[t, b, j] = np.sum((spikes >= bin_start) & (spikes < bin_end))

                for j, idx in enumerate(thal_idx):
                    spikes = spike_times[idx]
                    Y[t, b, j] = np.sum((spikes >= bin_start) & (spikes < bin_end))

            trial_types[t] = trial_types_raw[t]

    return X, Y, trial_types
```

**Critical note on the NWB structure:** The exact column names and trial structure will vary. The code above is a template. Run `print(nwb.trials.colnames)` and `print(nwb.units.colnames)` on the first downloaded file to discover the exact field names. The Chen/Svoboda dataset follows the Svoboda lab conventions, which may use custom column names like `'trial_instruction'` or `'response_direction'` rather than generic `'trial_type'`.

### 2.4 Trial Filtering

```python
def filter_correct_trials(X, Y, trial_types, outcomes):
    """Keep only correct trials where the animal performed the task."""
    correct_mask = outcomes == 'correct'  # or outcomes == 1
    return X[correct_mask], Y[correct_mask], trial_types[correct_mask]
```

We only use correct trials because incorrect trials may reflect lapses where the animal did not maintain working memory at all. Including them would add noise that doesn't reflect the transformation we're modeling.

### 2.5 Multi-Session Concatenation

Because neuron identities differ across sessions, we cannot simply concatenate. Two approaches:

**Approach A — Per-session models (recommended for initial analysis):**
Train a separate LSTM per session. Probe and ablate per session. Report results across sessions as a distribution. This avoids the neuron-alignment problem entirely and matches the hippocampal pipeline.

**Approach B — Pseudo-population (for later):**
Concatenate neurons across sessions into a single large pseudo-population, using trial-averaging within each session to align the temporal structure. This requires matching trial conditions across sessions.

Start with Approach A. Select the 5 sessions with the most ALM and thalamus neurons.

---

## 3. Known Intermediate Variables to Probe For

### 3.1 Persistent Choice-Selective Activity (PRIMARY TARGET)

**What it is:** During the delay period, a subset of ALM and thalamus neurons fire persistently at elevated rates, and their firing rate differs between lick-left and lick-right trials. This "choice coding" persists throughout the entire 800ms delay with no external stimulus.

**How to compute ground truth:** For each neuron in the biological thalamus recording, compute the mean firing rate during the delay for lick-left trials vs lick-right trials. The choice selectivity is the difference (or the d-prime). At the population level, project the population vector onto the choice-coding axis (first principal component of the trial-type difference) to get a single continuous variable per timestep per trial.

```python
def compute_choice_axis(Y, trial_types):
    """
    Compute the population-level choice signal in thalamus.

    Returns:
        choice_signal: (n_trials, n_timesteps) — projection onto choice axis
        choice_axis: (n_neurons,) — the axis itself
    """
    # Mean population vector for each trial type, averaged over time
    left_mean = Y[trial_types == 0].mean(axis=(0, 1))   # (n_neurons,)
    right_mean = Y[trial_types == 1].mean(axis=(0, 1))  # (n_neurons,)

    # Choice axis = difference of means, normalized
    choice_axis = right_mean - left_mean
    choice_axis = choice_axis / (np.linalg.norm(choice_axis) + 1e-8)

    # Project each trial's population vector onto this axis at each timestep
    # Y shape: (n_trials, n_timesteps, n_neurons)
    choice_signal = np.einsum('tbn,n->tb', Y, choice_axis)

    return choice_signal, choice_axis
```

**What makes it "persistent":** The choice_signal should be approximately constant throughout the delay period, not just a transient burst at the start. Compute the temporal autocorrelation — if it's high (>0.8 across the full delay), the signal is truly persistent.

**Why it might be mandatory:** The ALM-thalamus loop must maintain the choice representation across the 800ms gap. If the LSTM cannot predict thalamic activity during late delay without tracking persistent choice in its hidden states, then persistent maintenance is computationally irreducible — the math of bridging a temporal gap requires it.

**Why it might be zombie:** The LSTM might encode the choice in a transient burst at delay onset and store it in its cell state (which is specifically designed for long-term memory) without needing persistent activation in the hidden state. The LSTM's gating mechanism IS a form of persistent memory — but it doesn't manifest as persistent firing. It would be a zombie: correct output, alien internal mechanism.

### 3.2 Ramping Activity (SECONDARY TARGET)

**What it is:** Some ALM neurons show gradually increasing firing rates through the delay, ramping toward a threshold that triggers the motor response at the go cue. This is analogous to the evidence accumulation ramp in decision-making.

**How to compute:** Fit a linear slope to each neuron's delay-period firing rate. Neurons with significant positive or negative slopes are "rampers." The population-level ramp is the projection onto the first temporal principal component.

```python
def compute_ramp_signal(Y):
    """Extract the temporal ramp component from thalamic population."""
    # Average across trials
    Y_mean = Y.mean(axis=0)  # (n_timesteps, n_neurons)

    # Fit linear slope to each neuron's temporal profile
    n_t = Y_mean.shape[0]
    time = np.arange(n_t, dtype=float)
    time = (time - time.mean()) / (time.std() + 1e-8)

    slopes = np.array([np.polyfit(time, Y_mean[:, n], 1)[0]
                       for n in range(Y_mean.shape[1])])

    # Ramp axis = neurons weighted by their slope
    ramp_axis = slopes / (np.linalg.norm(slopes) + 1e-8)

    # Project each trial onto ramp axis
    ramp_signal = np.einsum('tbn,n->tb', Y, ramp_axis)
    return ramp_signal, ramp_axis
```

### 3.3 Preparatory-to-Execution Transition Signal (TERTIARY TARGET)

**What it is:** At the go cue, ALM switches from a "preparatory" regime to an "execution" regime. The dynamics rotate from a null-space (preparation doesn't drive movement) to an output-potent space (execution drives motor neurons). This is the condition-invariant signal (CIS) from Kaufman et al. (2016).

**How to compute:** PCA on the full trial (including pre-go and post-go periods). The CIS is the principal component that is constant across conditions before the go cue but diverges after. We only use this as a probe target if we extend the analysis window beyond the delay period.

### 3.4 Theta-Band Modulation (EXPLORATORY)

**What it is:** ALM-thalamus communication during the delay may be modulated by low-frequency oscillations (4-8 Hz theta). If the LSTM discovers theta-like periodic modulation in its hidden states, that would parallel the hippocampal gamma_amp finding.

**How to compute:** Bandpass filter the population rate (sum of all neurons) at 4-8 Hz. Extract instantaneous amplitude and phase via Hilbert transform.

---

## 4. LSTM Surrogate Architecture

### 4.1 Model

```python
import torch
import torch.nn as nn

class WMSurrogate(nn.Module):
    """
    LSTM surrogate for the ALM → Thalamus transformation
    during working memory delay.

    Input: ALM population activity (n_alm_neurons per timestep)
    Output: Thalamus population activity (n_thal_neurons per timestep)
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, n_layers=2,
                 dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_hidden=False):
        """
        x: (batch, timesteps, n_alm_neurons)
        returns: (batch, timesteps, n_thal_neurons)
        """
        h_seq, (h_n, c_n) = self.lstm(x)
        y_pred = self.output_proj(h_seq)

        if return_hidden:
            return y_pred, h_seq  # h_seq is what we probe
        return y_pred
```

### 4.2 Hidden Size Sweep

Run three hidden sizes to test the bottleneck pattern:

```
h=64:   Compressed — forced to select which variables to encode
h=128:  Sweet spot — enough capacity for rich representation
h=256:  Overcapacity — may develop superposition or zombie encoding
```

The hippocampal experiment showed h=128 as the sweet spot. The L5PC showed the same. We predict h=128 will be richest here too.

### 4.3 Training

```python
def train_surrogate(model, train_loader, val_loader, n_epochs=200,
                    lr=1e-3, patience=20):
    """Standard training with early stopping and cosine annealing."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                Y_pred = model(X_batch)
                val_loss += criterion(Y_pred, Y_batch).item()

        scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load('best_model.pt'))
    return model
```

### 4.4 Quality Check: Cross-Condition Correlation

Before probing, verify the model actually learned the transformation:

```python
def cross_condition_correlation(model, X_test, Y_test, trial_types):
    """
    Compute CC between predicted and actual thalamic activity,
    averaged across conditions.

    This is the same metric used in the hippocampal and L5PC experiments.
    """
    model.eval()
    with torch.no_grad():
        Y_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    # Average over trials per condition
    conditions = np.unique(trial_types)
    pred_means = []
    true_means = []
    for c in conditions:
        mask = trial_types == c
        pred_means.append(Y_pred[mask].mean(axis=0).flatten())
        true_means.append(Y_test[mask].mean(axis=0).flatten())

    pred_concat = np.concatenate(pred_means)
    true_concat = np.concatenate(true_means)

    cc = np.corrcoef(pred_concat, true_concat)[0, 1]
    return cc
```

**Threshold:** CC > 0.5 means the model learned something meaningful. CC > 0.7 means strong performance. CC < 0.3 means the model failed and probing results are unreliable (same issue as h=256 in the L5PC experiment).

---

## 5. Three-Level Probing

Following the L5PC guide structure, we probe at three levels. But here the levels are COGNITIVE, not biophysical.

### Level A — Single-Neuron Selectivity (expected mostly zombie)

Probe the LSTM hidden states for individual thalamic neuron tuning properties. For each thalamic neuron, compute its choice selectivity index (CSI = d-prime between left and right trials). Then ask: does Ridge regression from LSTM hidden states predict each neuron's CSI across timesteps?

This parallels Level A in L5PC (individual gating variables). Prediction: mostly zombie, because individual neuron tuning is a substrate detail, not a computational requirement.

### Level B — Population-Level Cognitive Variables (THE KEY LEVEL)

These are the variables that matter:

```python
probe_targets_level_B = {
    'choice_signal':      choice_signal,       # Persistent choice axis projection
    'ramp_signal':        ramp_signal,          # Temporal ramp toward go cue
    'population_rate':    Y.sum(axis=2),        # Total thalamic firing rate
    'choice_magnitude':   np.abs(choice_signal),# Strength of choice encoding
    'cross_condition_var': cross_cond_variance,  # How different are the conditions?
}
```

**Ridge ΔR² methodology (identical to hippocampal/L5PC):**

```python
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

def probe_single_variable(hidden_states, target, n_folds=5):
    """
    Probe hidden states for a single target variable.

    hidden_states: (n_trials, hidden_dim) — trial-averaged last-layer hidden
    target: (n_trials,) — trial-averaged target variable

    Returns: R2_trained (float)
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(hidden_states)
    y = target

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    r2_scores = []

    for train_idx, test_idx in kf.split(X):
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 10))
        ridge.fit(X[train_idx], y[train_idx])
        r2_scores.append(ridge.score(X[test_idx], y[test_idx]))

    return np.mean(r2_scores)
```

**Untrained baseline:** Train an identical LSTM with random weights (no gradient updates). Extract hidden states. Probe with the same Ridge regression. ΔR² = R²_trained - R²_untrained. Only ΔR² > 0.1 counts as "learned."

### Level C — Emergent Dynamics (gamma_amp equivalents)

These probe for dynamical properties of the thalamic population that emerge from the interaction of many neurons:

```python
probe_targets_level_C = {
    'delay_stability':     stability_index,     # How constant is choice coding?
    'theta_modulation':    theta_amplitude,      # 4-8 Hz oscillatory amplitude
    'population_synchrony': sync_index,          # Pairwise correlation magnitude
    'attractor_depth':     attractor_metric,     # Basin depth of choice attractor
    'transition_sharpness': go_cue_response,     # How fast does the state change at go?
}
```

The `delay_stability` is particularly important — it measures whether the choice representation is truly persistent (high stability) or drifting/decaying (low stability). If the LSTM learns to encode delay stability AND it's mandatory, that proves the LSTM discovered that persistent maintenance is computationally required.

---

## 6. Resample Ablation (NOT Mean-Clamping)

The L5PC experiment proved that mean-clamping is unreliable — it creates OOD artifacts that mimic causality. ALL ablation in this experiment uses resample ablation exclusively.

### 6.1 Resample Ablation Protocol

```python
def resample_ablation(model, X_test, Y_test, trial_types,
                      hidden_states, target_variable,
                      k_fractions=[0.05, 0.10, 0.20, 0.40, 0.60, 0.80],
                      n_repeats=10):
    """
    For each k-fraction:
    1. Identify top-k dimensions correlated with target_variable
    2. RESAMPLE those dimensions from their empirical distribution
       (preserves marginal statistics, breaks specific correlation)
    3. Compute cross-condition correlation of the modified output
    4. Compare against random-dimension resampling (control)
    5. Report z-score

    A z-score < -2 means the target variable is CAUSALLY USED.
    """
    hidden_dim = hidden_states.shape[1]
    baseline_cc = cross_condition_correlation(model, X_test, Y_test, trial_types)

    # Correlations between each hidden dim and the target
    correlations = np.array([
        np.abs(np.corrcoef(hidden_states[:, d], target_variable)[0, 1])
        for d in range(hidden_dim)
    ])

    results = {}
    for frac in k_fractions:
        k = max(1, int(frac * hidden_dim))

        # Top-k target-correlated dimensions
        target_dims = np.argsort(correlations)[-k:]

        # Resample ablation: replace each clamped dim with a random
        # sample from that dim's empirical distribution
        target_ccs = []
        for _ in range(n_repeats):
            cc = forward_with_resample(model, X_test, Y_test, trial_types,
                                        target_dims, hidden_states)
            target_ccs.append(cc)

        # Random-dimension control
        random_ccs = []
        for _ in range(n_repeats):
            rand_dims = np.random.choice(hidden_dim, k, replace=False)
            cc = forward_with_resample(model, X_test, Y_test, trial_types,
                                        rand_dims, hidden_states)
            random_ccs.append(cc)

        target_mean = np.mean(target_ccs)
        random_mean = np.mean(random_ccs)
        random_std = np.std(random_ccs) + 1e-8
        z = (target_mean - random_mean) / random_std

        results[f'k={frac:.0%}'] = {
            'target_cc': target_mean,
            'random_cc': random_mean,
            'z_score': z,
            'causal': z < -2.0
        }

    return results
```

### 6.2 The Forward-with-Resample Hook

```python
def forward_with_resample(model, X_test, Y_test, trial_types,
                           clamp_dims, hidden_states):
    """
    Run model forward but at each timestep, replace the specified
    hidden dimensions with random samples from their empirical
    distribution (across all trials at that timestep).

    This preserves marginal statistics while destroying the specific
    temporal correlation with the target variable.
    """
    model.eval()
    hidden_dim = model.lstm.hidden_size
    n_layers = model.lstm.num_layers

    # Collect empirical distributions per dim per timestep
    # from the unmodified forward pass
    with torch.no_grad():
        _, h_seq_clean = model(torch.tensor(X_test, dtype=torch.float32),
                                return_hidden=True)
        h_clean = h_seq_clean.numpy()  # (batch, time, hidden)

    # Modified forward pass with resampling
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    batch_size, seq_len, _ = X_tensor.shape

    # Manual LSTM unrolling with intervention
    h_t = torch.zeros(n_layers, batch_size, hidden_dim)
    c_t = torch.zeros(n_layers, batch_size, hidden_dim)
    outputs = []

    for t in range(seq_len):
        x_t = X_tensor[:, t:t+1, :]
        out_t, (h_t, c_t) = model.lstm(x_t, (h_t, c_t))

        # Intervene on last layer's hidden state
        h_last = h_t[-1].clone()  # (batch, hidden)
        for d in clamp_dims:
            # Sample from empirical distribution of dim d at timestep t
            empirical = h_clean[:, t, d]
            samples = np.random.choice(empirical, size=batch_size, replace=True)
            h_last[:, d] = torch.tensor(samples, dtype=torch.float32)
        h_t[-1] = h_last

        output = model.output_proj(h_last.unsqueeze(1))
        outputs.append(output)

    Y_pred = torch.cat(outputs, dim=1).detach().numpy()

    # Compute cross-condition correlation
    conditions = np.unique(trial_types)
    pred_means = [Y_pred[trial_types == c].mean(0).flatten() for c in conditions]
    true_means = [Y_test[trial_types == c].mean(0).flatten() for c in conditions]
    cc = np.corrcoef(np.concatenate(pred_means),
                     np.concatenate(true_means))[0, 1]
    return cc
```

---

## 7. Expected Outcomes and Interpretation

### Outcome A — Persistent choice signal is MANDATORY (z < -2 under resample)

This means the LSTM cannot predict thalamic delay-period activity without maintaining a persistent choice representation in its hidden states. The temporal gap REQUIRES continuous tracking of which choice the animal will make. This is the strongest possible result:

**Implication:** Working memory persistence is computationally irreducible. Any system that bridges a temporal gap in a choice task — biological or artificial — must maintain a persistent representation. This extends gamma_amp (mandatory oscillation for memory encoding) to persistent activity (mandatory attractor for memory maintenance). The zombie boundary lies between single-neuron biophysics (fully zombie) and population-level cognitive dynamics (mandatory).

**For consciousness:** If a prosthesis replaces the ALM-thalamus loop, it MUST implement persistent maintenance. A prosthesis that solves the task through alien transient computation would fail — the downstream motor system expects persistent choice signals. The patient's working memory experience is tied to a specific computational format.

### Outcome B — Persistent choice signal is LEARNED ZOMBIE (high ΔR², z > -2)

The LSTM encodes choice throughout the delay (high ΔR²) but doesn't use it (resample ablation has no effect). The LSTM found a different computational strategy — perhaps storing choice in the cell state without expressing it in the hidden state, or using a compressed code that doesn't project linearly onto the biological choice axis.

**Implication:** Working memory persistence is the brain's solution but not the only solution. The LSTM found an alien route. This extends the L5PC result (rich zombie) to a cognitive domain, suggesting that even cognitive-level computation can be zombified. The prosthetic engineering implication is optimistic: you don't need to replicate persistent activity, you just need correct output.

**For consciousness:** More concerning. If working memory doesn't require a specific computational format, then replacing the thalamocortical loop with any correct-output device preserves function but may alter the subjective experience of "holding something in mind."

### Outcome C — Nothing is learned (ΔR² ≈ 0 for all variables)

The LSTM predicts thalamic activity but its hidden states don't encode any known cognitive variable, even in trained vs untrained comparison. This means either the LSTM is using completely alien representations, or the biological intermediate variables we're probing for are the wrong ones.

**Action:** Apply SAE decomposition to discover what the LSTM IS encoding, since linear probes miss superposed features.

---

## 8. Cross-Circuit Comparison Table

After running this experiment, the complete DESCARTES table becomes:

```
Circuit             | Scale        | Function           | Probed | Learned | Mandatory
--------------------|--------------|--------------------|---------|---------|---------
Thalamic TC-nRt     | 2 neurons    | Oscillation        |  160   |    ?    |    ?
Hippocampal CA3→CA1 | Population   | Memory encoding    |   25   |    2    |    1
L5PC (Bahl)         | Single neuron| Dendritic integr.  |  234   |   87    |    0
ALM→Thalamus        | Population   | Working memory     |   ?    |    ?    |    ?
PMd→M1              | Population   | Motor planning     |   ?    |    ?    |    ?
IFG→vSMC            | Population   | Speech production  |   ?    |    ?    |    ?
Transformer→LSTM    | AI-to-AI     | Chain-of-thought   |   ?    |    ?    |    ?
```

The key test: does the "mandatory" column increase as we move up the cognitive hierarchy? If hippocampus has 1 mandatory variable and ALM→thalamus has more, that supports the scale-dependence hypothesis — higher cognitive functions resist zombification more strongly.

---

## 9. Photoinhibition Validation (Unique to This Dataset)

The Chen/Svoboda dataset includes sessions with photoinhibition (optogenetic silencing) of ALM during the delay period. This provides biological ground truth for causal structure that no other dataset offers.

### How to use it

The photoinhibition sessions show what happens when ALM is silenced: thalamic choice coding collapses and behavior degrades. This tells us which thalamic variables DEPEND on ALM input.

**Validation protocol:**
1. Train the LSTM on normal (no-photoinhibition) trials
2. Probe hidden states for choice signal — get ΔR² and resample z
3. Now test: feed the model with ALM activity from photoinhibition trials (where ALM choice coding is disrupted)
4. Check if the LSTM's hidden states still encode the choice signal
5. If the LSTM's choice encoding degrades when ALM input is disrupted (matching biology), the model has learned the correct causal structure
6. If the LSTM's choice encoding persists despite disrupted ALM input, the model found a shortcut that doesn't exist in biology

This is a UNIQUE validation opportunity. No other dataset lets you test whether your surrogate captures the causal dependencies between regions, not just the correlational structure.

---

## 10. Implementation Timeline

**Week 1: Data acquisition and exploration**
Download 10-15 sessions from DANDI 000363. Identify sessions with simultaneous ALM + thalamus coverage. Explore the NWB structure: column names, trial metadata, brain region labels. Extract spike data for one session as a proof of concept. Compute and visualize the choice signal in raw thalamic data.

**Week 2: Pipeline construction**
Build the data preprocessing pipeline: spike binning, trial alignment, train/val/test splitting. Train LSTM surrogates at h=64, h=128, h=256 on the best session. Compute cross-condition correlation. Verify the model learns the transformation.

**Week 3: Probing**
Extract hidden states from trained and untrained models. Compute all Level B probe targets (choice signal, ramp signal, population rate, choice magnitude). Run Ridge ΔR² with trained-vs-untrained comparison. Identify which variables are learned (ΔR² > 0.1).

**Week 4: Resample ablation and analysis**
Run resample ablation on all learned variables. Report z-scores at k = 5%, 10%, 20%, 40%, 60%, 80%. Classify as MANDATORY or ZOMBIE. Run photoinhibition validation if time permits. Generate the cross-circuit comparison table. Write up results.

---

## 11. Key Predictions

Based on the pattern from hippocampus and L5PC, here are the specific predictions:

**Choice signal (persistent activity):** PREDICTED MANDATORY. The choice axis projection during the delay is the cognitive computation itself. The LSTM must maintain choice information across the temporal gap. Like gamma_amp, this is an emergent population property that cannot be shortcutted because the downstream motor system expects it. Predicted resample z < -2 at k=40%.

**Ramp signal:** PREDICTED ZOMBIE. The temporal ramp is a correlate of motor preparation, not a computational requirement for maintaining choice. The LSTM can maintain choice without ramping. Predicted high ΔR² but resample z > -2.

**Population rate:** PREDICTED ZOMBIE. Total firing rate carries information about arousal and engagement but is not specific to working memory content. Predicted resample z > -2.

**Individual neuron selectivity (Level A):** PREDICTED MOSTLY ZOMBIE. Individual neuron tuning is a substrate detail. The LSTM will encode the population-level choice signal without replicating the specific tuning of individual biological neurons.

**Theta modulation:** UNCERTAIN. If the ALM-thalamus loop communicates through theta-frequency oscillations during the delay, the LSTM might discover theta as a mandatory timing variable — paralleling gamma_amp in the hippocampus. This would be the most exciting discovery: a second circuit where an oscillatory variable is computationally irreducible.

If the choice signal is mandatory and the ramp is zombie, the paper narrative is: **what matters for working memory is WHAT is maintained (choice identity), not HOW it's maintained (ramping, individual neurons). The cognitive content is irreducible; the neural format is fungible.**

---

## 12. Connection to Project H-Self

If persistent choice-selective activity is mandatory for working memory, this establishes a ladder:

```
gamma_amp (hippocampus)     → oscillatory timing is mandatory for memory encoding
choice_signal (ALM-thal)    → cognitive content is mandatory for memory maintenance
???  (multi-agent task)     → self-representation is mandatory for social behaviour
h-self (Project H-Self)     → self-model is mandatory for consciousness
```

Each rung tests whether a higher-level emergent property resists zombification. The Chen/Svoboda experiment is Rung 2 — the first test of a cognitive, not oscillatory, mandatory variable.

If it works, Rung 3 becomes the multi-agent RL experiment from Project H-Self, where the self-model is the predicted mandatory variable. The methodological pipeline is identical: train surrogate, probe for self-representation, resample-ablate. Only the data source changes.

The philosophical arc: from circuits (hippocampus) to cognition (working memory) to consciousness (self). Each step asks the same question — does the AI need to discover what the brain discovered? — at an increasingly abstract level of computation.
