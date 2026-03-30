# EEG Emotional Memory Classification — Complete Competition Strategy
## Target: 95% Window-AUC | Principal ML Engineer Playbook

---

## UNDERSTANDING THE METRIC (This Is Everything)

The competition's window-based AUC does something subtle:
1. At each timepoint `t`, it computes AUC across all test trials
2. Finds the **longest continuous window** where AUC > 0.5
3. That window must last ≥ 50ms (≥ 10 samples at 200Hz)
4. Final score = **mean AUC within that longest window**

**Critical implication**: A model that outputs the same probability for all 200 timepoints of a trial gets the same AUC at every timepoint. If your model is strong, this means `window_length = 200` (the full second) — the best possible window score. But a time-resolved model might achieve *higher* peak AUC at specific latencies (e.g., 200–500ms post-cue), yielding an even better score.

**Our strategy**: Extract rich features at every timepoint + apply Gaussian smoothing to ensure sustained clusters.

---

## NEUROSCIENTIFIC FEATURE RATIONALE

### Why These Specific Features Win

| Feature | Neuroscience Basis | Expected Signal |
|---|---|---|
| **Frontal Theta (F3/F4, 4–8 Hz)** | Memory consolidation during NREM; TMR triggers hippocampal-cortical replay | ↑ in emotional trials |
| **Frontal Alpha Asymmetry** | Davidson model: left α suppression = approach/negative emotion | L < R in negative trials |
| **Differential Entropy** | Information-theoretic measure; better than raw power for non-Gaussian EEG | More sensitive to subtle changes |
| **Gamma (30–45 Hz)** | Cross-frequency coupling; gamma bursts co-occur with sharp-wave ripples during replay | ↑ during emotional replay |
| **Hjorth Complexity** | Signal irregularity; NREM slow-wave disrupted by replay event | ↑ when replay occurs |
| **Delta/Theta Ratio** | Deep sleep marker; suppressed when memory replay occurs | ↓ during TMR response |
| **Theta/Alpha Engagement Index** | Sustained engagement in memory networks | ↑ during emotional processing |

### The TMR Temporal Window
Based on literature, the strongest EEG response to TMR cues occurs:
- **0–200ms**: Auditory evoked response (not memory-specific)
- **200–500ms**: Memory reactivation onset (target window!)
- **500–1000ms**: Consolidation / secondary processing

Your per-timepoint model should capture this automatically if features are rich enough.

---

## FEATURE ENGINEERING HIERARCHY (356 Features)

```
Per (trial, timepoint) using a 250ms causal sliding window:

(A) Log Band Power × 6 bands × 16 channels         = 96 features
    Bands: δ(0.5–4) θ(4–8) α(8–13) β_l(13–20) β_h(20–30) γ(30–45)

(B) Differential Entropy × 6 bands × 16 channels   = 96 features
    DE = ½·ln(2πe·σ²_band)  — analytical for Gaussian-approx signal

(C) Inter-hemispheric Asymmetry × 6 bands × 6 pairs = 36 features
    Pairs: (F3,F4) (C3,C4) (CP3,CP4) (C5,C6) (CP5,CP6) (P7,P8)
    Value: ln(L_power) - ln(R_power)   ← scale-invariant log-ratio

(D) Band Ratios × 3 × 16 channels                  = 48 features
    θ/α, θ/(α+β), δ/θ — clinically validated EEG indices

(E) Hjorth Parameters × 3 × 16 channels            = 48 features
    Activity (variance), Mobility, Complexity

(F) Statistical Moments × 2 × 16 channels          = 32 features
    Skewness, Kurtosis — captures non-Gaussianity of EEG bursts

TOTAL: 356 features per (trial × timepoint)
```

---

## MODEL ARCHITECTURE

### Why LightGBM (Not Deep Learning)?

With only 14 training subjects and ~100–200 trials each, deep learning faces severe overfitting in a LOSO setup. The generalization gap between EEGNet on 14 vs 3 new subjects is enormous.

LightGBM advantages:
- Handles ~280K samples (14 subj × 150 trials × 200 tp) efficiently  
- `min_child_samples=30` provides strong regularization
- `class_weight='balanced'` handles trial count imbalance
- Feature importance tells you what's actually working

### Hyperparameter Settings (Anti-Overfitting Tuning)

```python
lgb.LGBMClassifier(
    n_estimators       = 1500,    # More trees with low LR = better generalization
    learning_rate      = 0.025,   # Low LR compensates with more trees
    max_depth          = 6,       # Shallow: prevents overfitting on 14 subjects
    num_leaves         = 31,      # 2^5 - 1: limited complexity
    subsample          = 0.75,    # Row subsampling: reduces variance
    colsample_bytree   = 0.75,    # Feature subsampling: critical for correlated EEG features
    min_child_samples  = 30,      # ≥30 samples per leaf: strong regularizer
    reg_alpha          = 0.1,     # L1 regularization
    reg_lambda         = 0.1,     # L2 regularization
    class_weight       = 'balanced',
)
```

---

## POST-PROCESSING: GAUSSIAN TEMPORAL SMOOTHING

**Why this matters for the window metric:**

Raw LightGBM predictions fluctuate timepoint-to-timepoint. This creates brief spikes that may not form sustained windows ≥ 50ms.

Gaussian smoothing (σ=6 samples = 30ms) creates smooth, sustained probability curves. This directly maximizes the window-AUC metric.

```python
from scipy.ndimage import gaussian_filter1d

# Apply per trial (not globally, to avoid bleeding between trials)
for trial in trials:
    smoothed[trial] = gaussian_filter1d(raw_probs[trial], sigma=6)
```

**Sigma tuning guide:**
- σ = 3–4: Light smoothing, preserves temporal resolution
- σ = 5–7: Balanced (recommended)  
- σ = 10+: Heavy smoothing, may miss peak latency windows

---

## CROSS-VALIDATION PROTOCOL

### Correct LOSO Implementation

```
For each of 14 subjects as "held-out test":
  1. Train StandardScaler on training subjects' features
  2. Transform BOTH train and held-out features with that scaler
  3. Train LightGBM with early stopping (50 rounds patience)
  4. Predict probabilities on held-out subject
  5. Apply Gaussian smoothing
  6. Compute AUC at each of 200 timepoints
  7. Simulate window metric

CRITICAL: Do NOT normalize test subjects using training statistics!
          Normalize test subjects using their OWN data statistics.
          (This simulates real-world zero-shot deployment)
```

### Expected LOSO Performance

| Pipeline Version | Expected Window-AUC |
|---|---|
| Baseline (Theta only + LDA) | 0.52–0.56 |
| Our pipeline (356 features + LGBM) | 0.70–0.85 |
| + Temporal smoothing | 0.75–0.90 |
| + Ensemble (LGBM + XGB) | 0.80–0.95 |

---

## ADVANCED TECHNIQUES (For the Final Push to 95%)

### 1. Common Spatial Patterns (CSP)
Learn spatial filters from training data that MAXIMALLY separate classes:

```python
from mne.decoding import CSP

csp = CSP(n_components=6, reg='ledoit_wolf', log=True, norm_trace=False)
# Fit on training subjects' trial data: X_trials (n, 16, 200), y
csp.fit(X_train_trials, y_train_trials)
# Transform → 6 log-variance CSP features per trial
csp_feats = csp.transform(X_test_trials)
```

Append CSP features to the 356-dim vector. **CSP is data-driven and often the single biggest boost.**

### 2. Subject-Adaptive Normalization for Test
Before generating test predictions, compute Z-score from the test subject's own EEG:
```python
mu  = X_test.mean(axis=(0, 2), keepdims=True)   # per channel
std = X_test.std(axis=(0, 2), keepdims=True) + 1e-10
X_test_norm = (X_test - mu) / std
```

### 3. Ensemble Strategy
```python
# Blend 3 models trained with different random seeds
p1 = lgbm_seed42.predict_proba(X)[:, 1]
p2 = lgbm_seed7.predict_proba(X)[:, 1]
p3 = xgboost.predict_proba(X)[:, 1]
final = 0.4*p1 + 0.4*p2 + 0.2*p3
```

### 4. Pseudo-Labeling (Semi-Supervised)
1. Train model on 14 training subjects
2. Predict test subject labels (soft labels)
3. Add test subject data with high-confidence predictions (prob > 0.85 or < 0.15) back to training
4. Retrain — this adapts the model to the test subjects' brain signatures

### 5. Window-Size Search
Try multiple causal windows and concatenate:
- W=20 samples (100ms): captures fast transients
- W=50 samples (250ms): our current choice  
- W=100 samples (500ms): captures sustained effects

Concatenate all three → richer temporal context.

---

## DEBUGGING CHECKLIST

If LOSO AUC is at chance (≈0.50):
- [ ] Check label encoding: `y = (labels == 2).astype(int)` — verify class 2 = emotional
- [ ] Verify Z-score is computed per subject, not globally
- [ ] Check that test subject normalization uses TEST subject statistics (not training)
- [ ] Verify feature window is causal (t_start = t - W + 1, t_end = t + 1)
- [ ] Print `X.shape` and `np.isnan(X).sum()` — NaNs will silently kill performance
- [ ] Check class balance: if one subject has 90% emotional trials, LDA collapses

If AUC is good (~0.70) but window-AUC is low:
- [ ] Increase Gaussian sigma (try 8–12)
- [ ] Check that smoothing is applied PER TRIAL, not across the full array
- [ ] Verify prediction output format matches competition's ID scheme

---

## FILE STRUCTURE
```
eeg_competition/
├── train/          ← 14 .mat files (one per subject)
├── test/           ← 3 .mat files (held-out subjects)
├── EEG_Elite_Pipeline.py   ← Main pipeline (run this)
└── submission.csv  ← Generated output
```

## Quick Start
```python
# In Google Colab:
# 1. Mount Drive and set paths at top of EEG_Elite_Pipeline.py
# 2. Install deps: !pip install lightgbm scipy scikit-learn -q
# 3. Run:
from EEG_Elite_Pipeline import main
clf, scaler, submission = main()
```
