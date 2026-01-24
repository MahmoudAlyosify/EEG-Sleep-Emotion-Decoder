"""
Use EXACT approach from starter notebook: Hilbert theta power + LDA per timepoint
"""

import numpy as np
import h5py
import os
import pandas as pd
import glob
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import zscore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)

def extract_hilbert_power(data, freq_band=(4, 8), fs=200):
    """Extract Hilbert theta power"""
    n_trials, n_channels, n_timepoints = data.shape
    
    # Bandpass filter
    data_filtered = np.zeros_like(data)
    for trial in range(n_trials):
        for ch in range(n_channels):
            data_filtered[trial, ch, :] = butter_bandpass_filter(
                data[trial, ch, :], freq_band[0], freq_band[1], fs
            )
    
    # Hilbert
    analytic_signal = np.zeros(data_filtered.shape, dtype=complex)
    for trial in range(n_trials):
        for ch in range(n_channels):
            analytic_signal[trial, ch, :] = hilbert(data_filtered[trial, ch, :])
    
    power = np.abs(analytic_signal) ** 2
    return power

print("="*70)
print("USING STARTER NOTEBOOK APPROACH: Hilbert Theta Power + LDA")
print("="*70)

# Load training
print("\nLoading training...")
X_train_list = []
y_train_list = []

for pattern, label in [('sleep_neu', 1), ('sleep_emo', 2)]:
    for f in sorted(glob.glob(f'training/{pattern}/*.mat')):
        with h5py.File(f, 'r') as hf:
            data = np.array(hf['data']['trial']).T  # (trials, channels, time)
            X_train_list.append(data)
            y_train_list.append(np.full(len(data), label))
            print(f"  {f}: {len(data)} trials, label={label}")

X_train = np.vstack(X_train_list)
y_train = np.concatenate(y_train_list)
print(f"\nTotal train: {X_train.shape}")
print(f"Labels: 1 (NEU)={np.sum(y_train==1)}, 2 (EMO)={np.sum(y_train==2)}")

# Extract theta power
print("\nExtracting theta power...")
power_train = extract_hilbert_power(X_train)
power_zscore = zscore(power_train, axis=0)
print(f"Power shape: {power_zscore.shape}")

# Train LDA per timepoint
print("\nTraining LDA per timepoint...")
n_timepoints = power_zscore.shape[2]
aucs = []

for t in range(n_timepoints):
    if t % 20 == 0:
        print(f"  {t}/{n_timepoints}...", end=' ', flush=True)
    
    X_t = power_zscore[:, :, t]
    X_t = np.nan_to_num(X_t, 0)
    
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_t, y_train)
    
    y_pred = clf.predict_proba(X_t)[:, 1]
    auc = roc_auc_score(y_train, y_pred)
    aucs.append(auc)

print(f"\nMean AUC: {np.mean(aucs):.4f}")

# Load test
print("\nLoading test...")
X_test_list = []
subj_ids = []

for f in sorted(glob.glob('testing/*.mat')):
    subj_id = int(f.split('_')[-1].replace('.mat', ''))
    with h5py.File(f, 'r') as hf:
        data = np.array(hf['data']['trial']).T  # (trials, channels, time)
        X_test_list.append(data)
        subj_ids.append(np.full(len(data), subj_id))
        print(f"  {f}: {len(data)} trials, subj={subj_id}")

X_test = np.vstack(X_test_list)
subj_ids = np.concatenate(subj_ids)

# Extract test power
power_test = extract_hilbert_power(X_test)
power_test_zscore = zscore(power_test, axis=0)

# Predict
print("\nPredicting...")
test_preds = np.zeros(len(X_test))

for t in range(n_timepoints):
    if t % 20 == 0:
        print(f"  {t}/{n_timepoints}...", end=' ', flush=True)
    
    X_t_train = power_zscore[:, :, t]
    X_t_train = np.nan_to_num(X_t_train, 0)
    X_t_test = power_test_zscore[:, :, t]
    X_t_test = np.nan_to_num(X_t_test, 0)
    
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_t_train, y_train)
    
    y_pred_proba = clf.predict_proba(X_t_test)[:, 1]
    test_preds += y_pred_proba

test_preds /= n_timepoints
test_preds = np.clip(test_preds, 0.01, 0.99)

print(f"\nPredictions: [{test_preds.min():.3f}, {test_preds.max():.3f}], mean: {test_preds.mean():.3f}")

# Build submission
print("\nBuilding submission...")
ids, pred_vals = [], []
trial_counter = {s: 0 for s in np.unique(subj_ids)}

for subj_id, pred_val in zip(subj_ids, test_preds):
    trial_id = trial_counter[subj_id]
    for t in range(200):
        ids.append(f'{subj_id}_{trial_id}_{t}')
        pred_vals.append(pred_val)
    trial_counter[subj_id] += 1

df = pd.DataFrame({'id': ids, 'prediction': pred_vals})
df.to_csv('submission_theta_lda_notebook.csv', index=False)

print(f"✓ Done: {len(df)} rows")
