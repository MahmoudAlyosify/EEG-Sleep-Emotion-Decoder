"""
LAST RESORT: Use test data itself to find structure
Try: Isolation Forest anomaly scores, variance patterns, entropy
"""

import numpy as np
import pandas as pd
import h5py
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("UNSUPERVISED PATTERN DETECTION IN TEST DATA")
print("="*70)

# Load test data
print("\nLoading test...")
X_test_list = []
subj_ids = []

for f in sorted(glob.glob('testing/*.mat')):
    subj_id = int(f.split('_')[-1].replace('.mat', ''))
    with h5py.File(f, 'r') as hf:
        data = np.array(hf['data']['trial']).transpose(2, 1, 0)
        X_test_list.append(data)
        subj_ids.append(np.full(len(data), subj_id))

X_test = np.vstack(X_test_list)
subj_ids = np.concatenate(subj_ids)
print(f"Test: {X_test.shape}")

# Extract multiple feature types
print("\nExtracting features...")

# 1. Signal statistics
stats_feats = np.concatenate([
    X_test.mean(axis=2),      # (n, 16)
    X_test.std(axis=2),       # (n, 16)
    np.abs(X_test).max(axis=2),  # (n, 16)
], axis=1)

# 2. Entropy per trial per channel
entropy_feats = np.zeros((len(X_test), 16))
for i in range(len(X_test)):
    for ch in range(16):
        sig = X_test[i, ch, :]
        # Digitize into bins
        bins = np.histogram_bin_edges(sig, bins=10)
        hist, _ = np.histogram(sig, bins=bins)
        entropy_feats[i, ch] = entropy(hist[hist > 0])

# 3. Frequency domain features (simple power in bands)
freq_feats = np.zeros((len(X_test), 16, 5))
for i in range(len(X_test)):
    for ch in range(16):
        sig = X_test[i, ch, :]
        fft_power = np.abs(np.fft.rfft(sig))**2
        freqs = np.fft.rfftfreq(len(sig), 1/200)
        
        freq_feats[i, ch, 0] = fft_power[(freqs > 0.5) & (freqs < 4)].sum()   # Delta
        freq_feats[i, ch, 1] = fft_power[(freqs >= 4) & (freqs < 8)].sum()    # Theta
        freq_feats[i, ch, 2] = fft_power[(freqs >= 8) & (freqs < 12)].sum()   # Alpha
        freq_feats[i, ch, 3] = fft_power[(freqs >= 12) & (freqs < 30)].sum()  # Beta
        freq_feats[i, ch, 4] = fft_power[(freqs >= 30)].sum()                 # Gamma

freq_flat = freq_feats.reshape(len(X_test), -1)

# Combine all features
all_feats = np.hstack([stats_feats, entropy_feats, freq_flat])
print(f"Total features: {all_feats.shape}")

# Normalize
scaler = StandardScaler()
all_feats_norm = scaler.fit_transform(all_feats)

# Apply Isolation Forest to find anomalies
print("\nFinding anomalies with Isolation Forest...")
iso = IsolationForest(contamination=0.3, random_state=42, n_jobs=-1)
anomaly_labels = iso.fit_predict(all_feats_norm)  # -1 or 1

# Convert anomaly scores to probabilities
anomaly_scores = iso.score_samples(all_feats_norm)
# Normalize to [0, 1]
preds = 1 / (1 + np.exp(-anomaly_scores))  # Sigmoid
preds = np.clip(preds, 0.01, 0.99)

print(f"Anomaly distribution: {np.sum(anomaly_labels==-1)} anomalies, {np.sum(anomaly_labels==1)} normal")
print(f"Predictions: [{preds.min():.4f}, {preds.max():.4f}], mean: {preds.mean():.4f}")

# Build submission
print("\nBuilding submission...")
ids = []
pred_vals = []
trial_counter = {s: 0 for s in np.unique(subj_ids)}

for subj_id, pred_val in zip(subj_ids, preds):
    trial_id = trial_counter[subj_id]
    for t in range(200):
        ids.append(f'{subj_id}_{trial_id}_{t}')
        pred_vals.append(pred_val)
    trial_counter[subj_id] += 1

df = pd.DataFrame({'id': ids, 'prediction': pred_vals})
df.to_csv('submission_unsupervised_anomaly.csv', index=False)

print(f"✓ Done: {len(df)} rows")
