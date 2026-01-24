"""
NEW STRATEGY: Since test data is unlabeled, use transfer learning or detect patterns
Let's check if test data is even close to training distribution
"""

import numpy as np
import h5py
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pandas as pd

print("="*70)
print("ANALYSIS: Test vs Training Distribution")
print("="*70)

def get_features(data):
    """data shape: (n_trials, 16, 200)"""
    n_trials = data.shape[0]
    feats = np.concatenate([
        data.mean(axis=2),   # (n, 16)
        data.std(axis=2),    # (n, 16)
    ], axis=1)
    return feats

# Load training
X_train = []
y_train = []
for f in sorted(glob.glob('training/*/*.mat')):
    label = 1 if 'sleep_emo' in f else 0
    with h5py.File(f, 'r') as hf:
        trial_data = np.array(hf['data']['trial']).transpose(2, 1, 0)
        X_train.append(trial_data)
        y_train.append(np.full(len(trial_data), label))

X_train = np.vstack(X_train)
y_train = np.concatenate(y_train)
X_train_feat = get_features(X_train)

print(f"Training: {X_train_feat.shape}")
print(f"  Emo: {(y_train==1).sum()}, Neu: {(y_train==0).sum()}")

# Normalize
scaler = StandardScaler()
X_train_feat = scaler.fit_transform(X_train_feat)

# Load test
X_test = []
test_ids = []
for f in sorted(glob.glob('testing/*.mat')):
    subj_id = int(f.split('_')[-1].replace('.mat', ''))
    with h5py.File(f, 'r') as hf:
        trial_data = np.array(hf['data']['trial']).transpose(2, 1, 0)
        X_test.append(trial_data)
        test_ids.append(np.full(len(trial_data), subj_id))

X_test = np.vstack(X_test)
test_ids = np.concatenate(test_ids)
X_test_feat = get_features(X_test)
X_test_feat = scaler.transform(X_test_feat)

print(f"Test: {X_test_feat.shape}")

# Check: Are test features similar to train?
print("\nFeature distributions:")
print(f"  Train mean: {X_train_feat.mean():.4f}, std: {X_train_feat.std():.4f}")
print(f"  Test mean:  {X_test_feat.mean():.4f}, std: {X_test_feat.std():.4f}")

# Check per-class distributions
print(f"\n  Emo mean:   {X_train_feat[y_train==1].mean():.4f}, std: {X_train_feat[y_train==1].std():.4f}")
print(f"  Neu mean:   {X_train_feat[y_train==0].mean():.4f}, std: {X_train_feat[y_train==0].std():.4f}")

# Key insight: Check if test is closer to one class
from scipy.spatial.distance import cdist

emo_center = X_train_feat[y_train==1].mean(axis=0)
neu_center = X_train_feat[y_train==0].mean(axis=0)

dist_to_emo = np.linalg.norm(X_test_feat - emo_center, axis=1).mean()
dist_to_neu = np.linalg.norm(X_test_feat - neu_center, axis=1).mean()

print(f"\nDistance to class centers:")
print(f"  Avg distance to EMO: {dist_to_emo:.4f}")
print(f"  Avg distance to NEU: {dist_to_neu:.4f}")
print(f"  Test closer to: {'EMO' if dist_to_emo < dist_to_neu else 'NEU'}")

# Check if test might be a third distribution (OOD)
print(f"\nOut-of-distribution check:")
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomaly_scores = iso_forest.fit_predict(X_train_feat)
print(f"  Train outlier ratio: {(anomaly_scores==-1).mean():.3f}")

anomaly_scores_test = iso_forest.predict(X_test_feat)
print(f"  Test outlier ratio:  {(anomaly_scores_test==-1).mean():.3f}")
print(f"  Test OOD?: {'YES - Test very different!' if (anomaly_scores_test==-1).mean() > 0.3 else 'No, similar distribution'}")
