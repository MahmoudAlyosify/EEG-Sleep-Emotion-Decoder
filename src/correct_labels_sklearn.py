"""
FAST SKLEARN APPROACH: Correct labels from folder names
"""

import numpy as np
import pandas as pd
import h5py
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SKLEARN WITH CORRECT LABELS - Folder names (sleep_emo=1, sleep_neu=0)")
print("="*70)

def load_hdf5_data(filepath):
    with h5py.File(filepath, 'r') as f:
        return np.array(f['data']['trial']).transpose(2, 1, 0)

# LOAD WITH CORRECT LABELS
print("\n[1/5] Loading...")
train_files = sorted(glob.glob('training/*/*.mat'))

X_list, y_list = [], []

for f in train_files:
    label = 1 if 'sleep_emo' in f else 0
    trial_data = load_hdf5_data(f)  # (n_trials, 16 channels, 200 timepoints)
    
    # Process each trial
    for trial_idx in range(trial_data.shape[0]):
        X_list.append(trial_data[trial_idx])
        y_list.append(label)

X_train = np.array(X_list)
y_train = np.array(y_list)

print(f"  Train: {X_train.shape}")
print(f"  Classes: {np.bincount(y_train)}")

# Load test
test_files = sorted(glob.glob('testing/*.mat'))
X_test_list, subj_ids = [], []

for f in test_files:
    trial_data = load_hdf5_data(f)
    subj_id = int(os.path.basename(f).split('_')[-1].replace('.mat', ''))
    
    for trial_idx in range(trial_data.shape[0]):
        X_test_list.append(trial_data[trial_idx])
        subj_ids.append(subj_id)

X_test = np.array(X_test_list)
subj_ids = np.array(subj_ids)
print(f"  Test: {X_test.shape}")

# FEATURES: Flatten to get time-series features
print("\n[2/5] Extracting time-series features...")

def get_features(X):
    """Extract simple features from signals"""
    n_trials, n_ch, n_t = X.shape
    feats = np.zeros((n_trials, n_ch * 6))
    
    for i in range(n_trials):
        idx = 0
        for ch in range(n_ch):
            sig = X[i, ch, :]
            # 6 stats per channel
            feats[i, idx:idx+6] = [
                np.mean(sig), np.std(sig), np.max(sig), 
                np.min(sig), np.median(sig), np.ptp(sig)  # peak-to-peak
            ]
            idx += 6
    
    return feats

X_train_feat = get_features(X_train)
X_test_feat = get_features(X_test)

print(f"  Features: {X_train_feat.shape}")

# NORMALIZE
print("\n[3/5] Normalizing...")
scaler = StandardScaler()
X_train_feat = scaler.fit_transform(X_train_feat)
X_test_feat = scaler.transform(X_test_feat)

# TRAIN
print("\n[4/5] Training ensemble...")

rf = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
lr = LogisticRegression(C=0.5, max_iter=1000, random_state=42, n_jobs=-1)

print("  RF...", end=' ', flush=True)
rf.fit(X_train_feat, y_train)
print("✓")

print("  LR...", end=' ', flush=True)
lr.fit(X_train_feat, y_train)
print("✓")

# Evaluate
preds_tr_rf = rf.predict_proba(X_train_feat)[:, 1]
preds_tr_lr = lr.predict_proba(X_train_feat)[:, 1]
preds_tr = (preds_tr_rf + preds_tr_lr) / 2

auc = roc_auc_score(y_train, preds_tr)
print(f"\n  Training AUC: {auc:.4f}")

# PREDICT
print("\n[5/5] Generating submission...")

preds_te_rf = rf.predict_proba(X_test_feat)[:, 1]
preds_te_lr = lr.predict_proba(X_test_feat)[:, 1]
preds_te = (preds_te_rf + preds_te_lr) / 2
preds_te = np.clip(preds_te, 0.01, 0.99)

# Build fast submission
ids = []
preds = []

for subj_id in sorted(np.unique(subj_ids)):
    mask = subj_ids == subj_id
    subj_preds = preds_te[mask]
    
    for trial_id, pred_val in enumerate(subj_preds):
        for t in range(200):
            ids.append(f'{subj_id}_{trial_id}_{t}')
            preds.append(pred_val)

df = pd.DataFrame({'id': ids, 'prediction': preds})
df.to_csv('submission_correct_labels_sklearn.csv', index=False)

print(f"✓ Done!")
print(f"  Rows: {len(df)}")
print(f"  Range: [{preds_te.min():.3f}, {preds_te.max():.3f}]")
print(f"  Mean: {preds_te.mean():.3f}")
print(f"  Train AUC: {auc:.4f}")
print(f"\nExpected test AUC: ~{auc*0.9:.3f} (with generalization gap)")
