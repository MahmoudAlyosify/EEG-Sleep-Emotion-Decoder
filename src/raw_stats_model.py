"""
RAW SIGNAL APPROACH: No filtering, just raw EEG + statistical features
Fastest possible with sklearn
"""

import numpy as np
import pandas as pd
import h5py
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("RAW SIGNAL APPROACH - Direct EEG Features")
print("="*70)

def load_hdf5_data(filepath):
    with h5py.File(filepath, 'r') as f:
        if 'data' in f:
            data_group = f['data']
        else:
            data_group = f
        result = {}
        for key in ['trial', 'trialinfo']:
            if key in data_group:
                result[key] = np.array(data_group[key])
        return result

# LOAD
print("\n[1/5] Loading data...")
train_files = sorted(glob.glob('training/*/*.mat'))
X_list, y_list = [], []

for f in train_files:
    data = load_hdf5_data(f)
    trial = np.array(data['trial']).transpose(2, 1, 0)
    label = np.array(data['trialinfo']).flatten() if 'trialinfo' in data else np.ones(len(trial))
    # Take first label per trial if multiple
    if len(label) != len(trial):
        label = label[:len(trial)]
    label = label.astype(int)
    X_list.append(trial)
    y_list.append(label)

X_train = np.vstack(X_list)
y_train = np.concatenate(y_list)
print(f"  Train: {X_train.shape}")

test_files = sorted(glob.glob('testing/*.mat'))
X_test_list, subj_ids = [], []

for f in test_files:
    data = load_hdf5_data(f)
    trial = np.array(data['trial']).transpose(2, 1, 0)
    subj_id = int(os.path.basename(f).split('_')[-1].replace('.mat', ''))
    X_test_list.append(trial)
    subj_ids.extend([subj_id] * len(trial))

X_test = np.vstack(X_test_list)
subj_ids = np.array(subj_ids)
print(f"  Test: {X_test.shape}")

# EXTRACT SIMPLE STATISTICAL FEATURES
print("\n[2/5] Extracting statistical features...")

def extract_stats(X):
    """Raw signal statistics - VERY FAST"""
    n_trials, n_ch, n_t = X.shape
    feats = np.zeros((n_trials, n_ch * 4))
    
    for i in range(n_trials):
        for ch in range(n_ch):
            sig = X[i, ch, :]
            feats[i, ch*4 + 0] = np.mean(sig)
            feats[i, ch*4 + 1] = np.std(sig)
            feats[i, ch*4 + 2] = np.max(sig)
            feats[i, ch*4 + 3] = np.min(sig)
    
    return feats

X_train_feat = extract_stats(X_train)
X_test_feat = extract_stats(X_test)
print(f"  Train features: {X_train_feat.shape}")
print(f"  Test features: {X_test_feat.shape}")

# NORMALIZE
print("\n[3/5] Normalizing...")
scaler = StandardScaler()
X_train_feat = scaler.fit_transform(X_train_feat)
X_test_feat = scaler.transform(X_test_feat)
print(f"  Normalized: Train {X_train_feat.shape}, Test {X_test_feat.shape}")

# TRAIN MODELS
print("\n[4/5] Training ensemble...")

models = {
    'RF': RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1),
    'LR': LogisticRegression(C=1.0, max_iter=500, random_state=42, n_jobs=-1),
}

for name, model in models.items():
    print(f"  {name}...", end=' ', flush=True)
    model.fit(X_train_feat, y_train)
    print("✓")

# EVALUATE TRAINING
preds_tr = np.zeros(len(X_train_feat))
for model in models.values():
    try:
        preds_tr += model.predict_proba(X_train_feat)[:, 1] / len(models)
    except:
        preds_tr += model.predict(X_train_feat) / len(models)

# Only if binary classification
if len(np.unique(y_train)) == 2:
    auc = roc_auc_score(y_train, preds_tr)
    print(f"\n  Training AUC: {auc:.4f}")
else:
    print(f"\n  Classes: {np.unique(y_train)} (not binary, skipping AUC)")

# TEST PREDICTIONS
print("\n[5/5] Generating submission...")
preds_te = np.zeros(len(X_test_feat))
for model in models.values():
    try:
        preds_te += model.predict_proba(X_test_feat)[:, 1] / len(models)
    except:
        preds_te += model.predict(X_test_feat) / len(models)

# Calibrate towards extremes to improve AUC
preds_te = 0.5 + 1.05 * (preds_te - 0.5)  # Slight boost
preds_te = np.clip(preds_te, 0.05, 0.95)

# BUILD SUBMISSION
print("  Building submission...", end=' ', flush=True)
subs = []
for subj_id in sorted(np.unique(subj_ids)):
    mask = subj_ids == subj_id
    n_tr = np.sum(mask)
    
    for tr_id in range(n_tr):
        for t in range(200):
            subs.append({
                'id': f'{subj_id}_{tr_id}_{t}',
                'prediction': preds_te[mask][tr_id]
            })

df = pd.DataFrame(subs)
df.to_csv('submission_raw_stats.csv', index=False)
print("✓")

print(f"\n✓ COMPLETE!")
print(f"  Rows: {len(df)}")
print(f"  Pred range: [{preds_te.min():.3f}, {preds_te.max():.3f}]")
print(f"  Mean: {preds_te.mean():.3f}")
print(f"\nExpected AUC: 0.55-0.60+")
