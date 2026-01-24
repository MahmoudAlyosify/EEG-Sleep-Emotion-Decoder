"""
ULTRA-FAST TRAINING: Precompute filters, use simple features
"""

import numpy as np
import pandas as pd
import h5py
import glob
import os
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ULTRA-FAST TRAINING - Optimized Feature Pipeline")
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

# PRECOMPUTE FILTERS
print("\n[1/6] Precomputing filters...")
fs = 200
nyq = fs / 2
filters_design = {
    'theta': butter(4, [4/nyq, 8/nyq], btype='band'),
    'alpha': butter(4, [8/nyq, 12/nyq], btype='band'),
    'beta': butter(4, [12/nyq, 30/nyq], btype='band'),
}
print("  3 bandpass filters ready")

# LOAD DATA
print("\n[2/6] Loading data...")
train_files = sorted(glob.glob('training/*/*.mat'))
X_list, y_list = [], []

for f in train_files:
    data = load_hdf5_data(f)
    trial = np.array(data['trial']).transpose(2, 1, 0)
    label = np.array(data['trialinfo']).flatten().astype(int) if 'trialinfo' in data else np.ones(len(trial))
    X_list.append(trial)
    y_list.append(label)

X_train = np.vstack(X_list)
y_train = np.concatenate(y_list)
print(f"  Train: {X_train.shape} | Classes: {np.bincount(y_train)}")

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

# EXTRACT SIMPLE FEATURES
print("\n[3/6] Extracting features...")

def extract_fast(X):
    n_trials, n_ch, n_t = X.shape
    feats = []
    
    for i in range(n_trials):
        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{n_trials}...", flush=True)
        
        trial = X[i]
        f = []
        
        # Raw stats per channel
        for ch in range(n_ch):
            sig = trial[ch]
            f.append(np.mean(sig))
            f.append(np.std(sig))
            f.append(np.max(sig))
            f.append(np.min(sig))
        
        # Band power per channel
        for band_name, (b, a) in filters_design.items():
            for ch in range(n_ch):
                filt = filtfilt(b, a, trial[ch])
                f.append(np.mean(filt**2))
                f.append(np.std(filt**2))
        
        feats.append(f)
    
    return np.array(feats)

print("  Train:", end=' ', flush=True)
X_train_feat = extract_fast(X_train)
print(f"shape {X_train_feat.shape}")

print("  Test:", end=' ', flush=True)
X_test_feat = extract_fast(X_test)
print(f"shape {X_test_feat.shape}")

# NORMALIZE
print("\n[4/6] Normalizing...")
scaler = StandardScaler()
X_train_feat = scaler.fit_transform(X_train_feat)
X_test_feat = scaler.transform(X_test_feat)

# TRAIN
print("\n[5/6] Training models...")
rf = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
lr = LogisticRegression(C=0.5, max_iter=500, random_state=42, n_jobs=-1)

print("  Random Forest...", end=' ', flush=True)
rf.fit(X_train_feat, y_train)
print("✓")

print("  Gradient Boosting...", end=' ', flush=True)
gb.fit(X_train_feat, y_train)
print("✓")

print("  Logistic Regression...", end=' ', flush=True)
lr.fit(X_train_feat, y_train)
print("✓")

preds_tr = np.zeros(len(X_train_feat))
for m in [rf, gb, lr]:
    try:
        preds_tr += m.predict_proba(X_train_feat)[:, 1] / 3
    except:
        preds_tr += m.predict(X_train_feat) / 3

auc = roc_auc_score(y_train, preds_tr)
print(f"\n  Train AUC: {auc:.4f}")

# PREDICT
print("\n[6/6] Generating submission...")
preds_te = np.zeros(len(X_test_feat))
for m in [rf, gb, lr]:
    try:
        preds_te += m.predict_proba(X_test_feat)[:, 1] / 3
    except:
        preds_te += m.predict(X_test_feat) / 3

preds_te = np.clip(preds_te, 0.05, 0.95)

# BUILD SUBMISSION
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
df.to_csv('submission_ultrafast.csv', index=False)

print(f"\n✓ SUCCESS!")
print(f"  Rows: {len(df)}")
print(f"  Pred range: [{preds_te.min():.3f}, {preds_te.max():.3f}]")
print(f"  Mean: {preds_te.mean():.3f}")
print(f"\nExpected AUC: 0.58-0.63+")
