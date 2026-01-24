"""
Try advanced feature engineering + multiple models to find any hidden pattern
"""

import numpy as np
import pandas as pd
import h5py
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ADVANCED FEATURES + VOTING ENSEMBLE")
print("="*70)

def extract_advanced_features(X):
    """X shape: (n_trials, 16, 200)"""
    n_trials, n_ch, n_t = X.shape
    
    # Stats: mean, std, min, max, median, skew
    stats = np.concatenate([X.mean(axis=2), X.std(axis=2), 
                             X.min(axis=2), X.max(axis=2)], axis=1)
    
    # Energy in frequency bands (simple approach)
    freqs = np.fft.rfftfreq(n_t, 1/200)
    psd = np.zeros((n_trials, n_ch, 5))  # 5 freq bands
    for i in range(n_trials):
        for ch in range(n_ch):
            fft = np.abs(np.fft.rfft(X[i, ch, :]))**2
            psd[i, ch, 0] = fft[(freqs > 0.5) & (freqs < 4)].sum()    # Delta
            psd[i, ch, 1] = fft[(freqs >= 4) & (freqs < 8)].sum()     # Theta
            psd[i, ch, 2] = fft[(freqs >= 8) & (freqs < 12)].sum()    # Alpha
            psd[i, ch, 3] = fft[(freqs >= 12) & (freqs < 30)].sum()   # Beta
            psd[i, ch, 4] = fft[(freqs >= 30)].sum()                  # Gamma
    
    psd_flat = psd.reshape(n_trials, -1)
    
    # Permutation entropy (simplified - just variance of sorted segments)
    pe = np.zeros((n_trials, n_ch))
    for i in range(n_trials):
        for ch in range(n_ch):
            sig = X[i, ch, :]
            segments = np.array([sig[j:j+10] for j in range(0, len(sig)-10, 10)])
            pe[i, ch] = np.var([np.argsort(seg).tolist() for seg in segments])
    
    return np.hstack([stats, psd_flat, pe])

# Load data
print("\n[1/4] Loading...")
X_train, y_train = [], []
for f in sorted(glob.glob('training/*/*.mat')):
    label = 1 if 'sleep_emo' in f else 0
    with h5py.File(f, 'r') as hf:
        X_train.append(np.array(hf['data']['trial']).transpose(2, 1, 0))
        y_train.append(np.full(len(np.array(hf['data']['trial']).transpose(2, 1, 0)), label))

X_train = np.vstack(X_train)
y_train = np.concatenate(y_train)
print(f"Train: {X_train.shape}, Classes: {np.bincount(y_train)}")

# Extract features
print("\n[2/4] Features...")
X_train_feat = extract_advanced_features(X_train)
print(f"Features: {X_train_feat.shape}")

# Normalize
scaler = StandardScaler()
X_train_feat = scaler.fit_transform(X_train_feat)

# Train ensemble
print("\n[3/4] Training 3-model voting ensemble...")
models = {
    'RF': RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1),
    'GB': GradientBoostingClassifier(n_estimators=30, max_depth=5, random_state=42),
    'LR': LogisticRegression(C=0.5, max_iter=1000, random_state=42)
}

train_preds = np.zeros((len(y_train), len(models)))
for idx, (name, model) in enumerate(models.items()):
    print(f"  {name}...", end=' ', flush=True)
    model.fit(X_train_feat, y_train)
    train_preds[:, idx] = model.predict_proba(X_train_feat)[:, 1]
    print("✓")

# Voting
voting_pred = train_preds.mean(axis=1)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_train, voting_pred)
print(f"  Ensemble AUC: {auc:.4f}")

# Load & predict test
print("\n[4/4] Test prediction...")
X_test, test_ids = [], []
for f in sorted(glob.glob('testing/*.mat')):
    subj_id = int(f.split('_')[-1].replace('.mat', ''))
    with h5py.File(f, 'r') as hf:
        X_test.append(np.array(hf['data']['trial']).transpose(2, 1, 0))
        test_ids.append(np.full(len(np.array(hf['data']['trial']).transpose(2, 1, 0)), subj_id))

X_test = np.vstack(X_test)
test_ids = np.concatenate(test_ids)

X_test_feat = extract_advanced_features(X_test)
X_test_feat = scaler.transform(X_test_feat)

# Predict with each model
test_preds_ensemble = np.zeros((len(X_test), len(models)))
for idx, (name, model) in enumerate(models.items()):
    test_preds_ensemble[:, idx] = model.predict_proba(X_test_feat)[:, 1]

preds = test_preds_ensemble.mean(axis=1)
preds = np.clip(preds, 0.01, 0.99)

print(f"Predictions: [{preds.min():.3f}, {preds.max():.3f}], mean: {preds.mean():.3f}")

# Build submission
ids, pred_vals = [], []
trial_counter = {s: 0 for s in np.unique(test_ids)}
for subj_id, pred_val in zip(test_ids, preds):
    trial_id = trial_counter[subj_id]
    for t in range(200):
        ids.append(f'{subj_id}_{trial_id}_{t}')
        pred_vals.append(pred_val)
    trial_counter[subj_id] += 1

df = pd.DataFrame({'id': ids, 'prediction': pred_vals})
df.to_csv('submission_advanced_ensemble.csv', index=False)

print(f"\n✓ Submission: {len(df)} rows")
print(f"Expected AUC: ~{auc*0.8:.3f}")
