import numpy as np
import pandas as pd
import h5py
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def load_hdf5_data(filepath):
    with h5py.File(filepath, 'r') as f:
        return np.array(f['data']['trial']).transpose(2, 1, 0)

print("Loading data...")
X_train = []
y_train = []

for f in sorted(glob.glob('training/*/*.mat')):
    label = 1 if 'sleep_emo' in f else 0
    data = load_hdf5_data(f)  # (n_trials, 16, 200)
    X_train.append(data)
    y_train.append(np.full(len(data), label))

X_train = np.vstack(X_train)
y_train = np.concatenate(y_train)

print(f"Train: {X_train.shape}, Classes: {np.bincount(y_train)}")

# Fast features: just use mean and std per channel
print("Features...")
X_train_feat = np.concatenate([
    X_train.mean(axis=2),  # (n, 16)
    X_train.std(axis=2),   # (n, 16)
], axis=1)

# Normalize
scaler = StandardScaler()
X_train_feat = scaler.fit_transform(X_train_feat)

# Train
print("Training...")
rf = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_feat, y_train)

# Check AUC
y_pred = rf.predict_proba(X_train_feat)[:, 1]
auc = roc_auc_score(y_train, y_pred)
print(f"Training AUC: {auc:.4f}")

# Load test
print("Loading test...")
X_test = []
subj_ids = []

for f in sorted(glob.glob('testing/*.mat')):
    data = load_hdf5_data(f)
    subj_id = int(os.path.basename(f).split('_')[-1].replace('.mat', ''))
    X_test.append(data)
    subj_ids.append(np.full(len(data), subj_id))

X_test = np.vstack(X_test)
subj_ids = np.concatenate(subj_ids)

print(f"Test: {X_test.shape}")

# Test features
X_test_feat = np.concatenate([
    X_test.mean(axis=2),
    X_test.std(axis=2),
], axis=1)
X_test_feat = scaler.transform(X_test_feat)

# Predict
print("Predicting...")
preds = rf.predict_proba(X_test_feat)[:, 1]
preds = np.clip(preds, 0.01, 0.99)

print(f"Prediction range: [{preds.min():.3f}, {preds.max():.3f}], mean: {preds.mean():.3f}")

# Build submission
print("Building submission...")
ids = []
pred_vals = []
trial_counter = {s: 0 for s in np.unique(subj_ids)}

for i, (subj_id, pred_val) in enumerate(zip(subj_ids, preds)):
    trial_id = trial_counter[subj_id]
    for t in range(200):
        ids.append(f'{subj_id}_{trial_id}_{t}')
        pred_vals.append(pred_val)
    trial_counter[subj_id] += 1

df = pd.DataFrame({'id': ids, 'prediction': pred_vals})
df.to_csv('submission_correct_labels_rf.csv', index=False)

print(f"✓ Submission: {len(df)} rows")
print(f"Expected test AUC: ~{auc*0.9:.3f}")
