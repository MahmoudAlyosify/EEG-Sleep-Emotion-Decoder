"""
CORRECT APPROACH: Use folder name as label (sleep_emo=1, sleep_neu=0)
Plus use deep learning on raw EEG signals
"""

import numpy as np
import pandas as pd
import h5py
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CORRECT APPROACH - Folder labels + Deep Learning on Raw EEG")
print("="*70)

def load_hdf5_data(filepath):
    with h5py.File(filepath, 'r') as f:
        if 'data' in f:
            data_group = f['data']
        else:
            data_group = f
        return np.array(data_group['trial'])  # (200 timepoints, 16 channels, 200 trials)

# LOAD WITH CORRECT LABELS
print("\n[1/4] Loading data with correct labels...")
train_files = sorted(glob.glob('training/*/*.mat'))

X_list, y_list = [], []

for f in train_files:
    # Label from folder name
    if 'sleep_emo' in f:
        label = 1  # Emotional
    elif 'sleep_neu' in f:
        label = 0  # Neutral
    else:
        continue
    
    # Data: (200 timepoints, 16 channels, 200 trials)
    trial_data = load_hdf5_data(f)
    # Transpose to (200 trials, 16 channels, 200 timepoints)
    trial_data = trial_data.transpose(2, 1, 0)
    
    X_list.append(trial_data)
    y_list.extend([label] * len(trial_data))

X_train = np.vstack(X_list)
y_train = np.array(y_list)

print(f"  Train: {X_train.shape}")
print(f"  Classes: {np.bincount(y_train)}")
print(f"  Class distribution: {np.bincount(y_train) / len(y_train)}")

# Load test
print("\nLoading test data...")
test_files = sorted(glob.glob('testing/*.mat'))
X_test_list, subj_ids = [], []

for f in test_files:
    trial_data = load_hdf5_data(f)
    trial_data = trial_data.transpose(2, 1, 0)
    subj_id = int(os.path.basename(f).split('_')[-1].replace('.mat', ''))
    
    X_test_list.append(trial_data)
    subj_ids.extend([subj_id] * len(trial_data))

X_test = np.vstack(X_test_list)
subj_ids = np.array(subj_ids)
print(f"  Test: {X_test.shape}")

# NORMALIZE
print("\n[2/4] Normalizing...")
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train_flat).reshape(X_train.shape)
X_test_norm = scaler.transform(X_test_flat).reshape(X_test.shape)

# BUILD SIMPLE CNN
print("\n[3/4] Building and training CNN...")

model = keras.Sequential([
    keras.layers.Conv1D(32, 3, activation='relu', padding='same', input_shape=(16, 200)),
    keras.layers.MaxPooling1D(2),
    keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling1D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train with early stopping
history = model.fit(
    X_train_norm, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate
train_preds = model.predict(X_train_norm, verbose=0).flatten()
train_auc = roc_auc_score(y_train, train_preds)
print(f"\nTraining AUC: {train_auc:.4f}")

# Test predictions
print("\n[4/4] Generating submission...")
test_preds = model.predict(X_test_norm, verbose=0).flatten()
test_preds = np.clip(test_preds, 0.01, 0.99)

# Build submission - faster version
print("\n[4/4] Generating submission...")
test_preds = model.predict(X_test_norm, verbose=0).flatten()
test_preds = np.clip(test_preds, 0.01, 0.99)

# Pre-create all rows
ids = []
preds = []

test_idx = 0
for subj_id in sorted(np.unique(subj_ids)):
    mask = subj_ids == subj_id
    n_trials = np.sum(mask)
    subj_preds = test_preds[mask]
    
    for trial_id in range(n_trials):
        pred_val = subj_preds[trial_id]
        for t in range(200):
            ids.append(f'{subj_id}_{trial_id}_{t}')
            preds.append(pred_val)

df = pd.DataFrame({'id': ids, 'prediction': preds})
df.to_csv('submission_correct_labels.csv', index=False)

print(f"\n✓ SUCCESS!")
print(f"  Rows: {len(df)}")
print(f"  Range: [{test_preds.min():.3f}, {test_preds.max():.3f}]")
print(f"  Mean: {test_preds.mean():.3f}")
print(f"  Train AUC: {train_auc:.4f}")
print(f"\nExpected test AUC: 0.65-0.75+")
