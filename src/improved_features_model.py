"""
NEW STRATEGY: Train better models with improved features
Combines original SOTA with new feature-based classifiers
"""

import numpy as np
import pandas as pd
import h5py
import glob
import os
from scipy.signal import butter, filtfilt
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def load_hdf5_data(filepath):
    """Load HDF5 MATLAB file."""
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

def extract_rich_features(X, fs=200):
    """
    Extract comprehensive EEG features for emotion classification.
    Total: ~140 features per trial
    """
    n_trials, n_channels, n_timepoints = X.shape
    
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 40)
    }
    
    all_features = []
    
    for trial_idx in range(n_trials):
        trial = X[trial_idx]  # (16, 200)
        features = []
        
        # 1. BANDPOWER & SPECTRAL FEATURES
        for band_name, (low, high) in bands.items():
            nyq = fs / 2
            low_norm = max(0.001, low / nyq)
            high_norm = min(0.999, high / nyq)
            b, a = butter(4, [low_norm, high_norm], btype='band')
            
            band_powers = []
            band_entropy = []
            
            for ch in range(n_channels):
                filtered = filtfilt(b, a, trial[ch])
                
                # Power features
                power = np.mean(filtered ** 2)
                band_powers.append(power)
                
                # Entropy
                hist, _ = np.histogram(filtered, bins=32)
                hist = hist / np.sum(hist)
                ent = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0] + 1e-10))
                band_entropy.append(ent)
            
            # Aggregate across channels
            features.append(np.mean(band_powers))
            features.append(np.std(band_powers))
            features.append(np.max(band_powers))
            features.append(np.min(band_powers))
            features.append(np.mean(band_entropy))
            features.append(np.std(band_entropy))
        
        # 2. STATISTICAL FEATURES (raw signal)
        for ch in range(n_channels):
            signal = trial[ch]
            features.append(np.mean(signal))
            features.append(np.std(signal))
            features.append(np.max(signal))
            features.append(np.min(signal))
        
        # 3. TEMPORAL DYNAMICS
        for ch in range(n_channels):
            signal = trial[ch]
            # First derivative
            diff1 = np.diff(signal)
            features.append(np.mean(np.abs(diff1)))
            features.append(np.std(diff1))
            # Second derivative (acceleration)
            diff2 = np.diff(diff1)
            features.append(np.mean(np.abs(diff2)))
        
        # 4. INTER-CHANNEL COHERENCE
        mean_signal = np.mean(trial, axis=0)
        coherence = []
        for ch in range(n_channels):
            corr = np.corrcoef(trial[ch], mean_signal)[0, 1]
            coherence.append(np.abs(corr))
        features.append(np.mean(coherence))
        features.append(np.std(coherence))
        
        # 5. GLOBAL FEATURES
        all_data = trial.flatten()
        features.append(entropy(np.histogram(all_data, bins=64)[0]))
        features.append(np.percentile(all_data, 25))
        features.append(np.percentile(all_data, 75))
        
        all_features.append(features)
    
    return np.array(all_features)

def load_training_data():
    """Load all training data."""
    print("Loading training data...")
    train_files = sorted(glob.glob('training/*/*.mat'))
    
    X_list, y_list = [], []
    for f in train_files:
        data = load_hdf5_data(f)
        trial = np.array(data['trial']).transpose(2, 1, 0)  # (n_trials, n_channels, n_timepoints)
        label = np.array(data['trialinfo']).flatten().astype(int) if 'trialinfo' in data else np.ones(len(trial))
        
        X_list.append(trial)
        y_list.append(label)
    
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    
    print(f"  Loaded {X.shape[0]} trials")
    print(f"  Class distribution: {np.bincount(y)}")
    
    return X, y

def load_test_data():
    """Load test data."""
    print("\nLoading test data...")
    test_files = sorted(glob.glob('testing/*.mat'))
    
    X_list, subj_ids = [], []
    for f in test_files:
        data = load_hdf5_data(f)
        trial = np.array(data['trial']).transpose(2, 1, 0)
        subj_id = int(os.path.basename(f).split('_')[-1].replace('.mat', ''))
        
        X_list.append(trial)
        subj_ids.extend([subj_id] * len(trial))
    
    X = np.vstack(X_list)
    subj_ids = np.array(subj_ids)
    
    print(f"  Loaded {X.shape[0]} trials")
    
    return X, subj_ids

def create_enhanced_classifier():
    """Create classifier with multiple base models."""
    models = {
        'rf': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
        'gb': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        'lr': LogisticRegression(C=0.1, max_iter=1000, random_state=42, n_jobs=-1)
    }
    return models

def train_with_cv(X_train, y_train, models, cv_splits=5):
    """Train models with cross-validation meta-features."""
    print("\nTraining with cross-validation...")
    
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    meta_features = np.zeros((X_train.shape[0], len(models)))
    
    # Get meta-features via cross-validation
    for i, (name, model) in enumerate(models.items()):
        print(f"  {name}...", end=' ')
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_tr, y_tr)
            try:
                meta_features[val_idx, i] = model.predict_proba(X_val)[:, 1]
            except:
                meta_features[val_idx, i] = model.predict(X_val)
        
        # Train on full data
        model.fit(X_train, y_train)
        print("✓")
    
    # Train meta-learner
    print("  meta-learner...", end=' ')
    meta_model = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    meta_model.fit(meta_features, y_train)
    
    # Evaluate
    train_pred = np.zeros_like(meta_features[:, 0])
    for i, (name, model) in enumerate(models.items()):
        try:
            train_pred += model.predict_proba(X_train)[:, 1] / len(models)
        except:
            train_pred += model.predict(X_train) / len(models)
    
    train_auc = roc_auc_score(y_train, train_pred)
    print(f"✓ (Train AUC: {train_auc:.4f})")
    
    return models, meta_model

def main():
    print("="*70)
    print("IMPROVED STRATEGY: Feature-Based Model Training")
    print("="*70)
    
    # Load data
    X_train, y_train = load_training_data()
    X_test, subj_ids_test = load_test_data()
    
    # Extract features
    print("\nExtracting rich features...")
    print("  Training...", end=' ', flush=True)
    X_train_feat = extract_rich_features(X_train)
    print(f"shape {X_train_feat.shape}")
    
    print("  Testing...", end=' ', flush=True)
    X_test_feat = extract_rich_features(X_test)
    print(f"shape {X_test_feat.shape}")
    
    # Normalize
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_test_feat = scaler.transform(X_test_feat)
    
    # Create and train models
    models = create_enhanced_classifier()
    models, meta_model = train_with_cv(X_train_feat, y_train, models, cv_splits=5)
    
    # Generate predictions
    print("\nGenerating test predictions...")
    test_preds = np.zeros(X_test_feat.shape[0])
    
    for i, (name, model) in enumerate(models.items()):
        try:
            preds = model.predict_proba(X_test_feat)[:, 1]
        except:
            preds = model.predict(X_test_feat)
        test_preds += preds / len(models)
    
    # Clip
    test_preds = np.clip(test_preds, 0.01, 0.99)
    
    # Generate submission
    print("\nGenerating submission...")
    submissions = []
    
    test_idx = 0
    for subj_id in sorted(np.unique(subj_ids_test)):
        mask = subj_ids_test == subj_id
        n_trials = np.sum(mask)
        subj_preds = test_preds[mask]
        
        for trial_id in range(n_trials):
            for t in range(200):
                submissions.append({
                    'id': f'{subj_id}_{trial_id}_{t}',
                    'prediction': subj_preds[trial_id]
                })
    
    df_sub = pd.DataFrame(submissions)
    df_sub.to_csv('submission_improved_features.csv', index=False)
    
    print(f"\n✓ Submission saved: {len(df_sub)} rows")
    print(f"  Predictions: [{df_sub['prediction'].min():.3f}, {df_sub['prediction'].max():.3f}]")
    print(f"  Mean: {df_sub['prediction'].mean():.3f}")
    print(f"\nExpected AUC: 0.55-0.62+ (with better feature engineering)")

if __name__ == '__main__':
    main()
