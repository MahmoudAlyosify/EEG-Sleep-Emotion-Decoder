"""
FAST ENSEMBLE PIPELINE - Quick AUC Enhancement
Uses simple but effective bandpower features + stacking
Runtime: ~30 seconds
"""

import numpy as np
import pandas as pd
import h5py
import glob
import os
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

def load_hdf5_data(filepath):
    """Load HDF5 MATLAB v7.3 file."""
    with h5py.File(filepath, 'r') as f:
        if 'data' in f:
            data_group = f['data']
        else:
            data_group = f
        
        result = {}
        for key in ['trial', 'label', 'time', 'trialinfo']:
            if key in data_group:
                result[key] = np.array(data_group[key])
        return result

def extract_bandpower_features(X, fs=200):
    """Extract bandpower features from 5 frequency bands."""
    n_trials, n_channels, n_timepoints = X.shape
    
    # Frequency bands
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12), 
             'beta': (12, 30), 'gamma': (30, 40)}
    
    features_list = []
    
    for trial_idx in range(n_trials):
        trial = X[trial_idx]
        trial_features = []
        
        for band_name, (low, high) in bands.items():
            # Bandpass filter
            nyq = fs / 2
            low_norm = max(0.001, low / nyq)
            high_norm = min(0.999, high / nyq)
            b, a = butter(4, [low_norm, high_norm], btype='band')
            
            band_power = []
            for ch in range(n_channels):
                filtered = filtfilt(b, a, trial[ch])
                power = np.mean(filtered ** 2)
                band_power.append(power)
            
            # Aggregate
            trial_features.append(np.mean(band_power))
            trial_features.append(np.std(band_power))
            trial_features.append(np.max(band_power))
        
        features_list.append(trial_features)
    
    return np.array(features_list)

class FastStackingEnsemble:
    """Fast stacking ensemble with 3 base learners."""
    
    def __init__(self, cv_splits=3):
        self.cv_splits = cv_splits
        self.base_models = [
            ('lr', LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42))
        ]
        self.meta_learner = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        self.trained_models = []
    
    def fit(self, X, y):
        """Train with cross-validation meta-features."""
        print("  Training base models...")
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models):
            print(f"    {name}...", end=' ', flush=True)
            meta_features[:, i] = cross_val_predict(
                model, X, y, cv=self.cv_splits, method='predict_proba'
            )[:, 1]
            
            # Train final model on full data
            model.fit(X, y)
            self.trained_models.append(model)
            print("✓")
        
        # Train meta-learner
        print("  Training meta-learner...", end=' ', flush=True)
        self.meta_learner.fit(meta_features, y)
        print("✓")
    
    def predict_proba(self, X):
        """Predict using ensemble."""
        meta_features = np.zeros((X.shape[0], len(self.trained_models)))
        for i, model in enumerate(self.trained_models):
            try:
                meta_features[:, i] = model.predict_proba(X)[:, 1]
            except:
                meta_features[:, i] = model.predict(X)
        
        return self.meta_learner.predict_proba(meta_features)[:, 1]

def main():
    print("="*70)
    print("FAST ENSEMBLE PIPELINE - Targeting AUC > 0.66")
    print("="*70)
    
    # Load data
    print("\n[1/5] Loading data...")
    train_files = sorted(glob.glob('training/*/*.mat'))
    X_train_list, y_train_list, subj_ids_train = [], [], []
    
    for f in train_files:
        data = load_hdf5_data(f)
        trial = np.array(data['trial']).transpose(2, 1, 0)  # (n_trials, n_channels, n_timepoints)
        label = np.array(data['trialinfo']).flatten().astype(int) if 'trialinfo' in data else np.ones(len(trial))
        subj_id = int(os.path.basename(f).split('_')[1])
        
        X_train_list.append(trial)
        y_train_list.append(label)
        subj_ids_train.extend([subj_id] * len(label))
    
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    subj_ids_train = np.array(subj_ids_train)
    
    test_files = sorted(glob.glob('testing/*.mat'))
    X_test_list, subj_ids_test = [], []
    
    for f in test_files:
        data = load_hdf5_data(f)
        trial = np.array(data['trial']).transpose(2, 1, 0)
        subj_id = int(os.path.basename(f).split('_')[-1].replace('.mat', ''))
        X_test_list.append(trial)
        subj_ids_test.extend([subj_id] * len(trial))
    
    X_test = np.vstack(X_test_list)
    subj_ids_test = np.array(subj_ids_test)
    
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"  Classes: {np.bincount(y_train)}")
    
    # Extract features
    print("\n[2/5] Extracting bandpower features...")
    X_train_feat = extract_bandpower_features(X_train)
    X_test_feat = extract_bandpower_features(X_test)
    print(f"  Feature shape: {X_train_feat.shape}")
    
    # Normalize
    print("\n[3/5] Normalizing features...")
    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)
    X_test_feat = scaler.transform(X_test_feat)
    
    # Train ensemble
    print("\n[4/5] Training stacking ensemble...")
    ensemble = FastStackingEnsemble(cv_splits=5)
    ensemble.fit(X_train_feat, y_train)
    
    # Evaluate
    train_preds = ensemble.predict_proba(X_train_feat)
    train_auc = roc_auc_score(y_train, train_preds)
    print(f"  Training AUC: {train_auc:.4f}")
    
    # Generate submission
    print("\n[5/5] Generating submission...")
    test_preds = ensemble.predict_proba(X_test_feat)
    
    submissions = []
    test_idx = 0
    for subj_id in sorted(np.unique(subj_ids_test)):
        mask = subj_ids_test == subj_id
        n_trials = np.sum(mask)
        subj_preds = test_preds[mask]
        
        for trial_id in range(n_trials):
            for t in range(200):
                submissions.append({
                    'id': f'test_subject_{subj_id}_{trial_id}_{t}',
                    'prediction': subj_preds[trial_id]
                })
    
    df_sub = pd.DataFrame(submissions)
    df_sub.to_csv('submission_fast_ensemble.csv', index=False)
    
    print(f"\n✓ Submission saved: {len(df_sub)} rows")
    print(f"  Predictions: [{df_sub['prediction'].min():.3f}, {df_sub['prediction'].max():.3f}]")
    print(f"  Mean: {df_sub['prediction'].mean():.3f}")

if __name__ == '__main__':
    main()
