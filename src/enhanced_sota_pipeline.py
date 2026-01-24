"""
Enhanced SOTA Pipeline for AUC > 0.66
Adds: Multi-scale features, subject calibration, stacking, temporal attention
"""

import numpy as np
import h5py
import os
import pandas as pd
import glob
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform

from pyriemann.estimation import Covariances, Xdawn
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_covariance

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================================
# PART 1: MULTI-SCALE FEATURE EXTRACTION
# ============================================================================

def extract_multiscale_features(X, fs=200):
    """
    Extract multi-scale temporal and spectral features.
    
    Features extracted:
    1. Theta (4-8 Hz)
    2. Alpha (8-12 Hz)
    3. Beta (12-30 Hz)
    4. Gamma (30-40 Hz)
    5. Slow waves (0.5-4 Hz)
    6. Multi-scale variance
    7. Cross-channel coherence
    
    Parameters
    ----------
    X : ndarray
        Shape (n_trials, n_channels, n_timepoints)
    fs : float
        Sampling frequency
        
    Returns
    -------
    features : ndarray
        Shape (n_trials, n_features)
    """
    n_trials, n_channels, n_timepoints = X.shape
    features_list = []
    
    # Frequency bands
    bands = [
        ('slow', 0.5, 4),
        ('theta', 4, 8),
        ('alpha', 8, 12),
        ('beta', 12, 30),
        ('gamma', 30, 40)
    ]
    
    for trial_idx in range(n_trials):
        trial_features = []
        trial_data = X[trial_idx]  # (n_channels, n_timepoints)
        
        # 1. Band power features
        for band_name, low, high in bands:
            # Bandpass filter
            nyq = fs / 2
            low_norm = low / nyq
            high_norm = high / nyq
            b, a = butter(4, [low_norm, high_norm], btype='band')
            
            band_power_ch = []
            for ch in range(n_channels):
                filtered = filtfilt(b, a, trial_data[ch])
                power = np.mean(filtered ** 2)
                band_power_ch.append(power)
            
            trial_features.append(np.mean(band_power_ch))
            trial_features.append(np.std(band_power_ch))
            trial_features.append(np.max(band_power_ch))
        
        # 2. Temporal features
        trial_features.append(np.mean(np.std(trial_data, axis=1)))  # Avg channel variability
        trial_features.append(np.max(np.std(trial_data, axis=1)))   # Max channel variability
        
        # 3. Cross-channel correlation (coherence indicator)
        cov = np.cov(trial_data)
        # Extract upper triangle
        coherence = np.mean(np.abs(cov[np.triu_indices_from(cov, k=1)]))
        trial_features.append(coherence)
        
        # 4. Entropy-like features (diversity of power spectrum)
        fft = np.abs(np.fft.fft(trial_data, axis=1))
        spectrum_entropy = np.mean(-np.sum(fft * np.log(fft + 1e-10), axis=1))
        trial_features.append(spectrum_entropy)
        
        # 5. Peak frequency features per channel
        freqs = np.fft.fftfreq(n_timepoints, 1/fs)
        peak_freqs = []
        for ch in range(n_channels):
            peak_idx = np.argmax(fft[ch, :n_timepoints//2])
            peak_freqs.append(freqs[peak_idx])
        trial_features.append(np.mean(peak_freqs))
        trial_features.append(np.std(peak_freqs))
        
        features_list.append(trial_features)
    
    return np.array(features_list)


def extract_temporal_complexity_features(X):
    """
    Extract temporal complexity features using analytic signal.
    Measures instantaneous amplitude and frequency.
    
    Parameters
    ----------
    X : ndarray
        Shape (n_trials, n_channels, n_timepoints)
        
    Returns
    -------
    features : ndarray
        Shape (n_trials, n_features)
    """
    n_trials, n_channels, n_timepoints = X.shape
    features_list = []
    
    for trial_idx in range(n_trials):
        trial_features = []
        trial_data = X[trial_idx]
        
        # Analytic signal for each channel
        for ch in range(n_channels):
            # Analytic signal
            analytic = hilbert(trial_data[ch])
            amplitude = np.abs(analytic)
            phase = np.unwrap(np.angle(analytic))
            
            # Instantaneous frequency
            inst_freq = np.diff(phase)
            
            # Features
            trial_features.append(np.mean(amplitude))
            trial_features.append(np.std(amplitude))
            trial_features.append(np.mean(inst_freq))
            trial_features.append(np.std(inst_freq))
            trial_features.append(np.max(amplitude))
        
        features_list.append(trial_features)
    
    return np.array(features_list)


# ============================================================================
# PART 2: SUBJECT-SPECIFIC CALIBRATION
# ============================================================================

class SubjectCalibrator:
    """
    Learn subject-specific calibration parameters.
    Maps model predictions to subject-specific probability scales.
    """
    
    def __init__(self):
        self.subject_calibrators = {}
        self.global_calibrator = None
    
    def fit(self, predictions, labels, subject_ids):
        """
        Fit calibration curves per subject.
        
        Parameters
        ----------
        predictions : array-like
            Model predictions (n_samples,)
        labels : array-like
            True labels (n_samples,)
        subject_ids : array-like
            Subject ID per sample (n_samples,)
        """
        unique_subjects = np.unique(subject_ids)
        
        for subj_id in unique_subjects:
            mask = subject_ids == subj_id
            pred_subj = predictions[mask]
            label_subj = labels[mask]
            
            # Fit isotonic regression or logistic regression per subject
            if len(np.unique(label_subj)) > 1:  # Only if both classes present
                calibrator = CalibratedClassifierCV(
                    LogisticRegression(random_state=42),
                    method='sigmoid',
                    cv=3
                )
                calibrator.fit(pred_subj.reshape(-1, 1), label_subj)
                self.subject_calibrators[subj_id] = calibrator
        
        # Global calibrator as fallback
        if len(np.unique(labels)) > 1:
            self.global_calibrator = CalibratedClassifierCV(
                LogisticRegression(random_state=42),
                method='sigmoid',
                cv=3
            )
            self.global_calibrator.fit(predictions.reshape(-1, 1), labels)
    
    def calibrate(self, predictions, subject_ids):
        """
        Apply subject-specific calibration.
        
        Parameters
        ----------
        predictions : array-like
            Raw predictions (n_samples,)
        subject_ids : array-like
            Subject IDs (n_samples,)
            
        Returns
        -------
        calibrated : ndarray
            Calibrated predictions (n_samples,)
        """
        calibrated = np.copy(predictions).astype(float)
        
        for subj_id in np.unique(subject_ids):
            mask = subject_ids == subj_id
            
            if subj_id in self.subject_calibrators:
                calibrated[mask] = self.subject_calibrators[subj_id].predict_proba(
                    predictions[mask].reshape(-1, 1)
                )[:, 1]
            elif self.global_calibrator is not None:
                calibrated[mask] = self.global_calibrator.predict_proba(
                    predictions[mask].reshape(-1, 1)
                )[:, 1]
        
        return calibrated


# ============================================================================
# PART 3: STACKING CLASSIFIER
# ============================================================================

class StackingEnsemble:
    """
    Stacking ensemble combining multiple base learners.
    """
    
    def __init__(self, base_models=None, meta_model=None):
        """
        Parameters
        ----------
        base_models : list of sklearn estimators
            Base models for stacking
        meta_model : sklearn estimator
            Meta-learner that combines base models
        """
        self.base_models = base_models or [
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42))
        ]
        
        self.meta_model = meta_model or LogisticRegression(random_state=42, max_iter=1000)
        self.trained_base_models = []
        self.meta_trained = False
    
    def fit(self, X, y, cv_splits=5):
        """
        Fit stacking ensemble with cross-validation.
        
        Parameters
        ----------
        X : ndarray
            Features (n_samples, n_features)
        y : ndarray
            Labels (n_samples,)
        cv_splits : int
            Number of CV splits for meta-features
        """
        from sklearn.model_selection import cross_val_predict, StratifiedKFold
        
        n_samples = X.shape[0]
        meta_features = np.zeros((n_samples, len(self.base_models)))
        
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        
        # Generate meta-features using cross-validation
        for i, (model_name, model) in enumerate(self.base_models):
            try:
                meta_features[:, i] = cross_val_predict(
                    model, X, y, cv=skf, method='predict_proba'
                )[:, 1]
            except:
                # Fallback to predict
                meta_features[:, i] = cross_val_predict(
                    model, X, y, cv=skf, method='predict'
                )
        
        # Train meta-model on meta-features
        self.meta_model.fit(meta_features, y)
        
        # Train base models on full data for prediction
        self.trained_base_models = []
        for model_name, model in self.base_models:
            model.fit(X, y)
            self.trained_base_models.append(model)
        
        self.meta_trained = True
    
    def predict_proba(self, X):
        """
        Predict probabilities using stacking.
        
        Parameters
        ----------
        X : ndarray
            Features (n_samples, n_features)
            
        Returns
        -------
        proba : ndarray
            Predicted probabilities (n_samples,)
        """
        if not self.meta_trained:
            raise ValueError("Model not fitted yet")
        
        meta_features = np.zeros((X.shape[0], len(self.trained_base_models)))
        
        for i, model in enumerate(self.trained_base_models):
            try:
                meta_features[:, i] = model.predict_proba(X)[:, 1]
            except:
                meta_features[:, i] = model.predict(X)
        
        return self.meta_model.predict_proba(meta_features)[:, 1]


# ============================================================================
# PART 4: ENHANCED PIPELINE
# ============================================================================

class EnhancedSOTAPipeline:
    """
    Enhanced pipeline with multi-scale features, calibration, and stacking.
    """
    
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.subject_ids_train = None
        self.subject_ids_test = None
        
        self.multiscale_features = None
        self.wavelet_features = None
        self.stacking_ensemble = None
        self.calibrator = None
    
    def load_data(self):
        """Load training and test data with subject IDs."""
        print("Loading training data...")
        emo_path = os.path.join(self.train_path, 'sleep_emo')
        neu_path = os.path.join(self.train_path, 'sleep_neu')
        
        emo_files = sorted(glob.glob(os.path.join(emo_path, '*.mat')))
        neu_files = sorted(glob.glob(os.path.join(neu_path, '*.mat')))
        
        data_list = []
        labels_list = []
        subject_ids = []
        
        for f in emo_files:
            data = load_hdf5_data(f)['trial']
            subj_id = int(os.path.basename(f).split('_')[1])
            data_list.append(data)
            labels_list.append(np.ones(data.shape[0]))
            subject_ids.extend([subj_id] * data.shape[0])
        
        for f in neu_files:
            data = load_hdf5_data(f)['trial']
            subj_id = int(os.path.basename(f).split('_')[1])
            data_list.append(data)
            labels_list.append(np.zeros(data.shape[0]))
            subject_ids.extend([subj_id] * data.shape[0])
        
        self.X_train = np.vstack(data_list)
        self.y_train = np.hstack(labels_list)
        self.subject_ids_train = np.array(subject_ids)
        
        print(f"Loaded {self.X_train.shape[0]} training trials")
        
        # Load test data
        print("Loading test data...")
        test_files = glob.glob(os.path.join(self.test_path, '*.mat'))
        test_data_list = []
        test_subject_ids = []
        
        for f in test_files:
            data = load_hdf5_data(f)['trial']
            subj_id = int(os.path.basename(f).split('_')[-1].replace('.mat', ''))
            test_data_list.append(data)
            test_subject_ids.extend([subj_id] * data.shape[0])
        
        self.X_test = np.vstack(test_data_list)
        self.subject_ids_test = np.array(test_subject_ids)
        print(f"Loaded {self.X_test.shape[0]} test trials")
    
    def extract_features(self):
        """Extract multi-scale and temporal complexity features."""
        print("Extracting multi-scale features...")
        self.multiscale_features = extract_multiscale_features(self.X_train)
        print(f"  Train shape: {self.multiscale_features.shape}")
        
        print("Extracting temporal complexity features...")
        self.temporal_features = extract_temporal_complexity_features(self.X_train)
        print(f"  Train shape: {self.temporal_features.shape}")
        
        # Combine features
        X_features = np.hstack([
            self.multiscale_features,
            self.temporal_features
        ])
        
        print(f"Combined feature shape: {X_features.shape}")
        return X_features
    
    def train(self):
        """Train enhanced stacking ensemble."""
        print("Extracting features...")
        X_features = self.extract_features()
        
        # Normalize features
        scaler = StandardScaler()
        X_features = scaler.fit_transform(X_features)
        
        print("Training stacking ensemble...")
        self.stacking_ensemble = StackingEnsemble()
        self.stacking_ensemble.fit(X_features, self.y_train, cv_splits=5)
        
        print("Training subject calibrator...")
        # Get base predictions for calibration
        train_preds = self.stacking_ensemble.predict_proba(X_features)
        self.calibrator = SubjectCalibrator()
        self.calibrator.fit(train_preds, self.y_train, self.subject_ids_train)
        
        # Evaluate on training data
        calibrated_preds = self.calibrator.calibrate(train_preds, self.subject_ids_train)
        auc = roc_auc_score(self.y_train, calibrated_preds)
        print(f"Training AUC (calibrated): {auc:.4f}")
    
    def predict(self):
        """Generate predictions on test set."""
        print("Extracting test features...")
        
        # Multi-scale
        multiscale_test = extract_multiscale_features(self.X_test)
        
        # Temporal complexity
        temporal_test = extract_temporal_complexity_features(self.X_test)
        
        X_test_features = np.hstack([multiscale_test, temporal_test])
        
        # Normalize (using same scaling as training)
        scaler = StandardScaler()
        scaler.fit(np.vstack([
            np.hstack([self.multiscale_features, self.temporal_features])
        ]))
        X_test_features = scaler.transform(X_test_features)
        
        print("Generating test predictions...")
        predictions = self.stacking_ensemble.predict_proba(X_test_features)
        
        # Apply calibration
        predictions = self.calibrator.calibrate(predictions, self.subject_ids_test)
        
        # Smooth
        smoothed = apply_gaussian_smoothing_flat(predictions, sigma=1.5)
        
        return smoothed
    
    def create_submission(self, predictions, output_file='submission_enhanced.csv'):
        """Create submission CSV."""
        print(f"Creating submission: {output_file}")
        
        rows = []
        test_files = sorted(glob.glob(os.path.join(self.test_path, '*.mat')))
        
        for f in test_files:
            subj_id = int(os.path.basename(f).split('_')[-1].replace('.mat', ''))
            data = load_hdf5_data(f)
            n_trials = data['trial'].shape[0]
            
            # Find predictions for this subject
            mask = self.subject_ids_test == subj_id
            subj_preds = predictions[mask]
            
            pred_idx = 0
            for trial_idx in range(n_trials):
                for time_idx in range(200):
                    row = {
                        'id': f"{subj_id}_{trial_idx}_{time_idx}",
                        'prediction': float(subj_preds[pred_idx])
                    }
                    rows.append(row)
                pred_idx += 1
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        
        print(f"Submission saved: {output_file}")
        print(f"Total rows: {len(df)}")
        print(f"Min: {df['prediction'].min():.4f}, Max: {df['prediction'].max():.4f}, Mean: {df['prediction'].mean():.4f}")
        
        return df


def apply_gaussian_smoothing_flat(predictions_flat, sigma=1.5):
    """Apply Gaussian smoothing to flat prediction array."""
    n_samples = len(predictions_flat)
    n_timepoints = 200
    n_trials = n_samples // n_timepoints
    
    predictions_2d = predictions_flat.reshape(n_trials, n_timepoints)
    smoothed_2d = np.zeros_like(predictions_2d)
    
    for trial in range(n_trials):
        smoothed_2d[trial] = gaussian_filter1d(predictions_2d[trial], sigma=sigma)
    
    return smoothed_2d.flatten()


def load_hdf5_data(filepath):
    """Load HDF5 MATLAB v7.3 file."""
    def load_field(f, data_ref, field_name):
        field = data_ref[field_name]
        if isinstance(field, h5py.Dataset):
            ref_value = field[()]
            if isinstance(ref_value, h5py.Reference):
                return f[ref_value]
            elif hasattr(ref_value, 'shape') and ref_value.shape == (1, 1):
                ref = ref_value.item()
                if isinstance(ref, h5py.Reference):
                    return f[ref]
                else:
                    if isinstance(ref, bytes):
                        ref = ref.decode('utf-8')
                    return f[ref]
            else:
                return field
        else:
            return field
    
    with h5py.File(filepath, 'r') as f:
        data_ref = f['data']
        trial_data = load_field(f, data_ref, 'trial')
        
        try:
            trialinfo_data = load_field(f, data_ref, 'trialinfo')
            trialinfo = np.array(trialinfo_data).T
        except:
            trialinfo = None
        
        time_data = np.array(load_field(f, data_ref, 'time')).flatten()
        trial_data = np.array(load_field(f, data_ref, 'trial')).T
        
        mask = time_data >= 0
        if np.any(~mask):
            time_data = time_data[mask]
            trial_data = trial_data[:, :, mask]
        
        return {
            'trial': trial_data,
            'trialinfo': trialinfo,
            'time': time_data
        }


if __name__ == "__main__":
    TRAIN_PATH = r'D:\Deep Learning & Time Series - predicting-emotions-using-brain-waves\training'
    TEST_PATH = r'D:\Deep Learning & Time Series - predicting-emotions-using-brain-waves\testing'
    
    import time
    start = time.time()
    
    pipeline = EnhancedSOTAPipeline(TRAIN_PATH, TEST_PATH)
    pipeline.load_data()
    pipeline.train()
    predictions = pipeline.predict()
    pipeline.create_submission(predictions)
    
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
