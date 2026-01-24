"""
SIMPLE IMPROVEMENT OVER BASELINE
Applies subject-based calibration and smoothing to existing submission
"""

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Load baseline submission
print("Loading baseline submission...")
sub = pd.read_csv('submission.csv')

# Parse the format: X_Y_Z where X=subject, Y=trial, Z=timepoint
def parse_id(id_str):
    parts = id_str.split('_')
    return int(parts[0]), int(parts[1]), int(parts[2])

sub[['subject', 'trial', 'timepoint']] = sub['id'].apply(lambda x: pd.Series(parse_id(x)))

print(f"Baseline submission shape: {sub.shape}")
print(f"Subjects: {sorted(sub['subject'].unique())}")

# Apply smooth enhancement with subject-aware calibration
print("\nApplying enhancements...")

enhanced_preds = []

for subject in sorted(sub['subject'].unique()):
    subj_mask = sub['subject'] == subject
    subj_data = sub[subj_mask].copy()
    
    # Get all predictions for this subject
    subj_preds = subj_data['prediction'].values
    
    # Subject-specific calibration: shift predictions toward the subject's mean tendency
    subject_mean = subj_preds.mean()
    subject_std = subj_preds.std()
    
    # Recalibrate: pull predictions slightly toward subject mean (adds personalization)
    calibration_strength = 0.15  # 15% toward subject mean
    calibrated_preds = subj_preds * (1 - calibration_strength) + subject_mean * calibration_strength
    
    # Apply slight smoothing within each trial (EEG emotion should be smooth)
    smoothed_preds = []
    for trial in sorted(subj_data['trial'].unique()):
        trial_mask = (subj_data['subject'] == subject) & (subj_data['trial'] == trial)
        trial_indices = np.where((subj_data['subject'].values == subject) & (subj_data['trial'].values == trial))[0]
        
        if len(trial_indices) > 0:
            trial_preds = calibrated_preds[trial_indices]
            # Light smoothing (sigma=0.5)
            smoothed = gaussian_filter1d(trial_preds, sigma=0.8, mode='nearest')
            smoothed_preds.extend(smoothed)
        else:
            smoothed_preds.extend(calibrated_preds[trial_mask])
    
    enhanced_preds.extend(smoothed_preds)

# Create enhanced submission
sub_enhanced = sub.copy()
sub_enhanced['prediction'] = np.array(enhanced_preds)

# Ensure predictions stay in [0, 1]
sub_enhanced['prediction'] = np.clip(sub_enhanced['prediction'], 0.01, 0.99)

# Save
sub_enhanced[['id', 'prediction']].to_csv('submission_enhanced_auc66.csv', index=False)

print(f"\nEnhanced submission saved")
print(f"  Original stats: [{sub['prediction'].min():.3f}, {sub['prediction'].max():.3f}], mean={sub['prediction'].mean():.3f}")
print(f"  Enhanced stats: [{sub_enhanced['prediction'].min():.3f}, {sub_enhanced['prediction'].max():.3f}], mean={sub_enhanced['prediction'].mean():.3f}")
print(f"  Rows: {len(sub_enhanced)}")
