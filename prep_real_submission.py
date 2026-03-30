"""
Load real test data from both .mat files and HDF5, generate complete 346,800-row submission
"""
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py
from pathlib import Path
from tqdm import tqdm

def load_test_data_real():
    """Load all test data: subject 1 from HDF5, subjects 7,12 from .mat files"""
    test_data = {}
    
    # Subject 1 from data/testing HDF5
    print("Loading Subject 1 from data/testing...")
    with h5py.File("data/testing", 'r') as f:
        # Shape is (200, 16, 372) = (timepoints, channels, trials)
        trial_data = np.array(f['data/trial'])  # (200, 16, 372)
        # Rearrange to (trials, channels, timepoints)
        X_1 = trial_data.transpose(2, 1, 0)  # (372, 16, 200)
        test_data[1] = X_1
        print(f"  Subject 1: {X_1.shape}")
    
    # Subject 7 from .mat
    print("Loading Subject 7 from testing/test_subject_7.mat...")
    with h5py.File("testing/test_subject_7.mat", 'r') as f:
        trial_data = np.array(f['data/trial'])  # (200, 16, 479)
        X_7 = trial_data.transpose(2, 1, 0)  # (479, 16, 200)
        test_data[7] = X_7
        print(f"  Subject 7: {X_7.shape}")
    
    # Subject 12 from .mat
    print("Loading Subject 12 from testing/test_subject_12.mat...")
    with h5py.File("testing/test_subject_12.mat", 'r') as f:
        trial_data = np.array(f['data/trial'])  # (200, 16, 883)
        X_12 = trial_data.transpose(2, 1, 0)  # (883, 16, 200)
        test_data[12] = X_12
        print(f"  Subject 12: {X_12.shape}")
    
    return test_data

# Load data
test_data = load_test_data_real()

# Verify totals
total_trials = sum(X.shape[0] for X in test_data.values())
total_rows = total_trials * 200
print(f"\n✓ Loaded {total_trials} total trials")
print(f"✓ Expected submission rows: {total_rows:,}")
print(f"✓ Matches expected 346,800: {total_rows == 346800}")

# Create submission template for demonstration
print("\nGeneration plan:")
print("  1. Generate predictions for each subject's trials")
print("  2. Create submission with format: {subject}_{trial}_{timepoint}")
print(f"  3. Output: {total_rows:,} rows")
