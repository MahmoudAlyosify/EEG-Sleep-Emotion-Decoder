import scipy.io as sio
import h5py
import numpy as np
from pathlib import Path

print("Checking test data files in 'testing/' directory:")
test_dir = Path("testing")
test_files = sorted(test_dir.glob("*.mat"))
print(f"Found {len(test_files)} test files:")

total_trials = 0
subject_trials = {}

for fpath in test_files:
    try:
        # Try scipy v5 first
        mat = sio.loadmat(str(fpath), struct_as_record=False, squeeze_me=True)
        if 'd' in mat:  # Old format
            data = mat['d']
            if hasattr(data, 'trial'):
                trials_shape = data.trial.shape
                n_trials = trials_shape[0] if len(trials_shape) > 2 else 1
            else:
                n_trials = "unknown"
        else:
            n_trials = "unknown"
        print(f"  {fpath.name:30s} - scipy: {n_trials} trials")
    except:
        print(f"  {fpath.name:30s} - scipy failed")

print("\nChecking data/testing HDF5 file:")
try:
    with h5py.File("data/testing", 'r') as f:
        if 'data/trial' in f:
            shape = f['data/trial'].shape
            print(f"  data/trial shape: {shape}")
            if len(shape) == 3:
                print(f"    → {shape[2]} trials × {shape[1]} channels × {shape[0]} timepoints")
        else:
            print("  'data/trial' not found")
            print(f"  Available keys: {list(f.keys())}")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "="*60)
print("Testing file format expectations:")
tests_expected = 346800
tp_per_trial = 200
trials_needed = tests_expected // tp_per_trial
print(f"Expected rows: {tests_expected}")
print(f"Timepoints per trial: {tp_per_trial}")
print(f"Total trials needed: {trials_needed}")
print(f"If 3 subjects: {trials_needed//3} trials per subject (r={trials_needed%3})")
