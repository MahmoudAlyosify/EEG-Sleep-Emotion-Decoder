import scipy.io as sio
import h5py
import numpy as np
from pathlib import Path

print("Loading actual test data files:")
print("="*60)

test_files = {
    'testing/test_subject_7.mat': 7,
    'testing/test_subject_12.mat': 12,
}

test_data = {}
total_trials = 0

for fpath, subj_id in test_files.items():
    if Path(fpath).exists():
        try:
            # Try HDF5 first (for newer MATLAB formats)
            with h5py.File(fpath, 'r') as f:
                if 'data' in f and 'trial' in f['data']:
                    shape = f['data/trial'].shape
                    n_trials = shape[2] if len(shape) == 3 else shape[0]
                    print(f"Subject {subj_id} ({Path(fpath).name}):")
                    print(f"  Shape: {shape}")
                    print(f"  Trials: {n_trials}")
                    print(f"  Timepoints: {shape[0]}")
                    print(f"  Channels: {shape[1]}")
                else:
                    print(f"Subject {subj_id}: HDF5 structure unexpected")
        except:
            try:
                # Try scipy v5
                mat = sio.loadmat(fpath, struct_as_record=False, squeeze_me=True)
                if 'data' in mat:
                    data = mat['data']
                    if hasattr(data, 'trial'):
                        shape = data.trial.shape
                        print(f"Subject {subj_id} ({Path(fpath).name}):")
                        print(f"  Shape: {shape}")
                else:
                    print(f"Subject {subj_id}: scipy v5 structure unexpected")
            except Exception as e:
                print(f"Subject {subj_id}: Failed to load - {e}")
    else:
        print(f"Subject {subj_id} ({fpath}): FILE NOT FOUND")

print("\n" + "="*60)
print("Expected vs Actual:")
print(f"  Expected total: 346,800 rows (1,734 trials × 200 timepoints)")
print(f"  If only subjects 7,12: {(479 + 883) * 200:,} rows")
print(f"  If missing subject 1: {(372 * 200):,} rows from subject 1")
print(f"  TOTAL should be: {346800:,} rows")
