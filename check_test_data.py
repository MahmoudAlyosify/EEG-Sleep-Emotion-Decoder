import scipy.io as sio
import h5py
import numpy as np

# Try to load the testing file as MATLAB v5 format
try:
    mat = sio.loadmat('data/testing', struct_as_record=False, squeeze_me=True)
    print("Loaded as scipy.io (MATLAB v5):")
    for key in mat.keys():
        if not key.startswith('__'):
            obj = mat[key]
            if hasattr(obj, 'trial'):
                print(f"  {key}.trial shape: {obj.trial.shape}")
            elif hasattr(obj, 'shape'):
                print(f"  {key} shape: {obj.shape}")
            else:
                print(f"  {key}: {type(obj)}")
except Exception as e:
    print(f"MATLAB v5 failed: {e}")

# Try HDF5 format
try:
    with h5py.File('data/testing', 'r') as f:
        print("\nHDF5 structure:")
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape {obj.shape}")
        f.visititems(print_structure)
except Exception as e:
    print(f"HDF5 failed: {e}")

# Calculate expected rows
print("\n" + "="*50)
print("Calculations:")
expected_rows = 346800
timepoints = 200
print(f"Expected rows: {expected_rows}")
print(f"Timepoints per trial: {timepoints}")
trials_total = expected_rows // timepoints
print(f"Total trials: {trials_total}")
print(f"If 3 subjects: {trials_total // 3} trials per subject")
if trials_total % 3 != 0:
    print(f"If 2 subjects: {trials_total // 2} trials per subject (remainder: {trials_total % 2})")
print("="*50)
