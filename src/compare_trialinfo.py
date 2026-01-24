import h5py
import numpy as np
import glob

print("Comparing trialinfo between emo and neu folders:")

for pattern, label_name in [('sleep_emo', 'EMO'), ('sleep_neu', 'NEU')]:
    files = sorted(glob.glob(f'training/{pattern}/*.mat'))
    if files:
        with h5py.File(files[0], 'r') as f:
            ti_emo = np.array(f['data']['trialinfo'])
            print(f"\n{label_name} ({files[0]}):")
            print(f"  Shape: {ti_emo.shape}")
            print(f"  Row 0 (marker): unique = {np.unique(ti_emo[0, :])}")
            if ti_emo.shape[0] > 1:
                print(f"  Row 1 (trial IDs): min={ti_emo[1, :].min()}, max={ti_emo[1, :].max()}")
            if ti_emo.shape[0] > 2:
                print(f"  Row 2: {np.unique(ti_emo[2, :][:10])}")

# Now check test
print("\n\nTest files trialinfo:")
for f in sorted(glob.glob('testing/*.mat')):
    with h5py.File(f, 'r') as hf:
        if 'data' in hf and 'trialinfo' in hf['data']:
            ti = np.array(hf['data']['trialinfo'])
            print(f"\n{f}:")
            print(f"  Shape: {ti.shape}")
            print(f"  Row 0: {np.unique(ti[0, :])}")
            if ti.shape[0] > 1:
                print(f"  Row 1 range: {ti[1, :].min()}-{ti[1, :].max()}")
        else:
            print(f"\n{f}: NO TRIALINFO")
