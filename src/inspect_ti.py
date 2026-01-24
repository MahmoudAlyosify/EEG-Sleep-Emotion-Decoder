import h5py
import numpy as np

with h5py.File('training/sleep_emo/S_2_cleaned.mat', 'r') as f:
    ti = np.array(f['data']['trialinfo'])
    print("Trialinfo shape:", ti.shape)
    print("First 10 trials:")
    for i in range(min(10, ti.shape[1])):
        print(f"  Trial {i}: {ti[:, i]}")
