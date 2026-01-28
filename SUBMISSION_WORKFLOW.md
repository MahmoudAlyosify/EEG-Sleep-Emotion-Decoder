# Complete Submission Workflow Guide

## 📋 Overview

This guide walks through the complete submission generation process for the EEG Emotional Memory Classification challenge.

## ✅ Submission File Requirements

**Format**: CSV with two columns
- **Column 1**: `ID` - Format: `S_{subject_id}_{trial}_{timepoint}`
- **Column 2**: `Prediction` - Float value between 0.0 and 1.0

**Example**:
```
ID,Prediction
S_1_0_0,0.432
S_1_0_1,0.441
S_1_0_2,0.448
```

**Location**: `results/submission.csv`

## 🚀 Quick Start (3 Steps)

### 1. Generate Submission
Run the submission generator:

```bash
python generate_submission.py
```

**Output**:
```
✓ Loaded 3 test subjects
✓ Generated 600 predictions
✓ Created DataFrame with 600 rows
✓ Saved to: results/submission.csv
✅ VALID - Ready for upload
```

### 2. Verify File
Check that `results/submission.csv` exists:
```bash
ls -lh results/submission.csv
```

Expected output: ~17-20 KB file

### 3. Upload
Submit the file to the competition platform.

---

## 📚 Available Tools

### A. Quick Script (Fastest)
```bash
python generate_submission.py
```
- Generates dummy predictions
- Good for testing format
- Takes ~30 seconds

### B. Python Module (Most Flexible)
```python
from src.submission_generator import SubmissionGenerator

generator = SubmissionGenerator()
submission_df = generator.generate_complete_submission()
```
- Full control over predictions
- Extensive validation
- Takes ~1 minute

### C. Jupyter Notebook (Most Interactive)
Open and run: `notebooks/EEG_Emotional_Memory_Pipeline.ipynb`
- Cell 23: "Production Submission Generation"
- Integrates with trained models
- Takes ~5-10 minutes

---

## 🔄 Full Workflow

### Step 1: Load Your Model
```python
import numpy as np
import pandas as pd
from pathlib import Path

# Load your trained model
model = load_your_trained_model()  # Your implementation
```

### Step 2: Load Test Data
```python
from src.submission_generator import SubmissionGenerator

generator = SubmissionGenerator()
test_data = generator.load_test_data()
# Returns: {'1': array(...), '7': array(...), '12': array(...)}
```

### Step 3: Generate Predictions
```python
predictions_list = []

for subject_id, eeg_data in test_data.items():
    # Ensure correct shape: (trials, channels, timepoints)
    if eeg_data.ndim == 2:
        eeg_data = eeg_data[np.newaxis, :, :]
    
    n_trials = eeg_data.shape[0]
    
    for trial_idx in range(n_trials):
        trial_data = eeg_data[trial_idx]  # (channels, timepoints)
        
        # Use your model to get predictions (channels, 200)
        trial_pred = model.predict(trial_data)  # Shape: (200,)
        predictions_list.append(trial_pred)

predictions_array = np.array(predictions_list)  # (n_trials, 200)
```

### Step 4: Create Submission
```python
submission_df = generator.generate_from_predictions(
    predictions=predictions_array,
    subject_ids=['1', '7', '12'],  # One per trial
    n_timepoints=200
)
```

### Step 5: Validate
```python
checks = generator.validate_submission(submission_df)
is_valid = generator.print_validation_report(submission_df, checks)

if not is_valid:
    print("❌ Fix validation errors before submitting")
    sys.exit(1)
```

### Step 6: Save
```python
submission_path = generator.save_submission(submission_df)
print(f"✅ Submission saved to: {submission_path}")
```

---

## 📊 Data Format Examples

### Test Data Structure
```python
# Subject 1: 1 trial, 16 channels, 200 timepoints
test_data['1'].shape → (16, 200) or (1, 16, 200)

# Subject 7: 1 trial, 16 channels, 200 timepoints  
test_data['7'].shape → (16, 200) or (1, 16, 200)

# Subject 12: 1 trial, 16 channels, 200 timepoints
test_data['12'].shape → (16, 200) or (1, 16, 200)
```

### Predictions Shape
```python
# Input: (n_trials, n_channels, n_timepoints)
eeg_data.shape → (1, 16, 200)

# Output: (n_timepoints,)
predictions.shape → (200,)

# For all trials: (n_trials, n_timepoints)
predictions_array.shape → (3, 200)  # 3 subjects × 200 timepoints
```

### Submission Format
```python
# DataFrame format
submission_df.shape → (600, 2)  # 3 subjects × 200 timepoints

# CSV format
"""
ID,Prediction
S_1_0_0,0.432
S_1_0_1,0.441
...
S_12_0_199,0.521
"""
```

---

## 🔍 Validation Checklist

### Before Submission
- [ ] File is named `submission.csv`
- [ ] Located in `results/` directory
- [ ] Exactly 600 rows (3 subjects × 200 timepoints)
- [ ] Two columns: `ID` and `Prediction`
- [ ] All IDs follow format: `S_{subject}_{trial}_{timepoint}`
- [ ] All predictions are between 0.0 and 1.0
- [ ] No missing values (NaN)
- [ ] No infinite values
- [ ] No duplicate IDs
- [ ] File is readable CSV (open in text editor to verify)

### Quick Validation Script
```python
import pandas as pd

df = pd.read_csv('results/submission.csv')

print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"ID examples: {df['ID'].head(3).tolist()}")
print(f"Prediction range: [{df['Prediction'].min():.4f}, {df['Prediction'].max():.4f}]")
print(f"Nulls: {df.isnull().sum().sum()}")
print(f"Duplicates: {df['ID'].duplicated().sum()}")

# All checks
assert len(df) == 600, f"Expected 600 rows, got {len(df)}"
assert set(df.columns) == {'ID', 'Prediction'}
assert (df['Prediction'] >= 0).all() and (df['Prediction'] <= 1).all()
assert df['ID'].duplicated().sum() == 0

print("✅ All checks passed!")
```

---

## 🐛 Troubleshooting

### Issue: "No test data found"
**Cause**: Test files not in `testing/` directory
**Solution**: 
1. Place `.mat` files in `testing/` directory
2. Or use the provided dummy data generator

### Issue: "Shape mismatch"
**Cause**: Predictions have wrong shape
**Solution**: 
```python
# Ensure (n_trials, n_timepoints)
if predictions.ndim != 2:
    raise ValueError(f"Expected 2D, got {predictions.shape}")
if predictions.shape[1] != 200:
    raise ValueError(f"Expected 200 timepoints, got {predictions.shape[1]}")
```

### Issue: "Predictions outside [0, 1]"
**Cause**: Model outputs not normalized
**Solution**: 
```python
# Clip predictions
predictions = np.clip(predictions, 0, 1)

# Or use softmax
from scipy.special import softmax
predictions = softmax(logits, axis=1)[:, 1]
```

### Issue: "Duplicate IDs"
**Cause**: Same sample predicted twice
**Solution**: 
```python
duplicates = df['ID'].duplicated(keep=False)
print(df[duplicates])
```

---

## 📈 Performance Optimization

### Tips for Better Scores

1. **Smooth predictions**
```python
from scipy.ndimage import gaussian_filter1d
predictions = gaussian_filter1d(predictions, sigma=2.0)
predictions = np.clip(predictions, 0, 1)
```

2. **Calibrate per subject**
```python
# Subject-specific scaling
for subject_id in ['1', '7', '12']:
    mask = df['ID'].str.contains(f'S_{subject_id}_')
    df.loc[mask, 'Prediction'] *= subject_specific_scale
```

3. **Post-process outliers**
```python
# Remove extreme values
df['Prediction'] = df['Prediction'].clip(0.1, 0.9)
```

4. **Ensemble multiple models**
```python
pred1 = model1.predict(test_data)
pred2 = model2.predict(test_data)
pred3 = model3.predict(test_data)

final_predictions = 0.4*pred1 + 0.3*pred2 + 0.3*pred3
```

---

## 📞 Support

### File Structure Issues
→ See: `SUBMISSION_FORMAT.md`

### Model Integration Issues
→ See: `notebooks/EEG_Emotional_Memory_Pipeline.ipynb`

### General Questions
→ See: `README.md`

---

## ✨ Summary

| Step | Tool | Time | Output |
|------|------|------|--------|
| Generate | `generate_submission.py` | 30s | CSV file |
| Validate | Built-in validator | 10s | Report |
| Test Upload | Competition platform | 1m | Confirmation |

**Total time from data to submission: ~2-5 minutes**

---

**Version**: 1.0.0
**Last Updated**: January 24, 2026
**Status**: Production Ready ✅
