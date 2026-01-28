# Submission File Format Specification

## Overview

The submission file must be a CSV with exactly two columns in the following format:

```
ID,Prediction
S_1_0_0,0.432
S_1_0_1,0.441
S_1_0_2,0.448
...
S_3_5_199,0.521
```

## File Structure

### Filename
- **Required**: `submission.csv`
- **Location**: `results/` directory

### Format Specifications

#### Column 1: ID
- **Format**: `S_{subject_id}_{trial_index}_{timepoint}`
- **Components**:
  - `S_`: Prefix (constant)
  - `{subject_id}`: Subject identifier (1, 7, 12, etc.)
  - `{trial_index}`: Trial index within subject (0, 1, 2, ...)
  - `{timepoint}`: Timepoint index (0 to 199)

**Examples:**
```
S_1_0_0       → Subject 1, Trial 0, Timepoint 0
S_7_2_150     → Subject 7, Trial 2, Timepoint 150
S_12_0_199    → Subject 12, Trial 0, Timepoint 199
```

#### Column 2: Prediction
- **Type**: Float (numeric)
- **Range**: [0.0, 1.0]
- **Interpretation**:
  - 0.0 = Neutral memory (100% confidence)
  - 0.5 = Uncertain
  - 1.0 = Emotional memory (100% confidence)
- **Precision**: Should include decimal places (e.g., 0.432, not 0)

### Data Requirements

#### Test Subjects
- **Number of subjects**: 3 (IDs: 1, 7, 12)
- **Data source**: `.mat` files in `testing/` directory

#### Trials
- **Variable per subject**: Different subjects may have different numbers of trials
- **Format**: 2D array of shape (channels, timepoints)
  - Channels: 16 EEG sensors
  - Timepoints: 200 per trial (1 second at 200Hz)

#### Total Entries
- **Calculation**: n_subjects × n_trials_per_subject × n_timepoints
- **Example**: 3 subjects × 5 trials per subject × 200 timepoints = 3,000 entries

### Validation Checklist

Before submission, ensure:

✅ **Structure**
- [ ] File is named `submission.csv`
- [ ] Exactly 2 columns: `ID` and `Prediction`
- [ ] Header row present: `ID,Prediction`
- [ ] No extra columns or rows

✅ **Content**
- [ ] All IDs follow format: `S_{subject}_{trial}_{timepoint}`
- [ ] All predictions are numeric (float)
- [ ] All predictions in range [0.0, 1.0]
- [ ] No missing values (NaN)
- [ ] No infinite values
- [ ] No duplicate IDs

✅ **Data**
- [ ] Subject IDs: 1, 7, 12
- [ ] Trial indices: 0 to (n_trials - 1)
- [ ] Timepoint indices: 0 to 199
- [ ] All timepoints present for each trial

### File Size

**Expected file size**: ~50-100 KB (depending on compression)
- 3 subjects × ~5-10 trials × 200 timepoints = 3,000-6,000 rows
- Each row: ~20 bytes (ID) + ~5-10 bytes (prediction) = ~30 bytes
- Total: 3,000-6,000 × 30 = 90,000-180,000 bytes

### Example Submission File

```csv
ID,Prediction
S_1_0_0,0.4321
S_1_0_1,0.4410
S_1_0_2,0.4485
S_1_0_3,0.4556
S_1_0_4,0.4623
...
S_1_0_199,0.5234
S_1_1_0,0.3890
S_1_1_1,0.3967
...
S_7_0_0,0.5123
S_7_0_1,0.5187
...
S_12_0_0,0.4456
S_12_0_1,0.4521
...
```

## Generation Workflow

### Step 1: Prepare Data
```python
from submission_generator import SubmissionGenerator

generator = SubmissionGenerator()
test_data = generator.load_test_data()
```

### Step 2: Generate Predictions
```python
# Use your trained model to generate predictions
# predictions shape: (n_trials, n_timepoints)
predictions = model.predict(test_data)
```

### Step 3: Create Submission
```python
submission_df = generator.generate_from_predictions(
    predictions=predictions,
    subject_ids=['1', '1', '7', '12'],  # One per trial
    n_timepoints=200
)
```

### Step 4: Validate
```python
checks = generator.validate_submission(submission_df)
is_valid = generator.print_validation_report(submission_df, checks)
```

### Step 5: Save
```python
submission_path = generator.save_submission(submission_df)
```

## Error Handling

### Common Issues and Solutions

**Issue**: Predictions outside [0, 1] range
```python
# Solution: Clip predictions
predictions = np.clip(predictions, 0, 1)
```

**Issue**: Wrong number of timepoints
```python
# Solution: Ensure correct timepoint count
if predictions.shape[1] != 200:
    raise ValueError(f"Expected 200 timepoints, got {predictions.shape[1]}")
```

**Issue**: Missing subject or trial
```python
# Solution: Verify all subjects/trials are present
expected_entries = len(unique_subjects) * len(trials_per_subject) * 200
actual_entries = len(submission_df)
assert expected_entries == actual_entries
```

**Issue**: Duplicate IDs
```python
# Solution: Check for duplicates
duplicates = submission_df['ID'].duplicated().sum()
if duplicates > 0:
    print(f"Found {duplicates} duplicate IDs")
```

## Competition Scoring

### Metric: Window-Based AUC
The competition uses a custom metric:

1. **Sliding Window Approach**
   - Apply sliding windows of varying sizes
   - Compute AUC for each window

2. **Significance Thresholding**
   - Only count windows where AUC > 0.5 (above chance)
   - Only count continuous windows (no gaps)

3. **Final Score**
   - Mean AUC across all valid windows

### Tips for Higher Scores

✅ **Smooth Predictions**
- Apply Gaussian smoothing to create continuous high-probability regions
- Reduces spurious spikes

✅ **Ensemble Methods**
- Combine multiple models
- Increases robustness

✅ **Subject-Specific Tuning**
- Calibrate per subject if possible
- Accounts for individual differences

✅ **Post-Processing**
- Apply thresholding
- Remove noise

## Quick Reference: Command Line

```bash
# Generate submission from Python script
python src/submission_generator.py

# Validate submission
python -c "
from src.submission_generator import SubmissionGenerator
import pandas as pd

generator = SubmissionGenerator()
df = pd.read_csv('results/submission.csv')
checks = generator.validate_submission(df)
print('Valid' if all(checks.values()) else 'Invalid')
"
```

## Support

For issues with submission format:

1. **Check format**: Run validation script
2. **Review spec**: Reread this document
3. **Compare example**: Check provided example submission
4. **Debug**: Use verbose output to identify issues

---

**Version**: 1.0.0
**Last Updated**: January 24, 2026
**Status**: Ready for Production
