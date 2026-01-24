import pandas as pd
import glob

submissions = {
    'submission_correct_labels_rf.csv': 'RF with correct labels (emo/neu)',
    'submission_advanced_ensemble.csv': 'Advanced ensemble (freq features)',
}

print("="*90)
print("SUBMISSION COMPARISON")
print("="*90)

for fname, desc in submissions.items():
    try:
        df = pd.read_csv(fname)
        print(f"\n{fname}")
        print(f"  Description: {desc}")
        print(f"  Rows: {len(df)}")
        print(f"  Pred range: [{df.prediction.min():.4f}, {df.prediction.max():.4f}]")
        print(f"  Pred mean: {df.prediction.mean():.4f}, std: {df.prediction.std():.4f}")
    except Exception as e:
        print(f"\n{fname}: ERROR - {e}")
