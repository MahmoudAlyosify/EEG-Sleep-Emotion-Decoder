"""Try the complement: if emo/neu predictions score 0.500, try 1-predictions"""

import pandas as pd

# Load our best submission
df = pd.read_csv('submission_correct_labels_rf.csv')

# Try inverting predictions
df_inverted = df.copy()
df_inverted['prediction'] = 1 - df['prediction']
df_inverted['prediction'] = df_inverted['prediction'].clip(0.01, 0.99)

df_inverted.to_csv('submission_inverted_rf.csv', index=False)

print("Original RF submission:")
print(f"  Range: [{df.prediction.min():.4f}, {df.prediction.max():.4f}]")
print(f"  Mean: {df.prediction.mean():.4f}")

print("\nInverted submission (1 - predictions):")
print(f"  Range: [{df_inverted.prediction.min():.4f}, {df_inverted.prediction.max():.4f}]")
print(f"  Mean: {df_inverted.prediction.mean():.4f}")

print("\n✓ Created submission_inverted_rf.csv")
