
# ============================================================================
# EEG Submission Ensemble — v10.2 + v10.3
# ============================================================================
# Creates multiple ensemble submissions from existing CSV files:
#   1) Average ensemble
#   2) Rank-average ensemble
#   3) Weighted average (35% v10.2 + 65% v10.3)
#   4) Weighted average (25% v10.2 + 75% v10.3)
#   5) Weighted rank-average (35% / 65%)
#
# Same project paths as your environment.
# ============================================================================

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = r'D:\EEG Project\Project Overview and Specifications\eeg_competition'
FILE_V102 = os.path.join(BASE, 'submission_v10_2_temporal_modulator.csv')
FILE_V103 = os.path.join(BASE, 'submission_v10_3_winner_refined.csv')
OUT_DIR = os.path.join(BASE, 'ensemble_outputs_v10_2_v10_3')
os.makedirs(OUT_DIR, exist_ok=True)

# Optional additional input (ignored if missing)
FILE_V102B = os.path.join(BASE, 'submission_v10_2b_no_platt.csv')


def _load_submission(path):
    df = pd.read_csv(path)
    assert 'id' in df.columns and 'prediction' in df.columns, f'Missing columns in {path}'
    assert df['prediction'].between(0, 1).all(), f'Predictions out of [0,1] in {path}'
    return df


def _check_alignment(df1, df2, name1='A', name2='B'):
    assert len(df1) == len(df2), f'Length mismatch: {name1}={len(df1)} vs {name2}={len(df2)}'
    assert (df1['id'].values == df2['id'].values).all(), f'ID mismatch between {name1} and {name2}'


def _save(ids, preds, out_path, label):
    df = pd.DataFrame({'id': ids, 'prediction': np.clip(preds, 0.0, 1.0)})
    assert df['prediction'].between(0, 1).all()
    assert len(str(df.iloc[0]['id']).split('_')) == 3
    df.to_csv(out_path, index=False)
    print(f'✓ Saved {label:<24s} → {out_path}')
    print(f'  range=[{df.prediction.min():.6f}, {df.prediction.max():.6f}] mean={df.prediction.mean():.6f}')
    return df


def average_ensemble(p1, p2, w1=0.5, w2=0.5):
    return (w1 * p1 + w2 * p2) / (w1 + w2)


def rank_average_ensemble(p1, p2, w1=0.5, w2=0.5):
    r1 = rankdata(p1, method='average') / len(p1)
    r2 = rankdata(p2, method='average') / len(p2)
    return (w1 * r1 + w2 * r2) / (w1 + w2)


def average_three(p1, p2, p3, w1=0.2, w2=0.3, w3=0.5):
    return (w1*p1 + w2*p2 + w3*p3) / (w1+w2+w3)


def main():
    print('=' * 78)
    print(' EEG Submission Ensemble — v10.2 + v10.3')
    print('=' * 78)
    print(f'BASE    : {BASE}')
    print(f'V10.2   : {FILE_V102}')
    print(f'V10.3   : {FILE_V103}')
    print(f'OUT_DIR : {OUT_DIR}')

    df102 = _load_submission(FILE_V102)
    df103 = _load_submission(FILE_V103)
    _check_alignment(df102, df103, 'v10.2', 'v10.3')

    ids = df102['id'].values
    p102 = df102['prediction'].values.astype(np.float64)
    p103 = df103['prediction'].values.astype(np.float64)

    print('\nInput summaries:')
    print(f' v10.2  -> range=[{p102.min():.6f}, {p102.max():.6f}] mean={p102.mean():.6f}')
    print(f' v10.3  -> range=[{p103.min():.6f}, {p103.max():.6f}] mean={p103.mean():.6f}')

    out_avg = os.path.join(OUT_DIR, 'submission_ensemble_avg_v102_v103.csv')
    out_rank = os.path.join(OUT_DIR, 'submission_ensemble_rankavg_v102_v103.csv')
    out_w3565 = os.path.join(OUT_DIR, 'submission_ensemble_weighted_35_65_v102_v103.csv')
    out_w2575 = os.path.join(OUT_DIR, 'submission_ensemble_weighted_25_75_v102_v103.csv')
    out_rankw = os.path.join(OUT_DIR, 'submission_ensemble_rank_weighted_35_65_v102_v103.csv')

    _save(ids, average_ensemble(p102, p103, 0.5, 0.5), out_avg, 'average 50/50')
    _save(ids, rank_average_ensemble(p102, p103, 0.5, 0.5), out_rank, 'rank-average 50/50')
    _save(ids, average_ensemble(p102, p103, 0.35, 0.65), out_w3565, 'weighted avg 35/65')
    _save(ids, average_ensemble(p102, p103, 0.25, 0.75), out_w2575, 'weighted avg 25/75')
    _save(ids, rank_average_ensemble(p102, p103, 0.35, 0.65), out_rankw, 'weighted rank 35/65')

    # Optional 3-way ensemble if v10.2b exists
    if os.path.exists(FILE_V102B):
        print('\nOptional input found:')
        print(f' v10.2b : {FILE_V102B}')
        df102b = _load_submission(FILE_V102B)
        _check_alignment(df102, df102b, 'v10.2', 'v10.2b')
        p102b = df102b['prediction'].values.astype(np.float64)
        out_three = os.path.join(OUT_DIR, 'submission_ensemble_threeway_20_30_50_v102_v102b_v103.csv')
        _save(ids, average_three(p102, p102b, p103, 0.20, 0.30, 0.50), out_three, '3-way avg 20/30/50')

    print('\nDone. Suggested upload order:')
    print('  1) submission_ensemble_rankavg_v102_v103.csv')
    print('  2) submission_ensemble_weighted_25_75_v102_v103.csv')
    print('  3) submission_ensemble_weighted_35_65_v102_v103.csv')
    print('  4) submission_ensemble_rank_weighted_35_65_v102_v103.csv')
    print('  5) submission_ensemble_avg_v102_v103.csv')


if __name__ == '__main__':
    main()
