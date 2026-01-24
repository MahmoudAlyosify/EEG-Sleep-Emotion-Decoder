#!/usr/bin/env python
"""
Run Enhanced SOTA Pipeline for AUC > 0.66
"""

import sys
import os
sys.path.insert(0, r'D:\Deep Learning & Time Series - predicting-emotions-using-brain-waves')

from enhanced_sota_pipeline import EnhancedSOTAPipeline
import time

print("="*80)
print("ENHANCED SOTA PIPELINE - Targeting AUC > 0.66")
print("="*80)

TRAIN_PATH = r'D:\Deep Learning & Time Series - predicting-emotions-using-brain-waves\training'
TEST_PATH = r'D:\Deep Learning & Time Series - predicting-emotions-using-brain-waves\testing'

start_time = time.time()

try:
    print("\n[1/4] Loading data...")
    pipeline = EnhancedSOTAPipeline(TRAIN_PATH, TEST_PATH)
    pipeline.load_data()
    
    print("\n[2/4] Training enhanced ensemble...")
    pipeline.train()
    
    print("\n[3/4] Generating predictions...")
    predictions = pipeline.predict()
    
    print("\n[4/4] Creating submission...")
    submission = pipeline.create_submission(predictions, 'submission_enhanced_auc66.csv')
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE ✓")
    print("="*80)
    print(f"Submission file: submission_enhanced_auc66.csv")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print("="*80)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
