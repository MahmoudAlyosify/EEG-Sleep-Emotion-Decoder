#!/usr/bin/env python
"""
Standalone SOTA Pipeline Execution Script (Optimized)
"""

import sys
import os
import numpy as np

# Add workspace to path
sys.path.insert(0, r'D:\Deep Learning & Time Series - predicting-emotions-using-brain-waves')

from sota_pipeline import *
import time

def main():
    print("="*80)
    print("STATE-OF-THE-ART EEG EMOTIONAL MEMORY CLASSIFICATION PIPELINE")
    print("="*80)
    
    TRAIN_PATH = r'D:\Deep Learning & Time Series - predicting-emotions-using-brain-waves\training'
    TEST_PATH = r'D:\Deep Learning & Time Series - predicting-emotions-using-brain-waves\testing'
    
    start_time = time.time()
    
    try:
        # Initialize pipeline
        print("\n[1/6] Initializing pipeline...")
        pipeline = SOTAEEGPipeline(TRAIN_PATH, TEST_PATH)
        
        # Load data
        print("[2/6] Loading training and test data...")
        pipeline.load_data()
        print(f"      Training shape: {pipeline.X_train.shape}")
        print(f"      Test shape: {pipeline.X_test.shape}")
        
        # Preprocess
        print("[3/6] Advanced preprocessing...")
        print("      - Bandpass filtering (0.5-40 Hz)")
        print("      - Euclidean Alignment")
        pipeline.preprocess()
        print("      ✓ Preprocessing complete")
        
        # Train models with reduced epochs for faster execution
        print("[4/6] Training Model A (Deep Learning - EEG-TCNet)...")
        print("      Training 10 epochs (reduced for demo)...")
        pipeline.train_model_a(n_epochs=10)
        print("      ✓ Model A trained")
        
        print("[5/6] Training Model B (Riemannian Geometry)...")
        pipeline.train_model_b()
        print("      ✓ Model B trained")
        
        # Generate predictions
        print("[6/6] Generating ensemble predictions...")
        pred_a = pipeline.model_a.predict(pipeline.X_test, verbose=0).squeeze(axis=2)
        pred_b = pipeline.model_b.predict_proba(pipeline.X_test)
        predictions = ensemble_predictions(pred_a, pred_b, weight_a=0.6, weight_b=0.4)
        predictions = apply_gaussian_smoothing(predictions, sigma=2.0)
        print(f"      Predictions shape: {predictions.shape}")
        print(f"      Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        
        # Create submission
        print("\nCreating submission file...")
        test_files = sorted(glob.glob(os.path.join(TEST_PATH, '*.mat')))
        subject_ids = []
        trial_counts = []
        
        for f in test_files:
            basename = os.path.basename(f)
            subject_id = int(basename.split('_')[-1].replace('.mat', ''))
            subject_ids.append(subject_id)
            data = load_hdf5_data(f)
            n_trials = data['trial'].shape[0]
            trial_counts.append(n_trials)
        
        rows = []
        pred_idx = 0
        
        for subject_idx, (subject_id, n_trials) in enumerate(zip(subject_ids, trial_counts)):
            for trial_idx in range(n_trials):
                pred_trial = predictions[pred_idx]
                
                for timepoint_idx in range(pred_trial.shape[0]):
                    row = {
                        'id': f"{subject_id}_{trial_idx}_{timepoint_idx}",
                        'prediction': float(pred_trial[timepoint_idx])
                    }
                    rows.append(row)
                
                pred_idx += 1
        
        submission = pd.DataFrame(rows)
        output_file = 'submission_sota_ensemble.csv'
        submission.to_csv(output_file, index=False)
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE ✓")
        print("="*80)
        print(f"Submission file: {output_file}")
        print(f"Total rows: {len(submission)}")
        print(f"Predictions - Min: {submission['prediction'].min():.6f}")
        print(f"Predictions - Max: {submission['prediction'].max():.6f}")
        print(f"Predictions - Mean: {submission['prediction'].mean():.6f}")
        print(f"Predictions - Std: {submission['prediction'].std():.6f}")
        print(f"Total time: {elapsed_time/60:.1f} minutes")
        print("="*80)
        
        return submission
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    submission = main()
