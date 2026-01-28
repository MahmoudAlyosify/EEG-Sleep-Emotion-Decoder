"""
Submission File Generator for EEG Emotional Memory Classification Challenge

This script generates the submission CSV file with the required format:
{subject_id}_{trial}_{timepoint},prediction

Format:
- ID: S_{subject_id}_{trial_idx}_{timepoint}
- Prediction: Probability (0-1) that the memory is emotional

Test subjects: 3 subjects with multiple trials each
Timepoints: 200 per trial (1 second at 200Hz)
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
import logging
from typing import Tuple, List, Dict
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SubmissionGenerator:
    """
    Generates submission files according to competition specifications.
    """
    
    def __init__(self, base_path=None):
        """
        Initialize submission generator.
        
        Args:
            base_path: Base path to project directory
        """
        if base_path is None:
            base_path = Path(r'd:\Deep Learning & Time Series - predicting-emotions-using-brain-waves\EEG-Sleep-Emotion-Decoder')
        
        self.base_path = Path(base_path)
        self.testing_dir = self.base_path / 'testing'
        self.results_dir = self.base_path / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized submission generator")
        logger.info(f"Testing directory: {self.testing_dir}")
        logger.info(f"Results directory: {self.results_dir}")
    
    def load_test_data(self) -> Dict[str, np.ndarray]:
        """
        Load all test subject data from .mat files.
        
        Returns:
            Dictionary with subject_id -> EEG data mapping
        """
        test_data = {}
        
        if not self.testing_dir.exists():
            logger.warning(f"Testing directory not found: {self.testing_dir}")
            return test_data
        
        # Find all .mat files
        mat_files = list(self.testing_dir.glob('test_subject_*.mat'))
        
        logger.info(f"Found {len(mat_files)} test subject files")
        
        for mat_file in sorted(mat_files):
            try:
                # Extract subject ID from filename
                subject_id = mat_file.stem.replace('test_subject_', '')
                
                # Load data
                mat_data = sio.loadmat(str(mat_file))
                
                # Find EEG data (common keys: 'data', 'EEG', 'signal')
                eeg_data = None
                for key in ['data', 'EEG', 'signal', 'eeg']:
                    if key in mat_data:
                        eeg_data = mat_data[key]
                        break
                
                if eeg_data is None:
                    # Try first non-metadata key
                    for key, val in mat_data.items():
                        if not key.startswith('__') and isinstance(val, np.ndarray):
                            eeg_data = val
                            break
                
                if eeg_data is not None:
                    test_data[subject_id] = eeg_data
                    logger.info(f"  Loaded subject {subject_id}: shape {eeg_data.shape}")
                else:
                    logger.warning(f"  Could not find EEG data in {mat_file.name}")
            
            except Exception as e:
                logger.error(f"  Error loading {mat_file.name}: {e}")
        
        return test_data
    
    def generate_dummy_predictions(self, test_data: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, Dict]:
        """
        Generate dummy predictions for testing submission format.
        
        In production, replace this with actual model predictions.
        
        Args:
            test_data: Dictionary of test subject EEG data
            
        Returns:
            Tuple of (submission_df, metadata)
        """
        submission_rows = []
        metadata = {
            'n_subjects': 0,
            'n_trials': 0,
            'n_timepoints': 0,
            'n_predictions': 0
        }
        
        # Process each test subject
        for subject_id in sorted(test_data.keys()):
            eeg_data = test_data[subject_id]
            
            # Determine shape: could be (channels, trials, timepoints) or (trials, channels, timepoints)
            if eeg_data.ndim == 3:
                n_trials, n_channels, n_timepoints = eeg_data.shape
            elif eeg_data.ndim == 2:
                # Single trial: treat as (channels, timepoints)
                n_trials = 1
                n_channels, n_timepoints = eeg_data.shape
                eeg_data = eeg_data[np.newaxis, :, :]  # Add trial dimension
            else:
                logger.warning(f"Unexpected data shape for subject {subject_id}: {eeg_data.shape}")
                continue
            
            logger.info(f"Subject {subject_id}: {n_trials} trials, {n_timepoints} timepoints")
            
            metadata['n_subjects'] += 1
            metadata['n_trials'] += n_trials
            metadata['n_timepoints'] = max(metadata['n_timepoints'], n_timepoints)
            
            # Generate predictions for each trial and timepoint
            for trial_idx in range(n_trials):
                trial_data = eeg_data[trial_idx]  # (channels, timepoints)
                
                # Simple dummy prediction: based on mean signal power
                # In production, use your trained model here
                mean_power = np.mean(trial_data ** 2)
                base_prob = 0.5 + 0.3 * np.tanh((mean_power - 10) / 10)  # Normalize to ~0.5
                
                for timepoint in range(n_timepoints):
                    # Add small variation per timepoint
                    noise = np.random.normal(0, 0.05)
                    prediction = np.clip(base_prob + noise, 0, 1)
                    
                    sample_id = f"S_{subject_id}_{trial_idx}_{timepoint}"
                    submission_rows.append({
                        'ID': sample_id,
                        'Prediction': prediction
                    })
                    
                    metadata['n_predictions'] += 1
        
        submission_df = pd.DataFrame(submission_rows)
        return submission_df, metadata
    
    def generate_from_predictions(self, 
                                  predictions: np.ndarray,
                                  subject_ids: List[str] = None,
                                  n_timepoints: int = 200) -> pd.DataFrame:
        """
        Generate submission from pre-computed predictions.
        
        Args:
            predictions: Prediction array with shape (n_samples, n_timepoints)
                        or (n_subjects, n_trials, n_timepoints)
            subject_ids: List of subject IDs (if None, uses 1, 2, 3...)
            n_timepoints: Number of timepoints per trial
            
        Returns:
            Submission DataFrame
        """
        submission_rows = []
        
        # Ensure predictions is 2D (n_trials, n_timepoints)
        if predictions.ndim == 3:
            n_subjects, n_trials_per_subject, n_timepoints = predictions.shape
            predictions = predictions.reshape(-1, n_timepoints)
        elif predictions.ndim != 2:
            raise ValueError(f"Expected 2D or 3D predictions, got shape {predictions.shape}")
        
        n_trials, actual_timepoints = predictions.shape
        if n_timepoints != actual_timepoints:
            n_timepoints = actual_timepoints
        
        # Generate subject IDs if not provided
        if subject_ids is None:
            subject_ids = []
            trials_per_subject = n_trials // 3
            for subject_idx in range(1, 4):  # Subjects 1, 2, 3
                for _ in range(trials_per_subject):
                    subject_ids.append(str(subject_idx))
            # Handle remainder
            while len(subject_ids) < n_trials:
                subject_ids.append('3')
        
        # Create submission entries
        for trial_idx in range(n_trials):
            subject_id = subject_ids[trial_idx] if trial_idx < len(subject_ids) else '3'
            
            for timepoint in range(n_timepoints):
                sample_id = f"S_{subject_id}_{trial_idx}_{timepoint}"
                prediction = float(np.clip(predictions[trial_idx, timepoint], 0, 1))
                
                submission_rows.append({
                    'ID': sample_id,
                    'Prediction': prediction
                })
        
        submission_df = pd.DataFrame(submission_rows)
        return submission_df
    
    def validate_submission(self, submission_df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate submission format and content.
        
        Args:
            submission_df: Submission DataFrame
            
        Returns:
            Dictionary of validation checks
        """
        checks = {}
        
        # Check columns
        checks['has_id_column'] = 'ID' in submission_df.columns
        checks['has_prediction_column'] = 'Prediction' in submission_df.columns
        
        # Check data types
        checks['id_is_string'] = submission_df['ID'].dtype == 'object'
        checks['prediction_is_numeric'] = np.issubdtype(submission_df['Prediction'].dtype, np.number)
        
        # Check values
        checks['no_missing_ids'] = submission_df['ID'].isnull().sum() == 0
        checks['no_missing_predictions'] = submission_df['Prediction'].isnull().sum() == 0
        checks['predictions_in_range'] = (
            (submission_df['Prediction'] >= 0).all() and 
            (submission_df['Prediction'] <= 1).all()
        )
        
        # Check ID format
        id_format_valid = True
        for idx_val in submission_df['ID']:
            parts = str(idx_val).split('_')
            if len(parts) != 4 or parts[0] != 'S':
                id_format_valid = False
                break
        checks['id_format_valid'] = id_format_valid
        
        # Check no duplicates
        checks['no_duplicate_ids'] = submission_df['ID'].nunique() == len(submission_df)
        
        return checks
    
    def print_validation_report(self, submission_df: pd.DataFrame, checks: Dict[str, bool]):
        """Print validation report."""
        print("\n" + "="*60)
        print("SUBMISSION VALIDATION REPORT")
        print("="*60)
        
        print(f"\n📊 File Statistics:")
        print(f"  Total entries: {len(submission_df):,}")
        print(f"  Unique subjects: {len(set([s.split('_')[1] for s in submission_df['ID']])))}")
        print(f"  Prediction range: [{submission_df['Prediction'].min():.6f}, {submission_df['Prediction'].max():.6f}]")
        print(f"  Prediction mean: {submission_df['Prediction'].mean():.6f}")
        print(f"  Prediction std: {submission_df['Prediction'].std():.6f}")
        
        print(f"\n✓ Format Checks:")
        check_symbols = {
            True: "✅",
            False: "❌"
        }
        
        for check_name, result in checks.items():
            symbol = check_symbols[result]
            readable_name = check_name.replace('_', ' ').title()
            print(f"  {symbol} {readable_name}: {result}")
        
        print(f"\n📝 Sample Entries (First 10):")
        print(submission_df.head(10).to_string(index=False))
        
        all_valid = all(checks.values())
        status = "✅ VALID" if all_valid else "❌ INVALID"
        print(f"\n{status} - Submission format is {'ready' if all_valid else 'NOT ready'} for upload")
        print("="*60 + "\n")
        
        return all_valid
    
    def save_submission(self, submission_df: pd.DataFrame, filename: str = 'submission.csv') -> Path:
        """
        Save submission to CSV file.
        
        Args:
            submission_df: Submission DataFrame
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.results_dir / filename
        submission_df.to_csv(output_path, index=False)
        
        logger.info(f"✓ Submission saved: {output_path}")
        logger.info(f"  File size: {output_path.stat().st_size:,} bytes")
        
        return output_path
    
    def generate_complete_submission(self):
        """
        Complete submission generation workflow.
        
        Returns:
            Path to generated submission file
        """
        print("\n🚀 EEG EMOTIONAL MEMORY CLASSIFICATION - SUBMISSION GENERATOR")
        print("="*60)
        
        # Step 1: Load test data
        print("\n1️⃣  Loading test data...")
        test_data = self.load_test_data()
        
        if not test_data:
            logger.warning("No test data found. Using dummy predictions.")
            # Create dummy test data for demonstration
            test_data = {
                '1': np.random.randn(16, 200),
                '7': np.random.randn(16, 200),
                '12': np.random.randn(16, 200),
            }
        
        # Step 2: Generate predictions
        print("\n2️⃣  Generating predictions...")
        submission_df, metadata = self.generate_dummy_predictions(test_data)
        print(f"✓ Generated {len(submission_df):,} predictions")
        print(f"  Subjects: {metadata['n_subjects']}")
        print(f"  Trials: {metadata['n_trials']}")
        print(f"  Timepoints per trial: {metadata['n_timepoints']}")
        
        # Step 3: Validate submission
        print("\n3️⃣  Validating submission format...")
        checks = self.validate_submission(submission_df)
        is_valid = self.print_validation_report(submission_df, checks)
        
        # Step 4: Save submission
        print("\n4️⃣  Saving submission file...")
        output_path = self.save_submission(submission_df)
        
        print("\n✅ SUBMISSION GENERATION COMPLETE")
        print(f"\nSubmission file: {output_path}")
        print(f"Ready for upload: {is_valid}")
        
        return output_path


def main():
    """Main execution."""
    # Initialize generator
    generator = SubmissionGenerator()
    
    # Generate complete submission
    submission_path = generator.generate_complete_submission()
    
    return submission_path


if __name__ == '__main__':
    submission_file = main()
