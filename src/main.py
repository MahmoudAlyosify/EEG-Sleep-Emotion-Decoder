"""
Main training and inference pipeline for EEG Emotional Memory Classification.
Implements the complete hybrid ensemble approach.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
from pathlib import Path
import json
import logging
from tqdm import tqdm
from datetime import datetime

from preprocessing import EEGPreprocessor, SlidingWindowProcessor
from models import (
    EEGTCNet, RiemannianSVMClassifier, EnsembleEEGClassifier,
    apply_gaussian_smoothing
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EEGTrainingPipeline:
    """
    Complete training pipeline for EEG classification.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize pipeline.
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Configuration: {self.config}")
        
        # Initialize components
        self.preprocessor = EEGPreprocessor(
            low_freq=self.config['preprocessing']['low_freq'],
            high_freq=self.config['preprocessing']['high_freq'],
            srate=self.config['preprocessing']['sampling_rate']
        )
        
        self.window_processor = SlidingWindowProcessor(
            window_size=self.config['windowing']['window_size'],
            stride=self.config['windowing']['stride']
        )
        
        self.tcn_model = None
        self.riemannian_model = None
        self.ensemble_model = None
        
        self.train_history = {'loss': [], 'acc': []}
        self.val_history = {'loss': [], 'acc': []}
    
    def _load_config(self, config_path=None):
        """Load configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'preprocessing': {
                'low_freq': 0.5,
                'high_freq': 40,
                'sampling_rate': 200
            },
            'windowing': {
                'window_size': 200,
                'stride': 1
            },
            'model': {
                'n_channels': 32,
                'n_kernels': 64,
                'dropout': 0.5,
                'tcn_weight': 0.6,
                'riemannian_weight': 0.4
            },
            'training': {
                'batch_size': 32,
                'epochs': 100,
                'learning_rate': 0.001,
                'weight_decay': 1e-5
            },
            'postprocessing': {
                'gaussian_sigma': 2.0
            }
        }
    
    def load_mat_data(self, data_path, label=None):
        """
        Load .mat files.
        
        Args:
            data_path: Path to .mat file
            label: Class label for this sample
            
        Returns:
            Dictionary with data and metadata
        """
        mat_data = sio.loadmat(data_path)
        
        # Extract EEG data (common keys: 'data', 'EEG', 'signal')
        eeg_data = None
        for key in ['data', 'EEG', 'signal', 'eeg']:
            if key in mat_data:
                eeg_data = mat_data[key]
                break
        
        if eeg_data is None:
            # Try the first non-metadata key
            for key, val in mat_data.items():
                if not key.startswith('__') and isinstance(val, np.ndarray):
                    eeg_data = val
                    break
        
        return {
            'signal': eeg_data,
            'label': label,
            'path': str(data_path)
        }
    
    def prepare_data(self, train_data_list, val_data_list=None):
        """
        Prepare data for training.
        
        Args:
            train_data_list: List of (data_path, label) tuples
            val_data_list: Optional list of (data_path, label) tuples
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val) or (X_train, y_train, None, None)
        """
        logger.info(f"Loading {len(train_data_list)} training samples...")
        
        X_train_list = []
        y_train_list = []
        
        for data_path, label in tqdm(train_data_list, desc="Loading training data"):
            sample = self.load_mat_data(data_path, label)
            signal = sample['signal']
            
            # Preprocess
            signal = self.preprocessor.preprocess(signal)
            X_train_list.append(signal)
            y_train_list.append(label)
        
        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)
        
        X_val, y_val = None, None
        if val_data_list:
            logger.info(f"Loading {len(val_data_list)} validation samples...")
            X_val_list = []
            y_val_list = []
            
            for data_path, label in tqdm(val_data_list, desc="Loading validation data"):
                sample = self.load_mat_data(data_path, label)
                signal = sample['signal']
                signal = self.preprocessor.preprocess(signal)
                X_val_list.append(signal)
                y_val_list.append(label)
            
            X_val = np.array(X_val_list)
            y_val = np.array(y_val_list)
        
        return X_train, y_train, X_val, y_val
    
    def train_tcn_model(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train TCN model.
        
        Args:
            X_train: Training signals (n_samples, n_channels, n_timepoints)
            y_train: Training labels (n_samples,)
            X_val: Validation signals (optional)
            y_val: Validation labels (optional)
        """
        logger.info("Training TCN model...")
        
        # Initialize model
        self.tcn_model = EEGTCNet(
            n_channels=X_train.shape[1],
            n_timepoints=X_train.shape[2],
            n_classes=2,
            dropout=self.config['model']['dropout'],
            n_kernels=self.config['model']['n_kernels']
        ).to(self.device)
        
        # Create data loaders
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        
        # Optimizer and loss
        optimizer = optim.Adam(
            self.tcn_model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(self.config['training']['epochs']):
            self.tcn_model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            
            for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()
                
                # Forward pass
                logits = self.tcn_model(batch_x)  # (batch, 2, timepoints)
                
                # Temporal average pooling for classification loss
                logits_avg = logits.mean(dim=2)  # (batch, 2)
                loss = criterion(logits_avg, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Accuracy
                preds = torch.argmax(logits_avg, dim=1)
                epoch_acc += (preds == batch_y).float().mean().item()
            
            epoch_loss /= len(train_loader)
            epoch_acc /= len(train_loader)
            
            self.train_history['loss'].append(epoch_loss)
            self.train_history['acc'].append(epoch_acc)
            
            logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                       f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
    
    def train_riemannian_model(self, X_train, y_train):
        """
        Train Riemannian Geometry model.
        
        Args:
            X_train: Training signals (n_samples, n_channels, n_timepoints)
            y_train: Training labels (n_samples,)
        """
        logger.info("Training Riemannian model...")
        
        # Compute covariance matrices
        cov_matrices = np.array([np.cov(x) for x in X_train])
        
        # Initialize and fit classifier
        self.riemannian_model = RiemannianSVMClassifier(method='svm')
        self.riemannian_model.fit(cov_matrices, y_train)
        
        logger.info("Riemannian model trained successfully")
    
    def create_ensemble(self):
        """Create ensemble model."""
        if self.tcn_model is None or self.riemannian_model is None:
            raise ValueError("Both TCN and Riemannian models must be trained first")
        
        self.ensemble_model = EnsembleEEGClassifier(
            self.tcn_model,
            self.riemannian_model,
            tcn_weight=self.config['model']['tcn_weight'],
            riemannian_weight=self.config['model']['riemannian_weight']
        )
        
        logger.info("Ensemble model created")
    
    def predict(self, X_test, apply_smoothing=True):
        """
        Make predictions on test data.
        
        Args:
            X_test: Test signals (n_samples, n_channels, n_timepoints)
            apply_smoothing: Whether to apply Gaussian smoothing
            
        Returns:
            Predictions (n_samples, n_timepoints)
        """
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not created")
        
        logger.info(f"Making predictions on {len(X_test)} samples...")
        
        # Compute covariance matrices
        cov_matrices = np.array([np.cov(x) for x in X_test])
        
        # Convert to torch tensor
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Ensemble predictions
        predictions = self.ensemble_model.predict_ensemble(X_test_tensor, cov_matrices)
        
        # Apply smoothing
        if apply_smoothing:
            predictions = apply_gaussian_smoothing(
                predictions,
                sigma=self.config['postprocessing']['gaussian_sigma']
            )
        
        return predictions
    
    def save_model(self, save_dir='models'):
        """Save trained models."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Save TCN model
        if self.tcn_model:
            torch.save(
                self.tcn_model.state_dict(),
                save_path / 'tcn_model.pt'
            )
        
        # Save Riemannian model
        if self.riemannian_model:
            import pickle
            with open(save_path / 'riemannian_model.pkl', 'wb') as f:
                pickle.dump(self.riemannian_model, f)
        
        logger.info(f"Models saved to {save_path}")
    
    def load_model(self, model_dir='models'):
        """Load trained models."""
        model_path = Path(model_dir)
        
        if (model_path / 'tcn_model.pt').exists():
            self.tcn_model = EEGTCNet().to(self.device)
            self.tcn_model.load_state_dict(
                torch.load(model_path / 'tcn_model.pt', map_location=self.device)
            )
        
        if (model_path / 'riemannian_model.pkl').exists():
            import pickle
            with open(model_path / 'riemannian_model.pkl', 'rb') as f:
                self.riemannian_model = pickle.load(f)
        
        logger.info(f"Models loaded from {model_path}")


if __name__ == '__main__':
    # Example usage
    pipeline = EEGTrainingPipeline()
    
    # Example data paths (replace with actual paths)
    # train_data = [
    #     ('data/training/sleep_emo/S_2_cleaned.mat', 1),
    #     ('data/training/sleep_neu/S_2_cleaned.mat', 0),
    # ]
    # 
    # X_train, y_train, _, _ = pipeline.prepare_data(train_data)
    # pipeline.train_tcn_model(X_train, y_train)
    # pipeline.train_riemannian_model(X_train, y_train)
    # pipeline.create_ensemble()
    # 
    # predictions = pipeline.predict(X_train)
    # pipeline.save_model()
