"""
Deep Learning and Riemannian Geometry models for EEG classification.
Includes TCN, ATCNet for temporal analysis and Riemannian methods for spatial analysis.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from pyriemann.tangentspace import TangentSpace
from scipy.linalg import sqrtm
import warnings

warnings.filterwarnings('ignore')


class TemporalConvolutionalBlock(nn.Module):
    """
    Temporal Convolutional Block with residual connections.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.5):
        """
        Initialize TCN block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            dilation: Dilation rate
            dropout: Dropout rate
        """
        super(TemporalConvolutionalBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )
        
        self.net = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.Dropout(dropout),
            self.conv2,
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class AttentionLayer(nn.Module):
    """
    Attention mechanism for temporal focusing.
    """
    
    def __init__(self, channels):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv1d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        att = self.attention(x)
        return x * att


class EEGTCNet(nn.Module):
    """
    Modified EEG-TCNet with dense prediction output.
    Outputs probabilities for every timepoint (200 predictions per trial).
    """
    
    def __init__(self, n_channels=32, n_timepoints=200, n_classes=2, 
                 dropout=0.5, n_kernels=64):
        """
        Initialize EEG-TCNet.
        
        Args:
            n_channels: Number of EEG channels
            n_timepoints: Number of timepoints per trial
            n_classes: Number of output classes
            dropout: Dropout rate
            n_kernels: Number of convolutional kernels
        """
        super(EEGTCNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.n_classes = n_classes
        
        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv1d(n_channels, n_kernels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_kernels),
            nn.ReLU()
        )
        
        # TCN blocks with increasing dilation
        self.tcn_blocks = nn.ModuleList([
            TemporalConvolutionalBlock(n_kernels, n_kernels * 2, 3, dilation=1, dropout=dropout),
            TemporalConvolutionalBlock(n_kernels * 2, n_kernels * 2, 3, dilation=2, dropout=dropout),
            TemporalConvolutionalBlock(n_kernels * 2, n_kernels * 4, 3, dilation=4, dropout=dropout),
        ])
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            AttentionLayer(n_kernels * 2),
            AttentionLayer(n_kernels * 2),
            AttentionLayer(n_kernels * 4),
        ])
        
        # Dense prediction layer (one prediction per timepoint)
        self.dense_pred = nn.Sequential(
            nn.Conv1d(n_kernels * 4, n_kernels * 2, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_kernels * 2, n_classes, 1)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input (batch_size, n_channels, n_timepoints)
            
        Returns:
            Dense predictions (batch_size, n_classes, n_timepoints)
        """
        # Initial convolution
        x = self.init_conv(x)
        
        # TCN blocks with attention
        for tcn_block, attention in zip(self.tcn_blocks, self.attention_layers):
            x = tcn_block(x)
            x = attention(x)
        
        # Dense predictions
        out = self.dense_pred(x)
        
        return out
    
    def predict_timepoints(self, x):
        """
        Get probability predictions for each timepoint.
        
        Args:
            x: Input (batch_size, n_channels, n_timepoints)
            
        Returns:
            Probabilities (batch_size, n_timepoints) for binary classification
        """
        logits = self.forward(x)  # (batch, 2, timepoints)
        probs = torch.softmax(logits, dim=1)  # (batch, 2, timepoints)
        return probs[:, 1, :]  # Return positive class probability


class RiemannianSVMClassifier:
    """
    Riemannian Geometry classifier using Tangent Space Mapping + SVM/Logistic Regression.
    Captures spatial patterns in EEG covariance structures.
    """
    
    def __init__(self, method='svm', n_components=100):
        """
        Initialize classifier.
        
        Args:
            method: 'svm' or 'logistic'
            n_components: Number of components in tangent space
        """
        self.method = method
        self.tangent_space_mapper = TangentSpace(n_components=n_components)
        
        if method == 'svm':
            self.classifier = SVC(kernel='rbf', C=1.0, probability=True)
        elif method == 'logistic':
            self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit(self, cov_matrices, labels):
        """
        Fit Riemannian classifier.
        
        Args:
            cov_matrices: (n_samples, n_channels, n_channels)
            labels: (n_samples,)
        """
        # Map to tangent space
        X_tangent = self.tangent_space_mapper.fit_transform(cov_matrices)
        
        # Fit classifier
        self.classifier.fit(X_tangent, labels)
    
    def predict(self, cov_matrices):
        """
        Make predictions.
        
        Args:
            cov_matrices: (n_samples, n_channels, n_channels)
            
        Returns:
            Predictions (n_samples,)
        """
        X_tangent = self.tangent_space_mapper.transform(cov_matrices)
        return self.classifier.predict(X_tangent)
    
    def predict_proba(self, cov_matrices):
        """
        Get probability predictions.
        
        Args:
            cov_matrices: (n_samples, n_channels, n_channels)
            
        Returns:
            Probabilities (n_samples, n_classes)
        """
        X_tangent = self.tangent_space_mapper.transform(cov_matrices)
        return self.classifier.predict_proba(X_tangent)


class EnsembleEEGClassifier:
    """
    Ensemble combining Deep Learning (TCN) and Riemannian Geometry methods.
    """
    
    def __init__(self, tcn_model, riemannian_model, 
                 tcn_weight=0.6, riemannian_weight=0.4):
        """
        Initialize ensemble.
        
        Args:
            tcn_model: EEGTCNet model
            riemannian_model: RiemannianSVMClassifier
            tcn_weight: Weight for TCN predictions
            riemannian_weight: Weight for Riemannian predictions
        """
        self.tcn_model = tcn_model
        self.riemannian_model = riemannian_model
        self.tcn_weight = tcn_weight
        self.riemannian_weight = riemannian_weight
    
    def predict_ensemble(self, raw_signals, cov_matrices):
        """
        Make ensemble predictions.
        
        Args:
            raw_signals: (batch, channels, timepoints) as torch tensor
            cov_matrices: (batch, channels, channels) as numpy array
            
        Returns:
            Ensemble predictions (batch, timepoints) for binary classification
        """
        # TCN predictions
        with torch.no_grad():
            tcn_probs = self.tcn_model.predict_timepoints(raw_signals)  # (batch, timepoints)
            tcn_probs = tcn_probs.cpu().numpy()
        
        # Riemannian predictions (need to expand to timepoints)
        riemannian_probs = self.riemannian_model.predict_proba(cov_matrices)  # (batch, 2)
        riemannian_probs = riemannian_probs[:, 1]  # (batch,) - positive class
        riemannian_probs = np.tile(riemannian_probs[:, np.newaxis], (1, tcn_probs.shape[1]))
        
        # Weighted ensemble
        ensemble_probs = (self.tcn_weight * tcn_probs + 
                         self.riemannian_weight * riemannian_probs)
        
        return ensemble_probs


def apply_gaussian_smoothing(predictions, sigma=2.0):
    """
    Apply Gaussian smoothing to predictions for continuous high-probability windows.
    
    Args:
        predictions: (batch, timepoints) or (timepoints,)
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Smoothed predictions with same shape
    """
    from scipy.ndimage import gaussian_filter1d
    
    if predictions.ndim == 1:
        return gaussian_filter1d(predictions, sigma=sigma)
    else:
        return np.array([gaussian_filter1d(pred, sigma=sigma) for pred in predictions])
