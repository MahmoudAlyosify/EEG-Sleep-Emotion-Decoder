"""
Preprocessing module for EEG data.
Includes Euclidean Alignment (EA) and bandpass filtering.
"""

import numpy as np
from scipy import signal
from scipy.linalg import sqrtm
import warnings

warnings.filterwarnings('ignore')


class EEGPreprocessor:
    """
    Advanced EEG preprocessing with Euclidean Alignment and filtering.
    """
    
    def __init__(self, low_freq=0.5, high_freq=40, srate=200):
        """
        Initialize preprocessor.
        
        Args:
            low_freq: Low cutoff frequency (Hz)
            high_freq: High cutoff frequency (Hz)
            srate: Sampling rate (Hz)
        """
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.srate = srate
        self.reference_matrix = None
        
    def bandpass_filter(self, data, low_freq=None, high_freq=None, order=4):
        """
        Apply Butterworth bandpass filter to EEG data.
        
        Args:
            data: Input signal (channels, samples) or (samples,)
            low_freq: Low cutoff frequency
            high_freq: High cutoff frequency
            order: Filter order
            
        Returns:
            Filtered signal with same shape as input
        """
        if low_freq is None:
            low_freq = self.low_freq
        if high_freq is None:
            high_freq = self.high_freq
            
        # Normalize frequencies to Nyquist
        nyquist = self.srate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Ensure frequencies are in valid range
        low = np.clip(low, 0.001, 0.999)
        high = np.clip(high, low + 0.001, 0.999)
        
        # Design filter
        sos = signal.butter(order, [low, high], btype='band', output='sos')
        
        # Apply filter
        if data.ndim == 1:
            return signal.sosfilt(sos, data)
        else:
            return np.array([signal.sosfilt(sos, ch) for ch in data])
    
    def euclidean_alignment(self, data, reference=None, return_reference=False):
        """
        Apply Euclidean Alignment (EA) to normalize covariance structure.
        
        EA aligns the covariance matrix of each trial to a reference by
        finding the optimal rotation.
        
        Args:
            data: Input signal (channels, samples)
            reference: Reference covariance matrix. If None, uses identity.
            return_reference: If True, returns the reference matrix
            
        Returns:
            Aligned signal or (aligned_signal, reference_matrix)
        """
        # Compute covariance
        cov = np.cov(data)
        
        # Use identity as reference if not provided
        if reference is None:
            reference = np.eye(cov.shape[0])
            self.reference_matrix = reference
        
        # Compute square root of covariance
        sqrt_cov = sqrtm(cov)
        
        # Compute alignment matrix
        inv_sqrt_cov = np.linalg.inv(sqrt_cov)
        sqrt_ref = sqrtm(reference)
        
        # Alignment transformation
        alignment_matrix = sqrt_ref @ inv_sqrt_cov
        
        # Apply alignment
        aligned_data = alignment_matrix @ data
        
        if return_reference:
            return aligned_data, reference
        return aligned_data
    
    def preprocess(self, data, apply_ea=True, apply_filter=True):
        """
        Complete preprocessing pipeline.
        
        Args:
            data: Raw EEG data (channels, samples)
            apply_ea: Whether to apply Euclidean Alignment
            apply_filter: Whether to apply bandpass filter
            
        Returns:
            Preprocessed data
        """
        processed = data.copy()
        
        # Apply bandpass filter
        if apply_filter:
            processed = self.bandpass_filter(processed)
        
        # Apply Euclidean Alignment
        if apply_ea:
            processed = self.euclidean_alignment(processed)
        
        return processed
    
    def normalize_data(self, data, axis=None):
        """
        Normalize data to zero mean and unit variance.
        
        Args:
            data: Input data
            axis: Axis along which to normalize (default: all)
            
        Returns:
            Normalized data
        """
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        return (data - mean) / (std + 1e-8)


class SlidingWindowProcessor:
    """
    Sliding window processor for creating time-resolved samples.
    """
    
    def __init__(self, window_size=200, stride=1):
        """
        Initialize sliding window processor.
        
        Args:
            window_size: Size of each window
            stride: Step size between windows
        """
        self.window_size = window_size
        self.stride = stride
    
    def create_windows(self, data, labels=None):
        """
        Create sliding windows from continuous data.
        
        Args:
            data: Input signal (channels, samples) or (samples,)
            labels: Optional labels array (samples,)
            
        Returns:
            windows: (n_windows, channels, window_size)
            window_labels: (n_windows,) if labels provided, else None
        """
        if data.ndim == 1:
            data = data[np.newaxis, :]
        
        n_channels, n_samples = data.shape
        windows = []
        window_labels = [] if labels is not None else None
        
        for start in range(0, n_samples - self.window_size + 1, self.stride):
            end = start + self.window_size
            window = data[:, start:end]
            windows.append(window)
            
            if labels is not None:
                # Use majority label in window or mean if continuous
                window_label = np.mean(labels[start:end])
                window_labels.append(window_label)
        
        windows = np.array(windows)
        
        if labels is not None:
            return windows, np.array(window_labels)
        return windows
    
    def covariance_matrices(self, windows):
        """
        Compute covariance matrices for each window.
        Used for Riemannian Geometry methods.
        
        Args:
            windows: (n_windows, channels, window_size)
            
        Returns:
            cov_matrices: (n_windows, channels, channels)
        """
        n_windows = windows.shape[0]
        n_channels = windows.shape[1]
        cov_matrices = np.zeros((n_windows, n_channels, n_channels))
        
        for i in range(n_windows):
            cov_matrices[i] = np.cov(windows[i])
        
        return cov_matrices
