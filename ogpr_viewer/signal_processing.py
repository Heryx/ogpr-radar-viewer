"""
GPR Signal Processing Module

Implements various signal processing algorithms for GPR data:
- Background removal
- Bandpass filtering
- Gain functions
- Hilbert transform
- Time-zero correction
"""

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter
from typing import Tuple, Optional, Literal


class SignalProcessor:
    """
    Signal processing for GPR data.
    """
    
    def __init__(self, data: np.ndarray, sampling_time_ns: float = 0.125):
        """
        Initialize processor with radar data.
        
        Args:
            data: 2D array of shape (samples, traces) or 3D (samples, channels, slices)
            sampling_time_ns: Time sampling interval in nanoseconds
        """
        self.original_data = data.copy()
        self.data = data.copy()
        self.sampling_time_ns = sampling_time_ns
        self.sampling_freq_mhz = 1000.0 / sampling_time_ns  # Convert to MHz
        
        # Get 2D data for processing
        if data.ndim == 3:
            # Use first channel if 3D
            self.data_2d = data[:, 0, :]
        else:
            self.data_2d = data
        
        self.processed_data = self.data_2d.copy()
    
    def remove_background(self, method: Literal['mean', 'median'] = 'mean') -> np.ndarray:
        """
        Remove background and horizontal banding.
        
        Args:
            method: 'mean' or 'median' for background estimation
        
        Returns:
            Background-removed data
        """
        data = self.processed_data.copy()
        
        if method == 'mean':
            # Subtract mean trace (removes DC and vertical coherent noise)
            mean_trace = np.mean(data, axis=1, keepdims=True)
            data = data - mean_trace
        elif method == 'median':
            # Subtract median trace (more robust to outliers)
            median_trace = np.median(data, axis=1, keepdims=True)
            data = data - median_trace
        
        self.processed_data = data
        return data
    
    def apply_bandpass(self, 
                       low_freq: float, 
                       high_freq: float, 
                       order: int = 4) -> np.ndarray:
        """
        Apply Butterworth bandpass filter.
        
        Args:
            low_freq: Low cutoff frequency (MHz)
            high_freq: High cutoff frequency (MHz)
            order: Filter order (higher = steeper rolloff)
        
        Returns:
            Filtered data
        """
        # Normalize frequencies to Nyquist frequency
        nyquist = self.sampling_freq_mhz / 2.0
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Ensure frequencies are in valid range
        low = np.clip(low, 0.01, 0.99)
        high = np.clip(high, 0.01, 0.99)
        
        if low >= high:
            print("Warning: Low freq >= High freq. Skipping filter.")
            return self.processed_data
        
        # Design Butterworth bandpass filter
        try:
            sos = signal.butter(order, [low, high], btype='band', output='sos')
            
            # Apply filter to each trace
            data = self.processed_data.copy()
            for i in range(data.shape[1]):
                data[:, i] = signal.sosfilt(sos, data[:, i])
            
            self.processed_data = data
            return data
        except Exception as e:
            print(f"Error applying bandpass filter: {e}")
            return self.processed_data
    
    def apply_gain(self, 
                   gain_type: Literal['exp', 'linear', 'agc'] = 'exp',
                   factor: float = 2.0,
                   window_ns: float = 50.0) -> np.ndarray:
        """
        Apply time-varying gain.
        
        Args:
            gain_type: Type of gain function
                - 'exp': Exponential gain
                - 'linear': Linear gain
                - 'agc': Automatic Gain Control
            factor: Gain factor (for exp and linear)
            window_ns: Window size for AGC (nanoseconds)
        
        Returns:
            Gain-corrected data
        """
        data = self.processed_data.copy()
        n_samples = data.shape[0]
        
        # Time vector
        time = np.arange(n_samples) * self.sampling_time_ns
        time_normalized = time / time[-1]  # Normalize to [0, 1]
        
        if gain_type == 'exp':
            # Exponential gain: amplification increases exponentially with depth
            gain_curve = np.exp(factor * time_normalized)
            
        elif gain_type == 'linear':
            # Linear gain: linearly increasing amplification
            gain_curve = 1.0 + factor * time_normalized
            
        elif gain_type == 'agc':
            # Automatic Gain Control: normalize within sliding window
            window_samples = int(window_ns / self.sampling_time_ns)
            gain_curve = np.ones(n_samples)
            
            for i in range(data.shape[1]):
                trace = data[:, i]
                # Calculate RMS in sliding window
                for j in range(n_samples):
                    start = max(0, j - window_samples // 2)
                    end = min(n_samples, j + window_samples // 2)
                    rms = np.sqrt(np.mean(trace[start:end]**2))
                    if rms > 1e-10:  # Avoid division by zero
                        data[j, i] = trace[j] / rms
            
            self.processed_data = data
            return data
        
        # Apply gain curve
        gain_curve = gain_curve.reshape(-1, 1)
        data = data * gain_curve
        
        self.processed_data = data
        return data
    
    def apply_hilbert(self) -> np.ndarray:
        """
        Apply Hilbert transform to get instantaneous amplitude (envelope).
        
        Returns:
            Envelope of the signal
        """
        data = self.processed_data.copy()
        
        # Apply Hilbert transform to each trace
        for i in range(data.shape[1]):
            analytic_signal = signal.hilbert(data[:, i])
            data[:, i] = np.abs(analytic_signal)
        
        self.processed_data = data
        return data
    
    def correct_time_zero(self, method: Literal['max', 'threshold'] = 'threshold',
                         threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and correct time-zero (first break).
        
        Args:
            method: Detection method
                - 'max': Use maximum amplitude
                - 'threshold': Use threshold crossing
            threshold: Threshold value (fraction of max amplitude)
        
        Returns:
            Tuple of (corrected_data, time_zero_indices)
        """
        data = self.processed_data.copy()
        n_traces = data.shape[1]
        time_zero = np.zeros(n_traces, dtype=int)
        
        for i in range(n_traces):
            trace = np.abs(data[:, i])
            
            if method == 'max':
                # Find first maximum
                time_zero[i] = np.argmax(trace)
            elif method == 'threshold':
                # Find first threshold crossing
                thresh_val = threshold * np.max(trace)
                crossings = np.where(trace > thresh_val)[0]
                if len(crossings) > 0:
                    time_zero[i] = crossings[0]
        
        # Align traces to median time-zero
        median_tz = int(np.median(time_zero))
        corrected = np.zeros_like(data)
        
        for i in range(n_traces):
            shift = median_tz - time_zero[i]
            if shift > 0:
                corrected[shift:, i] = data[:-shift, i]
            elif shift < 0:
                corrected[:shift, i] = data[-shift:, i]
            else:
                corrected[:, i] = data[:, i]
        
        self.processed_data = corrected
        return corrected, time_zero
    
    def dewow(self, window_size: int = 50) -> np.ndarray:
        """
        Remove 'wow' (low-frequency drift) using moving average.
        
        Args:
            window_size: Size of moving average window (samples)
        
        Returns:
            Dewowed data
        """
        data = self.processed_data.copy()
        
        for i in range(data.shape[1]):
            # Compute moving average
            ma = np.convolve(data[:, i], 
                           np.ones(window_size)/window_size, 
                           mode='same')
            data[:, i] = data[:, i] - ma
        
        self.processed_data = data
        return data
    
    def normalize(self, method: Literal['minmax', 'zscore', 'robust'] = 'minmax') -> np.ndarray:
        """
        Normalize data.
        
        Args:
            method: Normalization method
                - 'minmax': Scale to [0, 1]
                - 'zscore': Zero mean, unit variance
                - 'robust': Use percentiles (robust to outliers)
        
        Returns:
            Normalized data
        """
        data = self.processed_data.copy()
        
        if method == 'minmax':
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
        
        elif method == 'zscore':
            mean = np.mean(data)
            std = np.std(data)
            if std > 0:
                data = (data - mean) / std
        
        elif method == 'robust':
            p5 = np.percentile(data, 5)
            p95 = np.percentile(data, 95)
            if p95 > p5:
                data = (data - p5) / (p95 - p5)
                data = np.clip(data, 0, 1)
        
        self.processed_data = data
        return data
    
    def reset(self):
        """Reset to original data."""
        self.processed_data = self.data_2d.copy()
    
    def get_processed_data(self) -> np.ndarray:
        """Get current processed data."""
        return self.processed_data.copy()
    
    def get_time_axis(self) -> np.ndarray:
        """Get time axis in nanoseconds."""
        n_samples = self.processed_data.shape[0]
        return np.arange(n_samples) * self.sampling_time_ns
    
    def get_depth_axis(self, velocity: float = 0.1) -> np.ndarray:
        """
        Get depth axis in meters.
        
        Args:
            velocity: EM wave velocity (m/ns), default 0.1 m/ns (typical for soil)
        
        Returns:
            Depth array in meters
        """
        time = self.get_time_axis()
        return time * velocity / 2.0  # Two-way travel time