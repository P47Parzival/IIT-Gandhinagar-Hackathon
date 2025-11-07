"""
ADWIN (Adaptive Windowing) Drift Detection
Based on AnDri (McMaster University) - https://www.cas.mcmaster.ca/~fchiang/pubs/andri.pdf

Detects distribution shifts in time series data using statistical change detection.
Critical for defending against data poisoning attacks.
"""

import numpy as np
from typing import List, Optional
from collections import deque


class ADWIN:
    """
    Adaptive Windowing for drift detection.
    
    Algorithm:
    1. Maintain sliding window of recent values
    2. For each possible cut point, test if two sub-windows have different means
    3. Use Hoeffding bound for statistical significance test
    4. If drift detected, drop old sub-window and reset
    
    Parameters:
        delta (float): Confidence level (default: 0.002 = 99.8% confidence)
        max_window_size (int): Maximum window size to prevent memory explosion
    """
    
    def __init__(self, delta: float = 0.002, max_window_size: int = 1000):
        # Validate parameters
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if max_window_size < 2:
            raise ValueError(f"max_window_size must be >= 2, got {max_window_size}")
        
        self.delta = delta
        self.max_window_size = max_window_size
        self.window: deque = deque(maxlen=max_window_size)
        self.drift_detected = False
        self.total_elements = 0
        
    def add_element(self, value: float) -> bool:
        """
        Add new value to window and check for drift.
        
        Args:
            value: New data point (e.g., reconstruction error)
        
        Returns:
            True if drift detected, False otherwise
        
        Algorithm (from AnDri paper):
        1. Append value to window
        2. For each possible cut point i in window:
           - Split: W0 = window[:i], W1 = window[i:]
           - Test: |mean(W0) - mean(W1)| > epsilon_cut
           - epsilon_cut = sqrt((1/(2*m)) * ln(4*n/delta))
             where m = harmonic_mean(|W0|, |W1|), n = |window|
        3. If test passes: drift detected, drop W0, keep W1
        """
        # Validate input
        if np.isnan(value) or np.isinf(value):
            raise ValueError(f"Invalid value: {value} (NaN or Inf not allowed)")
        
        self.window.append(value)
        self.total_elements += 1
        self.drift_detected = False
        
        # Need at least 2 elements to detect drift
        if len(self.window) < 2:
            return False
        
        # Check all possible cut points
        n = len(self.window)
        window_array = np.array(self.window)
        
        for i in range(1, n):  # Cut point at index i (W0 = [:i], W1 = [i:])
            W0 = window_array[:i]
            W1 = window_array[i:]
            
            n0 = len(W0)
            n1 = len(W1)
            
            # Compute means
            mean0 = np.mean(W0)
            mean1 = np.mean(W1)
            
            # Compute epsilon_cut (Hoeffding bound)
            # m = harmonic mean of n0, n1
            m = 2 * n0 * n1 / (n0 + n1)  # Harmonic mean formula
            
            # epsilon_cut = sqrt((1/(2*m)) * ln(4*n/delta))
            # BUG FIX #60: Validate log argument is positive before sqrt
            log_arg = 4 * n / self.delta
            if log_arg <= 1.0:
                # log_arg <= 1 means log(log_arg) <= 0, sqrt of negative = NaN
                # This should never happen with proper delta (0 < delta < 1), but safety check
                continue  # Skip this cut point
            
            epsilon_cut = np.sqrt((1 / (2 * m)) * np.log(log_arg))
            
            # Test for drift
            if abs(mean0 - mean1) > epsilon_cut:
                # Drift detected! Drop W0, keep W1
                self.drift_detected = True
                
                # Keep only W1 (recent data)
                self.window = deque(window_array[i:], maxlen=self.max_window_size)
                
                return True
        
        return False
    
    def reset(self):
        """Clear window after drift handled manually."""
        self.window.clear()
        self.drift_detected = False
    
    def get_window_stats(self) -> dict:
        """
        Get statistics about current window.
        
        Returns:
            Dictionary with window size, mean, std, min, max
        """
        if len(self.window) == 0:
            return {
                'size': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        window_array = np.array(self.window)
        return {
            'size': len(self.window),
            'mean': float(np.mean(window_array)),
            'std': float(np.std(window_array)),
            'min': float(np.min(window_array)),
            'max': float(np.max(window_array))
        }
    
    def get_state(self) -> dict:
        """
        Export state for saving.
        
        Returns:
            Dictionary with window and parameters
        """
        return {
            'delta': self.delta,
            'max_window_size': self.max_window_size,
            'window': list(self.window),
            'total_elements': self.total_elements
        }
    
    @classmethod
    def from_state(cls, state: dict) -> 'ADWIN':
        """
        Restore from saved state.
        
        Args:
            state: Dictionary from get_state()
        
        Returns:
            Restored ADWIN instance
        """
        adwin = cls(delta=state['delta'], max_window_size=state['max_window_size'])
        adwin.window = deque(state['window'], maxlen=state['max_window_size'])
        adwin.total_elements = state['total_elements']
        return adwin
    
    def __repr__(self) -> str:
        stats = self.get_window_stats()
        return (
            f"ADWIN(window_size={stats['size']}, mean={stats['mean']:.4f}, "
            f"drift_detected={self.drift_detected})"
        )

