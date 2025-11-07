"""
SPOT (Streaming Peaks-Over-Threshold) Adaptive Thresholding
Based on FluxEV (WSDM'21) - https://sdiaa.github.io/papers/WSDM21.pdf

Uses Extreme Value Theory (EVT) and Generalized Pareto Distribution (GPD)
to set statistically optimal anomaly thresholds without labels.
"""

import numpy as np
from typing import Optional, List


class SPOTThreshold:
    """
    Streaming Peaks-Over-Threshold using Method of Moments GPD fitting.
    
    Algorithm:
    1. Calibration: Fit GPD to tail of reconstruction errors
    2. Online: Update threshold as new peaks arrive
    3. Three zones:
       - Normal: error < t (initial threshold)
       - Peak: t <= error < q_alpha (updates model)
       - Anomaly: error >= q_alpha (flagged)
    
    Parameters:
        initial_quantile (float): Initial threshold quantile (default: 0.98 = top 2%)
        extreme_prob (float): Target extreme probability (default: 0.001 = 0.1% anomaly rate)
        min_excesses (int): Minimum peaks needed for reliable GPD fit (default: 50)
    """
    
    def __init__(
        self,
        initial_quantile: float = 0.98,
        extreme_prob: float = 0.001,
        min_excesses: int = 50
    ):
        # Validate parameters
        if not (0 < initial_quantile < 1):
            raise ValueError(f"initial_quantile must be in (0, 1), got {initial_quantile}")
        if not (0 < extreme_prob < 1):
            raise ValueError(f"extreme_prob must be in (0, 1), got {extreme_prob}")
        if min_excesses < 2:
            raise ValueError(f"min_excesses must be >= 2, got {min_excesses}")
        
        self.initial_quantile = initial_quantile
        self.extreme_prob = extreme_prob  # alpha in paper
        self.min_excesses = min_excesses
        
        # GPD parameters
        self.t: Optional[float] = None      # Initial threshold
        self.xi: Optional[float] = None     # Shape parameter
        self.beta: Optional[float] = None   # Scale parameter
        self.q_alpha: Optional[float] = None  # Adaptive threshold
        
        # Excesses (peaks above t)
        self.excesses: List[float] = []
        self.n_init: int = 0  # Number of initial calibration samples
        
        self.calibrated = False
        
    def calibrate(self, errors: np.ndarray) -> None:
        """
        Calibration phase: Fit initial GPD to error tail.
        
        Args:
            errors: Array of reconstruction errors from training data
        
        Algorithm:
        1. Set t = percentile(errors, initial_quantile)
        2. Extract excesses Y = errors[errors > t] - t
        3. Fit GPD using Method of Moments:
           mean_Y = mean(Y)
           var_Y = var(Y)
           xi = 0.5 * (1 - mean_Y^2 / var_Y)     # shape
           beta = 0.5 * mean_Y * (mean_Y^2/var_Y + 1)  # scale
        4. Compute q_alpha = t + (beta/xi) * ((n*alpha/Nt)^(-xi) - 1)
        """
        if len(errors) < self.min_excesses:
            raise ValueError(
                f"Need at least {self.min_excesses} errors for calibration, got {len(errors)}"
            )
        
        # Step 1: Set initial threshold
        self.t = float(np.percentile(errors, self.initial_quantile * 100))
        self.n_init = len(errors)
        
        # Step 2: Extract excesses (peaks above threshold)
        peaks = errors[errors > self.t]
        self.excesses = (peaks - self.t).tolist()
        
        if len(self.excesses) < self.min_excesses:
            # Adjust threshold to get enough excesses
            new_quantile = max(0.9, 1 - self.min_excesses / len(errors))
            self.t = float(np.percentile(errors, new_quantile * 100))
            peaks = errors[errors > self.t]
            self.excesses = (peaks - self.t).tolist()
            
            # Validate we have enough excesses after retry
            if len(self.excesses) < 2:
                raise ValueError(
                    f"Insufficient variance in errors for SPOT calibration. "
                    f"Got only {len(self.excesses)} excesses after lowering threshold to {new_quantile*100:.1f}%. "
                    f"Data may be too uniform (mean={errors.mean():.6f}, std={errors.std():.6f}). "
                    f"Need at least {self.min_excesses} excesses for reliable GPD fit."
                )
        
        # Step 3: Fit GPD using Method of Moments (faster than MLE)
        self._fit_gpd()
        
        # Step 4: Compute adaptive threshold
        self._update_threshold()
        
        self.calibrated = True
        
    def _fit_gpd(self) -> None:
        """
        Fit Generalized Pareto Distribution using Method of Moments.
        
        Formulas from FluxEV paper:
        xi = 0.5 * (1 - mean^2(Y) / var(Y))
        beta = 0.5 * mean(Y) * (mean^2(Y)/var(Y) + 1)
        """
        if len(self.excesses) < 2:
            # Fallback: assume light tail (xi=0 -> exponential)
            self.xi = 0.0
            self.beta = np.mean(self.excesses) if self.excesses else 1e-6
            return
        
        Y = np.array(self.excesses)
        mean_Y = np.mean(Y)
        var_Y = np.var(Y)
        
        # Numerical stability checks
        if var_Y < 1e-10:
            # Constant excesses -> degenerate case
            self.xi = 0.0
            self.beta = mean_Y if mean_Y > 0 else 1e-6
            return
        
        # Method of Moments formulas
        ratio = mean_Y**2 / var_Y
        
        # Shape parameter (xi)
        # xi must be < 1 for finite variance (cap at 0.99)
        self.xi = 0.5 * (1 - ratio)
        self.xi = float(np.clip(self.xi, -0.5, 0.99))  # Stability bounds
        
        # Scale parameter (beta)
        # beta must be > 0
        self.beta = 0.5 * mean_Y * (ratio + 1)
        self.beta = float(max(self.beta, 1e-6))  # Ensure positive
        
    def _update_threshold(self) -> None:
        """
        Compute adaptive threshold q_alpha from GPD parameters.
        
        Formula from SPOT paper:
        q_alpha = t + (beta/xi) * ((n*alpha/N_t)^(-xi) - 1)
        
        where:
        - t: initial threshold
        - n: total samples seen
        - alpha: target extreme probability
        - N_t: number of excesses (peaks)
        - xi, beta: GPD parameters
        """
        N_t = len(self.excesses)
        n = self.n_init  # Use calibration sample size
        alpha = self.extreme_prob
        
        if N_t == 0:
            # No excesses yet, use initial threshold
            self.q_alpha = self.t
            return
        
        # Handle xi ≈ 0 (exponential distribution limit)
        if abs(self.xi) < 1e-6:
            # q_alpha = t + beta * ln(n*alpha / N_t)
            # BUG FIX #61: Validate log argument is positive
            log_arg = n * alpha / N_t
            if log_arg <= 0:
                # Shouldn't happen, but safety check
                self.q_alpha = self.t
                return
            self.q_alpha = self.t + self.beta * np.log(log_arg)
        else:
            # General GPD formula
            # BUG FIX #61: Check for numerical overflow in power operation
            base = n * alpha / N_t
            exponent = -self.xi
            
            # Prevent overflow: if base^exponent would be > 1e10 or < 1e-10, cap it
            if exponent * np.log(base) > 23:  # log(1e10) ≈ 23
                prob_term = 1e10
            elif exponent * np.log(base) < -23:  # log(1e-10) ≈ -23
                prob_term = 1e-10
            else:
                prob_term = base ** exponent
            
            self.q_alpha = self.t + (self.beta / self.xi) * (prob_term - 1)
        
        # Safety: threshold should be >= initial threshold, and finite
        if not np.isfinite(self.q_alpha):
            self.q_alpha = self.t
        else:
            self.q_alpha = float(max(self.q_alpha, self.t))
        
    def update(self, new_error: float) -> bool:
        """
        Online update: Check if error is Peak (updates model) or Anomaly.
        
        Args:
            new_error: New reconstruction error
        
        Returns:
            True if anomaly (error >= q_alpha), False otherwise
        
        Logic:
        - If error < t: Normal (no action)
        - If t <= error < q_alpha: Peak (add to excesses, refit GPD)
        - If error >= q_alpha: Anomaly (flag but don't update model)
        """
        if not self.calibrated:
            raise RuntimeError("Must call calibrate() before update()")
        
        # Validate input
        if np.isnan(new_error) or np.isinf(new_error):
            raise ValueError(f"Invalid error value: {new_error} (NaN or Inf not allowed)")
        
        # Check anomaly
        is_anomaly = (new_error >= self.q_alpha)
        
        # Update model if in peak zone
        if self.t <= new_error < self.q_alpha:
            # Add to excesses
            self.excesses.append(new_error - self.t)
            
            # Prune old excesses to prevent memory explosion (keep last 10,000)
            # GPD fitting only needs representative sample, not entire history
            if len(self.excesses) > 10000:
                self.excesses = self.excesses[-10000:]
            
            # Refit GPD and update threshold
            self._fit_gpd()
            self._update_threshold()
        
        return is_anomaly
    
    def get_threshold(self) -> float:
        """Return current adaptive threshold q_alpha."""
        if not self.calibrated:
            raise RuntimeError("Must call calibrate() before get_threshold()")
        return self.q_alpha
    
    def get_state(self) -> dict:
        """
        Export state for saving to model checkpoint.
        
        Returns:
            Dictionary with all parameters and excesses
        """
        return {
            't': self.t,
            'xi': self.xi,
            'beta': self.beta,
            'q_alpha': self.q_alpha,
            'excesses': self.excesses.copy(),
            'n_init': self.n_init,
            'initial_quantile': self.initial_quantile,
            'extreme_prob': self.extreme_prob,
            'min_excesses': self.min_excesses,
            'calibrated': self.calibrated
        }
    
    @classmethod
    def from_state(cls, state: dict) -> 'SPOTThreshold':
        """
        Restore from saved state.
        
        Args:
            state: Dictionary from get_state()
        
        Returns:
            Restored SPOTThreshold instance
        """
        spot = cls(
            initial_quantile=state['initial_quantile'],
            extreme_prob=state['extreme_prob'],
            min_excesses=state['min_excesses']
        )
        spot.t = state['t']
        spot.xi = state['xi']
        spot.beta = state['beta']
        spot.q_alpha = state['q_alpha']
        spot.excesses = state['excesses'].copy()
        spot.n_init = state['n_init']
        spot.calibrated = state['calibrated']
        return spot
    
    def __repr__(self) -> str:
        if not self.calibrated:
            return "SPOTThreshold(not calibrated)"
        return (
            f"SPOTThreshold(t={self.t:.4f}, q_alpha={self.q_alpha:.4f}, "
            f"xi={self.xi:.4f}, beta={self.beta:.4f}, excesses={len(self.excesses)})"
        )

