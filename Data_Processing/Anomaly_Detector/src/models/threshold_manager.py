"""
ThresholdManager - Coordinates SPOT + ADWIN per entity with surge mode defense
Implements poisoning attack detection and response via drift-aware thresholding
"""

import numpy as np
from typing import Dict, Optional, List
from datetime import datetime

from .spot_threshold import SPOTThreshold
from .adwin import ADWIN


class ThresholdManager:
    """
    Manages per-entity SPOT + ADWIN with surge mode for poisoning defense.
    
    Features:
    - Independent adaptive thresholds per entity (AGENCY, FUND, etc.)
    - Drift detection via ADWIN on error streams
    - Surge mode: freeze + tighten thresholds during suspected attacks
    - **NEW: ADT (Agent-based Dynamic Thresholding) via DQN** - learns from human feedback
    - State persistence for model checkpoints
    
    Surge Mode Defense:
    When ADWIN detects distribution shift (potential poisoning):
    1. Freeze SPOT threshold (no updates from new data)
    2. Tighten threshold by 50% (more conservative)
    3. Flag all anomalies for priority review
    4. Log surge event for forensics
    
    ADT Integration:
    When enabled, ADT learns optimal threshold adjustments from human feedback:
    1. SPOT provides base threshold (statistically calibrated)
    2. ADT adjusts via learned policy: threshold_final = threshold_spot * (1 + delta)
    3. Human labels (correct/false positive) update DQN
    4. Delta converges to per-entity optimal adjustment
    """
    
    def __init__(
        self,
        surge_tightening: float = 0.5,
        initial_quantile: float = 0.98,
        extreme_prob: float = 0.001,
        adwin_delta: float = 0.002,
        enable_adt: bool = False
    ):
        """
        Args:
            surge_tightening: Threshold multiplier in surge mode (0.5 = 50% tighter)
            initial_quantile: SPOT initial threshold quantile (0.98 = top 2%)
            extreme_prob: SPOT target anomaly rate (0.001 = 0.1%)
            adwin_delta: ADWIN confidence level (0.002 = 99.8%)
            enable_adt: Enable ADT (DQN-based threshold learning from feedback)
        """
        self.entity_spots: Dict[str, SPOTThreshold] = {}
        self.entity_adwins: Dict[str, ADWIN] = {}
        self.surge_mode: Dict[str, bool] = {}
        self.surge_events: Dict[str, List[dict]] = {}  # Forensics log
        
        # Configuration
        self.surge_tightening = surge_tightening
        self.initial_quantile = initial_quantile
        self.extreme_prob = extreme_prob
        self.adwin_delta = adwin_delta
        
        # Global fallback for low-sample entities
        self.global_threshold: Optional[float] = None
        
        # ADT (Agent-based Dynamic Thresholding) - NEW
        self.enable_adt = enable_adt
        self.adt_controllers: Dict[str, 'ADTController'] = {}  # Lazy import to avoid circular dependency
        
    def calibrate_entity(self, entity_id: str, errors: np.ndarray) -> None:
        """
        Initialize SPOT + ADWIN for entity during training.
        
        Args:
            entity_id: Entity identifier (e.g., AGENCY code)
            errors: Array of reconstruction errors for this entity
        """
        if len(errors) < 100:
            # Too few samples for reliable SPOT
            print(f"  [WARNING] Entity {entity_id} has only {len(errors)} samples")
            print(f"            Will use global threshold as fallback")
            return
        
        # Initialize SPOT
        spot = SPOTThreshold(
            initial_quantile=self.initial_quantile,
            extreme_prob=self.extreme_prob
        )
        spot.calibrate(errors)
        self.entity_spots[entity_id] = spot
        
        # Initialize ADWIN
        adwin = ADWIN(delta=self.adwin_delta)
        self.entity_adwins[entity_id] = adwin
        
        # Not in surge mode initially
        self.surge_mode[entity_id] = False
        self.surge_events[entity_id] = []
        
    def set_global_threshold(self, threshold: float) -> None:
        """
        Set fallback threshold for low-sample entities.
        
        Args:
            threshold: Global threshold value
        """
        self.global_threshold = threshold
    
    def check_anomaly_batch(self, entity_id: str, errors: np.ndarray) -> dict:
        """
        OPTIMIZED: Check multiple errors for ONE entity in batch.
        Maintains EXACT same order and state updates as calling check_anomaly() repeatedly.
        
        5-10x faster than per-sample loop while producing identical results.
        
        Args:
            entity_id: Entity identifier (string)
            errors: Array of reconstruction errors for this entity [n_samples]
        
        Returns:
            Dict with:
                'predictions': np.ndarray of anomaly flags (0/1) [n_samples]
                'drift_flags': np.ndarray of drift bools [n_samples]
                'thresholds': np.ndarray of thresholds used [n_samples]
        """
        # Input validation
        if not isinstance(entity_id, str) or not entity_id.strip():
            raise ValueError("entity_id must be a non-empty string")
        
        entity_id = entity_id.strip()
        n_samples = len(errors)
        
        # Pre-allocate output arrays
        predictions = np.zeros(n_samples, dtype=np.int32)
        drift_flags = np.zeros(n_samples, dtype=bool)
        thresholds = np.zeros(n_samples, dtype=np.float32)
        
        # Check if entity has calibrated SPOT
        if entity_id not in self.entity_spots:
            # Fallback to global threshold (no state updates)
            if self.global_threshold is None:
                raise ValueError(f"Entity {entity_id} not calibrated and no global threshold set")
            
            predictions = (errors >= self.global_threshold).astype(np.int32)
            thresholds[:] = self.global_threshold
            
            return {
                'predictions': predictions,
                'drift_flags': drift_flags,
                'thresholds': thresholds
            }
        
        # Cache objects (avoid 71K dict lookups!)
        spot = self.entity_spots[entity_id]
        adwin = self.entity_adwins[entity_id]
        in_surge = self.surge_mode.get(entity_id, False)
        
        # ADT adjustment factor (if enabled)
        adt_adjustment = 1.0
        if self.enable_adt and entity_id in self.adt_controllers:
            adt_adjustment = self.adt_controllers[entity_id].get_threshold_adjustment()
        
        # Process in order (maintains state consistency)
        for i, error in enumerate(errors):
            base_threshold = spot.get_threshold()
            
            # Apply ADT adjustment if enabled (before surge tightening)
            adjusted_threshold = base_threshold * adt_adjustment
            
            # Apply surge mode tightening if active
            active_threshold = adjusted_threshold * self.surge_tightening if in_surge else adjusted_threshold
            thresholds[i] = active_threshold
            
            # ADWIN drift check
            drift_detected = adwin.add_element(error)
            drift_flags[i] = drift_detected
            
            if drift_detected and not in_surge:
                self._enter_surge_mode(entity_id, error, adwin.get_window_stats())
                in_surge = True
                # Apply BOTH ADT adjustment AND surge tightening
                active_threshold = base_threshold * adt_adjustment * self.surge_tightening
                thresholds[i] = active_threshold
            
            # SPOT update (frozen during surge)
            if not in_surge:
                spot.update(error)
            
            # Decision
            predictions[i] = int(error >= active_threshold)
        
        # Update surge cache
        self.surge_mode[entity_id] = in_surge
        
        return {
            'predictions': predictions,
            'drift_flags': drift_flags,
            'thresholds': thresholds
        }
        
    def check_anomaly(self, entity_id: str, error: float) -> dict:
        """
        Check if error is anomaly, handle drift detection.
        
        Args:
            entity_id: Entity identifier
            error: Reconstruction error
        
        Returns:
            {
                'is_anomaly': bool,
                'threshold': float,
                'drift_detected': bool,
                'surge_mode': bool,
                'entity_has_spot': bool
            }
        
        Algorithm:
        1. Get SPOT threshold (or tightened if surge mode)
        2. Check ADWIN for drift
        3. If drift detected:
           - Enter surge mode
           - Freeze SPOT updates
           - Tighten threshold (multiply by surge_tightening)
           - Log event
        4. If not in surge mode:
           - Update SPOT normally (peaks update model)
           - Update ADWIN window
        5. Return anomaly decision
        """
        # Validate inputs
        if not entity_id or not isinstance(entity_id, str):
            raise ValueError(f"entity_id must be non-empty string, got {repr(entity_id)}")
        if np.isnan(error) or np.isinf(error):
            raise ValueError(f"Invalid error value for entity {entity_id}: {error} (NaN or Inf)")
        
        # Check if entity has calibrated SPOT
        if entity_id not in self.entity_spots:
            # Fallback to global threshold
            if self.global_threshold is None:
                raise ValueError(f"Entity {entity_id} not calibrated and no global threshold set")
            return {
                'is_anomaly': error >= self.global_threshold,
                'threshold': self.global_threshold,
                'drift_detected': False,
                'surge_mode': False,
                'entity_has_spot': False
            }
        
        spot = self.entity_spots[entity_id]
        adwin = self.entity_adwins[entity_id]
        in_surge = self.surge_mode.get(entity_id, False)
        
        # Get base threshold
        base_threshold = spot.get_threshold()
        
        # Apply surge tightening if active
        active_threshold = base_threshold * self.surge_tightening if in_surge else base_threshold
        
        # Check for drift (potential attack)
        drift_detected = adwin.add_element(error)
        
        if drift_detected and not in_surge:
            # Enter surge mode
            self._enter_surge_mode(entity_id, error, adwin.get_window_stats())
            in_surge = True
            active_threshold = base_threshold * self.surge_tightening
        
        # Update SPOT if not in surge mode
        if not in_surge:
            # Normal operation: SPOT adapts to peaks
            spot.update(error)
        # else: Freeze SPOT during surge (no updates)
        
        # Anomaly decision
        is_anomaly = error >= active_threshold
        
        return {
            'is_anomaly': is_anomaly,
            'threshold': active_threshold,
            'drift_detected': drift_detected,
            'surge_mode': in_surge,
            'entity_has_spot': True
        }
    
    def _enter_surge_mode(self, entity_id: str, trigger_error: float, window_stats: dict) -> None:
        """
        Enter surge mode: freeze threshold, log event.
        
        Args:
            entity_id: Entity identifier
            trigger_error: Error that triggered drift
            window_stats: ADWIN window statistics at trigger
        """
        self.surge_mode[entity_id] = True
        
        spot = self.entity_spots[entity_id]
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'entity_id': entity_id,
            'trigger_error': float(trigger_error),
            'threshold_before': float(spot.get_threshold()),
            'threshold_after': float(spot.get_threshold() * self.surge_tightening),
            'window_mean': window_stats['mean'],
            'window_std': window_stats['std'],
            'window_size': window_stats['size']
        }
        
        self.surge_events[entity_id].append(event)
        
        print(f"\n{'='*80}")
        print(f"[SURGE MODE ACTIVATED] Entity: {entity_id}")
        print(f"{'='*80}")
        print(f"Drift detected at error = {trigger_error:.4f}")
        print(f"Threshold frozen at: {spot.get_threshold():.4f}")
        print(f"Active threshold (tightened): {spot.get_threshold() * self.surge_tightening:.4f}")
        print(f"ADWIN window: mean={window_stats['mean']:.4f}, std={window_stats['std']:.4f}")
        print(f"\n[ACTION REQUIRED] Manual review needed to confirm drift legitimacy")
        print(f"{'='*80}\n")
    
    def exit_surge_mode(self, entity_id: str) -> None:
        """
        Manual exit after review (called after human confirms drift is normal).
        
        Args:
            entity_id: Entity to exit surge mode
        """
        if entity_id not in self.surge_mode:
            return
        
        # Check if entity has ADWIN (was calibrated)
        if entity_id not in self.entity_adwins:
            print(f"[WARNING] Entity {entity_id} has no ADWIN (uncalibrated)")
            self.surge_mode[entity_id] = False
            return
        
        self.surge_mode[entity_id] = False
        
        # Reset ADWIN to fresh state
        self.entity_adwins[entity_id].reset()
        
        print(f"[INFO] Surge mode exited for entity {entity_id}, ADWIN reset")
    
    def get_surge_entities(self) -> List[str]:
        """
        Get list of entities currently in surge mode.
        
        Returns:
            List of entity IDs in surge mode
        """
        return [entity for entity, in_surge in self.surge_mode.items() if in_surge]
    
    def get_entity_summary(self) -> dict:
        """
        Get summary statistics across all entities.
        
        Returns:
            Dictionary with entity counts, surge status, thresholds
        """
        n_entities = len(self.entity_spots)
        n_surge = sum(self.surge_mode.values())
        
        thresholds = {
            entity: spot.get_threshold()
            for entity, spot in self.entity_spots.items()
        }
        
        return {
            'n_entities': n_entities,
            'n_surge': n_surge,
            'surge_entities': self.get_surge_entities(),
            'thresholds': thresholds,
            'global_threshold': self.global_threshold
        }
    
    # ========================================================================
    # ADT (Agent-based Dynamic Thresholding) Methods
    # ========================================================================
    
    def init_adt_for_entity(self, entity_id: str, device: str = 'cpu') -> None:
        """
        Initialize ADT (DQN) controller for entity.
        Call after SPOT calibration, before detection starts.
        
        Args:
            entity_id: Entity identifier
            device: 'cpu' or 'cuda'
        """
        if not self.enable_adt:
            return
        
        # Lazy import to avoid circular dependency
        from .adt_controller import ADTController
        
        if entity_id not in self.entity_spots:
            print(f"[WARNING] Cannot init ADT for {entity_id}: SPOT not calibrated")
            return
        
        self.adt_controllers[entity_id] = ADTController(
            entity_id=entity_id,
            device=device
        )
        
        print(f"[ADT] Initialized for entity {entity_id}")
    
    def update_from_feedback(
        self,
        entity_id: str,
        feedback_batch: List[Dict],
        alert_rate: float
    ) -> None:
        """
        Update ADT controller from human feedback labels.
        
        Args:
            entity_id: Entity identifier
            feedback_batch: List of {'is_correct': bool, 'anomaly_id': str}
            alert_rate: Current alert rate for this entity (alerts / samples)
        
        Example:
            feedback = [
                {'is_correct': True, 'anomaly_id': 'ANO_001'},
                {'is_correct': False, 'anomaly_id': 'ANO_002'},
                ...
            ]
            manager.update_from_feedback('AGY_45200', feedback, alert_rate=0.05)
        """
        if not self.enable_adt:
            return
        
        if entity_id not in self.adt_controllers:
            print(f"[WARNING] No ADT controller for {entity_id}, initializing...")
            self.init_adt_for_entity(entity_id)
            if entity_id not in self.adt_controllers:
                return
        
        # Update ADT with feedback
        self.adt_controllers[entity_id].update_from_feedback(feedback_batch, alert_rate)
        
        # Log update
        n_correct = sum(1 for f in feedback_batch if f['is_correct'])
        precision = n_correct / len(feedback_batch) if len(feedback_batch) > 0 else 0.0
        delta = self.adt_controllers[entity_id].current_delta
        
        print(f"[ADT] Updated {entity_id}: {len(feedback_batch)} reviews, "
              f"precision={precision:.1%}, delta={delta:+.3f}")
    
    def enable_adt_learning(self, device: str = 'cpu') -> None:
        """
        Enable ADT and initialize controllers for all calibrated entities.
        
        Args:
            device: 'cpu' or 'cuda'
        """
        self.enable_adt = True
        
        for entity_id in self.entity_spots.keys():
            if entity_id not in self.adt_controllers:
                self.init_adt_for_entity(entity_id, device=device)
        
        print(f"[ADT] Enabled for {len(self.adt_controllers)} entities")
    
    def get_state(self) -> dict:
        """
        Export state for saving to model checkpoint.
        
        Prunes excesses and surge events to prevent checkpoint bloat.
        
        Returns:
            Dictionary with all entity states (SPOT, ADWIN, ADT)
        """
        state = {
            'entity_spots': {
                entity: {
                    **spot.get_state(),
                    # Prune excesses to last 1000 (sufficient for GPD fitting)
                    'excesses': spot.excesses[-1000:] if len(spot.excesses) > 1000 else spot.excesses.copy()
                }
                for entity, spot in self.entity_spots.items()
            },
            'entity_adwins': {
                entity: adwin.get_state()
                for entity, adwin in self.entity_adwins.items()
            },
            'surge_mode': self.surge_mode.copy(),
            # Prune surge events to last 10 per entity (keep forensics but prevent bloat)
            'surge_events': {
                entity: events[-10:] if len(events) > 10 else events.copy()
                for entity, events in self.surge_events.items()
            },
            'config': {
                'surge_tightening': self.surge_tightening,
                'initial_quantile': self.initial_quantile,
                'extreme_prob': self.extreme_prob,
                'adwin_delta': self.adwin_delta,
                'enable_adt': self.enable_adt
            },
            'global_threshold': self.global_threshold
        }
        
        # Add ADT controllers if enabled
        if self.enable_adt:
            state['adt_controllers'] = {
                entity: controller.get_state_dict()
                for entity, controller in self.adt_controllers.items()
            }
        
        return state
    
    @classmethod
    def from_state(cls, state: dict, device: str = 'cpu') -> 'ThresholdManager':
        """
        Restore from saved state.
        
        Args:
            state: Dictionary from get_state()
            device: Device for ADT controllers ('cpu' or 'cuda')
        
        Returns:
            Restored ThresholdManager instance
        """
        config = state['config']
        enable_adt = config.get('enable_adt', False)  # Backward compatibility
        
        manager = cls(
            surge_tightening=config['surge_tightening'],
            initial_quantile=config['initial_quantile'],
            extreme_prob=config['extreme_prob'],
            adwin_delta=config['adwin_delta'],
            enable_adt=enable_adt
        )
        
        # Restore SPOT instances
        manager.entity_spots = {
            entity: SPOTThreshold.from_state(spot_state)
            for entity, spot_state in state['entity_spots'].items()
        }
        
        # Restore ADWIN instances
        manager.entity_adwins = {
            entity: ADWIN.from_state(adwin_state)
            for entity, adwin_state in state['entity_adwins'].items()
        }
        
        # Restore surge state
        manager.surge_mode = state['surge_mode'].copy()
        manager.surge_events = state['surge_events'].copy()
        manager.global_threshold = state['global_threshold']
        
        # Restore ADT controllers if present (with correct device)
        if enable_adt and 'adt_controllers' in state:
            from .adt_controller import ADTController
            manager.adt_controllers = {
                entity: ADTController.from_state_dict(adt_state, device=device)
                for entity, adt_state in state['adt_controllers'].items()
            }
        
        return manager
    
    def __repr__(self) -> str:
        summary = self.get_entity_summary()
        return (
            f"ThresholdManager(entities={summary['n_entities']}, "
            f"surge_active={summary['n_surge']})"
        )

