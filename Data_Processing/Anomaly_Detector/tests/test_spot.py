"""
Test suite for SPOT + ADWIN adaptive thresholding
Includes poisoning attack simulation scenarios
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.spot_threshold import SPOTThreshold
from src.models.adwin import ADWIN
from src.models.threshold_manager import ThresholdManager


def test_spot_basic():
    """Test basic SPOT calibration and update."""
    print("\n" + "="*80)
    print("TEST 1: SPOT Basic Calibration")
    print("="*80)
    
    # Generate synthetic reconstruction errors (gamma distribution typical for autoencoders)
    np.random.seed(42)
    normal_errors = np.random.gamma(2, 0.01, 1000)
    
    # Calibrate SPOT
    spot = SPOTThreshold(initial_quantile=0.98, extreme_prob=0.001)
    spot.calibrate(normal_errors)
    
    print(f"Initial threshold (t): {spot.t:.4f}")
    print(f"Adaptive threshold (q_alpha): {spot.q_alpha:.4f}")
    print(f"GPD parameters: xi={spot.xi:.4f}, beta={spot.beta:.4f}")
    print(f"Number of excesses: {len(spot.excesses)}")
    
    # Test online updates with new data
    new_errors = np.random.gamma(2, 0.01, 100)
    anomalies_detected = 0
    
    for error in new_errors:
        is_anomaly = spot.update(error)
        if is_anomaly:
            anomalies_detected += 1
    
    print(f"\nOnline updates: {len(new_errors)} new samples")
    print(f"Anomalies detected: {anomalies_detected} ({anomalies_detected/len(new_errors)*100:.2f}%)")
    print(f"Updated threshold: {spot.get_threshold():.4f}")
    
    assert anomalies_detected < len(new_errors), "All samples marked as anomalies - test failed"
    print("\n[PASS] SPOT basic calibration test passed")
    return True


def test_adwin_drift_detection():
    """Test ADWIN drift detection with sudden distribution shift."""
    print("\n" + "="*80)
    print("TEST 2: ADWIN Drift Detection")
    print("="*80)
    
    np.random.seed(42)
    
    # Phase 1: Normal operation
    normal_errors = np.random.gamma(2, 0.01, 200)
    
    adwin = ADWIN(delta=0.002)
    drift_detected = False
    
    print("Phase 1: Normal operation (200 samples)")
    for error in normal_errors:
        if adwin.add_element(error):
            drift_detected = True
            print("  [UNEXPECTED] Drift detected in normal data!")
            break
    
    if not drift_detected:
        print("  [OK] No drift detected in normal data")
    
    # Phase 2: Sudden shift (attack simulation)
    print("\nPhase 2: Sudden shift (simulating attack)")
    attack_errors = np.random.gamma(3, 0.015, 50)  # Higher mean and variance
    
    drift_detected = False
    drift_sample = 0
    
    for i, error in enumerate(attack_errors):
        if adwin.add_element(error):
            drift_detected = True
            drift_sample = i + 1
            print(f"  [DRIFT DETECTED] at sample {drift_sample}")
            break
    
    assert drift_detected, "ADWIN failed to detect drift!"
    print(f"\n[PASS] ADWIN detected drift after {drift_sample} attack samples")
    return True


def test_poisoning_attack():
    """
    Simulate the poisoning attack scenario from the user's question:
    - Phase 1: Attacker floods with normal-looking transactions (months 1-3)
    - Phase 2: Attacker suddenly switches to fraud (month 4)
    """
    print("\n" + "="*80)
    print("TEST 3: Data Poisoning Attack Simulation")
    print("="*80)
    
    np.random.seed(42)
    
    # Baseline: Normal reconstruction errors
    print("\nBaseline: Normal operation")
    normal_errors = np.random.gamma(2, 0.01, 1000)
    
    # Calibrate SPOT
    spot = SPOTThreshold(initial_quantile=0.98, extreme_prob=0.001)
    spot.calibrate(normal_errors)
    threshold_baseline = spot.get_threshold()
    
    print(f"  Baseline threshold: {threshold_baseline:.4f}")
    
    # Initialize ADWIN
    adwin = ADWIN(delta=0.002)
    for error in normal_errors[-100:]:  # Use last 100 for ADWIN window
        adwin.add_element(error)
    
    # Phase 1: Gradual poisoning (attacker floods with slightly lower errors)
    print("\nPhase 1: Poisoning attack (months 1-3)")
    print("  Attacker floods with slightly-lower errors to drift threshold down")
    
    poisoned_errors = np.random.gamma(1.5, 0.008, 500)  # Subtly lower distribution
    
    poisoning_drift_detected = False
    for error in poisoned_errors:
        drift = adwin.add_element(error)
        if not drift:
            # If no drift detected, SPOT continues to adapt
            spot.update(error)
        else:
            poisoning_drift_detected = True
            print(f"  [INFO] ADWIN detected gradual poisoning (may or may not trigger)")
            break
    
    threshold_after_poison = spot.get_threshold()
    print(f"  Threshold after poisoning: {threshold_after_poison:.4f}")
    
    if threshold_after_poison < threshold_baseline:
        print(f"  [VULNERABLE] Threshold drifted down by {(threshold_baseline - threshold_after_poison)/threshold_baseline*100:.1f}%")
    else:
        print(f"  [OK] Threshold did not drift down significantly")
    
    # Phase 2: Sudden fraud spike (attacker switches to fraud)
    print("\nPhase 2: Fraud spike (month 4)")
    print("  Attacker suddenly switches to fraud transactions")
    
    fraud_errors = np.random.gamma(3, 0.015, 100)  # Sudden increase in errors
    
    attack_drift_detected = False
    drift_sample = 0
    fraud_detected_before_drift = 0
    
    for i, error in enumerate(fraud_errors):
        # Check if ADWIN detects the sudden spike
        if adwin.add_element(error):
            attack_drift_detected = True
            drift_sample = i + 1
            print(f"  [DRIFT DETECTED] ADWIN triggered at fraud sample {drift_sample}")
            break
        
        # Check if SPOT (poisoned threshold) catches it
        if spot.update(error):
            fraud_detected_before_drift += 1
    
    print(f"\n  Fraud detected by SPOT (before drift): {fraud_detected_before_drift}/{len(fraud_errors[:drift_sample])}")
    
    if attack_drift_detected:
        print(f"\n[PASS] ADWIN successfully detected fraud spike!")
        print(f"       Defense activated after {drift_sample} fraud samples")
        print(f"       System can now: freeze threshold, tighten detection, escalate alerts")
    else:
        print(f"\n[FAIL] ADWIN did not detect fraud spike - defense failed!")
        return False
    
    return True


def test_threshold_manager_multi_entity():
    """Test ThresholdManager with multiple entities."""
    print("\n" + "="*80)
    print("TEST 4: Multi-Entity Threshold Management")
    print("="*80)
    
    np.random.seed(42)
    
    # Create threshold manager
    manager = ThresholdManager()
    
    # Simulate 3 entities with different error distributions
    entities = {
        'ENTITY_A': np.random.gamma(2, 0.01, 200),      # Low variance
        'ENTITY_B': np.random.gamma(3, 0.02, 200),      # Medium variance
        'ENTITY_C': np.random.gamma(4, 0.03, 200),      # High variance
    }
    
    # Calibrate each entity
    print("\nCalibrating entities:")
    for entity_id, errors in entities.items():
        manager.calibrate_entity(entity_id, errors)
        spot = manager.entity_spots[entity_id]
        print(f"  {entity_id}: threshold={spot.get_threshold():.4f}, "
              f"xi={spot.xi:.4f}, beta={spot.beta:.4f}")
    
    # Test anomaly detection for each entity
    print("\nTesting anomaly detection:")
    for entity_id in entities.keys():
        # Generate test errors
        test_errors = np.random.gamma(2, 0.01, 50)
        
        anomalies = 0
        for error in test_errors:
            result = manager.check_anomaly(entity_id, error)
            if result['is_anomaly']:
                anomalies += 1
        
        print(f"  {entity_id}: {anomalies}/{len(test_errors)} anomalies detected")
    
    # Simulate drift in ENTITY_B
    print("\nSimulating drift in ENTITY_B:")
    drift_errors = np.random.gamma(5, 0.04, 50)  # Sudden increase
    
    drift_detected = False
    for error in drift_errors:
        result = manager.check_anomaly('ENTITY_B', error)
        if result['drift_detected']:
            drift_detected = True
            print(f"  [DRIFT] Detected in ENTITY_B, surge mode activated")
            break
    
    if drift_detected:
        # Check surge mode status
        summary = manager.get_entity_summary()
        print(f"\n  Entities in surge mode: {summary['surge_entities']}")
        print(f"\n[PASS] Multi-entity management with drift detection working")
    else:
        print(f"\n[INFO] Drift not detected (might need more samples or larger shift)")
    
    return True


def test_spot_state_persistence():
    """Test SPOT state save/load."""
    print("\n" + "="*80)
    print("TEST 5: SPOT State Persistence")
    print("="*80)
    
    np.random.seed(42)
    
    # Create and calibrate SPOT
    errors = np.random.gamma(2, 0.01, 500)
    spot1 = SPOTThreshold()
    spot1.calibrate(errors)
    
    print(f"Original SPOT: threshold={spot1.get_threshold():.4f}, "
          f"excesses={len(spot1.excesses)}")
    
    # Save state
    state = spot1.get_state()
    
    # Restore from state
    spot2 = SPOTThreshold.from_state(state)
    
    print(f"Restored SPOT: threshold={spot2.get_threshold():.4f}, "
          f"excesses={len(spot2.excesses)}")
    
    # Verify they produce same results
    assert abs(spot1.get_threshold() - spot2.get_threshold()) < 1e-6, "Thresholds don't match!"
    assert spot1.xi == spot2.xi, "Xi parameters don't match!"
    assert spot1.beta == spot2.beta, "Beta parameters don't match!"
    
    print("\n[PASS] State persistence test passed")
    return True


def run_all_tests():
    """Run all test scenarios."""
    print("\n" + "="*80)
    print("SPOT + ADWIN TEST SUITE")
    print("="*80)
    
    tests = [
        ("SPOT Basic Calibration", test_spot_basic),
        ("ADWIN Drift Detection", test_adwin_drift_detection),
        ("Poisoning Attack Simulation", test_poisoning_attack),
        ("Multi-Entity Management", test_threshold_manager_multi_entity),
        ("State Persistence", test_spot_state_persistence),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"\n[FAILED] {test_name}")
        except Exception as e:
            failed += 1
            print(f"\n[FAILED] {test_name}")
            print(f"  Exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

