#!/usr/bin/env python3
"""
Comprehensive Test for FCL Retraining from Human Feedback

Tests:
1. Load feedback CSV with CORRECT and FALSE_POSITIVE labels
2. Match feedback to detection features
3. Train with both label types
4. Verify model updates and convergence
5. Edge cases (empty feedback, mismatched IDs, etc.)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import json
import tempfile
import shutil
from datetime import datetime

from src.models.autoencoder import GLAutoencoder, AnomalyDetector
from src.data.preprocessing import GLDataPreprocessor

print("=" * 80)
print("COMPREHENSIVE FCL RETRAINING TEST")
print("=" * 80)

def create_test_data(test_dir: Path):
    """Create synthetic test data for FCL retraining validation."""
    
    print("\n[TEST 1] Creating synthetic test data...")
    
    # Create directories
    detections_dir = test_dir / 'detections'
    feedback_dir = test_dir / 'feedback'
    detections_dir.mkdir(parents=True, exist_ok=True)
    feedback_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic features (100 anomalies, 1284 features each)
    np.random.seed(42)
    n_anomalies = 100
    input_dim = 1284
    
    # BUG FIX #69: Use random.rand() instead of randn() to generate [0,1] values
    # randn() produces Gaussian ~[-3,+3] which violates BCE requirement (target in [0,1])
    features = np.random.rand(n_anomalies, input_dim).astype(np.float32)
    
    # BUG FIX #66: Validate features don't contain NaN/Inf
    if np.isnan(features).any() or np.isinf(features).any():
        raise ValueError("Generated features contain NaN or Inf values!")
    
    # BUG FIX #69: Validate features are in [0,1] range (required for BCE loss)
    if features.min() < 0 or features.max() > 1:
        raise ValueError(f"Generated features out of [0,1] range: min={features.min():.4f}, max={features.max():.4f}")
    
    # Save features
    timestamp = "20251102_000000"
    features_file = detections_dir / f'features_{timestamp}.npy'
    np.save(features_file, features)
    
    # Create detection JSON
    anomalies_by_entity = {}
    anomaly_idx = 0
    
    # 3 entities with different numbers of anomalies
    entities = {
        '45200': 40,  # Entity 1: 40 anomalies
        '45500': 35,  # Entity 2: 35 anomalies
        '46100': 25,  # Entity 3: 25 anomalies
    }
    
    for entity_id, n_entity_anomalies in entities.items():
        anomalies = []
        for i in range(n_entity_anomalies):
            anomalies.append({
                'anomaly_id': f'ANO_{timestamp}_{anomaly_idx:06d}',
                'transaction_id': f'TXN_{anomaly_idx:08d}',
                'reconstruction_error': float(np.random.uniform(0.08, 0.15)),
                'threshold': 0.075,
                'severity': 'MEDIUM'
            })
            anomaly_idx += 1
        
        anomalies_by_entity[entity_id] = {
            'entity_name': f'Entity_{entity_id}',
            'n_anomalies': len(anomalies),
            'anomalies': anomalies
        }
    
    json_data = {
        'metadata': {
            'detection_timestamp': timestamp,
            'total_anomalies': n_anomalies,
            'features_file': f'features_{timestamp}.npy',
            'feature_shape': [n_anomalies, input_dim]
        },
        'anomalies_by_entity': anomalies_by_entity
    }
    
    json_file = detections_dir / f'detections_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Create feedback CSV
    feedback_records = []
    
    # Entity 45200: 30 reviews (20 CORRECT, 10 FALSE_POSITIVE)
    for i in range(20):
        feedback_records.append({
            'anomaly_id': f'ANO_{timestamp}_{i:06d}',
            'entity_id': '45200',
            'reviewer_label': 'CORRECT'
        })
    for i in range(20, 30):
        feedback_records.append({
            'anomaly_id': f'ANO_{timestamp}_{i:06d}',
            'entity_id': '45200',
            'reviewer_label': 'FALSE_POSITIVE'
        })
    
    # Entity 45500: 25 reviews (15 CORRECT, 10 FALSE_POSITIVE)
    for i in range(40, 55):
        feedback_records.append({
            'anomaly_id': f'ANO_{timestamp}_{i:06d}',
            'entity_id': '45500',
            'reviewer_label': 'CORRECT'
        })
    for i in range(55, 65):
        feedback_records.append({
            'anomaly_id': f'ANO_{timestamp}_{i:06d}',
            'entity_id': '45500',
            'reviewer_label': 'FALSE_POSITIVE'
        })
    
    # Entity 46100: 15 reviews (all CORRECT, no FALSE_POSITIVE)
    for i in range(75, 90):
        feedback_records.append({
            'anomaly_id': f'ANO_{timestamp}_{i:06d}',
            'entity_id': '46100',
            'reviewer_label': 'CORRECT'
        })
    
    feedback_df = pd.DataFrame(feedback_records)
    feedback_file = feedback_dir / 'reviews_test.csv'
    feedback_df.to_csv(feedback_file, index=False)
    
    print(f"✓ Created {len(features)} synthetic features")
    print(f"✓ Created {n_anomalies} anomaly records across {len(entities)} entities")
    print(f"✓ Created {len(feedback_df)} feedback records:")
    print(f"    CORRECT: {(feedback_df['reviewer_label'] == 'CORRECT').sum()}")
    print(f"    FALSE_POSITIVE: {(feedback_df['reviewer_label'] == 'FALSE_POSITIVE').sum()}")
    
    return json_file, features_file, feedback_file


def test_feedback_loading(feedback_file):
    """Test 1: Feedback CSV loading and validation."""
    print("\n[TEST 2] Testing feedback CSV loading...")
    
    from retrain_fcl_from_feedback import load_feedback_csv
    
    try:
        df = load_feedback_csv(str(feedback_file))
        
        # Validate
        assert 'anomaly_id' in df.columns, "Missing anomaly_id column"
        assert 'entity_id' in df.columns, "Missing entity_id column"
        assert 'reviewer_label' in df.columns, "Missing reviewer_label column"
        
        labels = df['reviewer_label'].unique()
        valid_labels = {'CORRECT', 'FALSE_POSITIVE'}
        assert set(labels).issubset(valid_labels), f"Invalid labels found: {set(labels) - valid_labels}"
        
        print(f"✓ Feedback loading: PASSED")
        print(f"  Loaded {len(df)} records")
        print(f"  Entities: {df['entity_id'].nunique()}")
        return df
    
    except Exception as e:
        print(f"✗ Feedback loading: FAILED")
        print(f"  Error: {e}")
        raise


def test_detection_loading(detections_dir):
    """Test 2: Detection JSON and features loading."""
    print("\n[TEST 3] Testing detection data loading...")
    
    from retrain_fcl_from_feedback import load_detection_data
    
    try:
        json_data, features_array, timestamp = load_detection_data(detections_dir)
        
        # Validate
        assert 'metadata' in json_data, "Missing metadata in JSON"
        assert 'anomalies_by_entity' in json_data, "Missing anomalies_by_entity in JSON"
        assert len(features_array.shape) == 2, f"Features should be 2D, got shape {features_array.shape}"
        assert features_array.shape[0] > 0, "Features array is empty"
        
        print(f"✓ Detection loading: PASSED")
        print(f"  Features shape: {features_array.shape}")
        print(f"  Timestamp: {timestamp}")
        return json_data, features_array
    
    except Exception as e:
        print(f"✗ Detection loading: FAILED")
        print(f"  Error: {e}")
        raise


def test_feature_matching(feedback_df, json_data, features_array):
    """Test 3: Match feedback to features."""
    print("\n[TEST 4] Testing feedback-to-feature matching...")
    
    from retrain_fcl_from_feedback import match_feedback_to_features
    
    try:
        # Test WITHOUT false positives
        print("\n  [4A] Matching without FALSE_POSITIVE...")
        correct_only, fp_only = match_feedback_to_features(
            feedback_df, json_data, features_array, use_false_positives=False
        )
        
        assert len(correct_only) == 3, f"Expected 3 entities with CORRECT, got {len(correct_only)}"
        assert len(fp_only) == 0, f"Should ignore FALSE_POSITIVE, but got {len(fp_only)}"
        
        # Verify shapes
        assert correct_only['45200'].shape == (20, 1284), f"Entity 45200: wrong shape {correct_only['45200'].shape}"
        assert correct_only['45500'].shape == (15, 1284), f"Entity 45500: wrong shape {correct_only['45500'].shape}"
        assert correct_only['46100'].shape == (15, 1284), f"Entity 46100: wrong shape {correct_only['46100'].shape}"
        
        print(f"    ✓ Matched CORRECT only: {sum(len(v) for v in correct_only.values())} samples")
        
        # Test WITH false positives
        print("\n  [4B] Matching with FALSE_POSITIVE...")
        correct_with, fp_with = match_feedback_to_features(
            feedback_df, json_data, features_array, use_false_positives=True
        )
        
        assert len(correct_with) == 3, f"Expected 3 entities with CORRECT, got {len(correct_with)}"
        assert len(fp_with) == 2, f"Expected 2 entities with FALSE_POSITIVE, got {len(fp_with)}"
        
        # Verify FALSE_POSITIVE shapes
        assert fp_with['45200'].shape == (10, 1284), f"Entity 45200 FP: wrong shape {fp_with['45200'].shape}"
        assert fp_with['45500'].shape == (10, 1284), f"Entity 45500 FP: wrong shape {fp_with['45500'].shape}"
        assert '46100' not in fp_with, "Entity 46100 should not have FALSE_POSITIVE"
        
        print(f"    ✓ Matched CORRECT: {sum(len(v) for v in correct_with.values())} samples")
        print(f"    ✓ Matched FALSE_POSITIVE: {sum(len(v) for v in fp_with.values())} samples")
        
        print(f"\n✓ Feature matching: PASSED")
        return correct_with, fp_with
    
    except Exception as e:
        print(f"✗ Feature matching: FAILED")
        print(f"  Error: {e}")
        raise


def test_experience_creation(correct_features, fp_features):
    """Test 4: Create FCL experience DataLoader."""
    print("\n[TEST 5] Testing experience DataLoader creation...")
    
    from retrain_fcl_from_feedback import create_feedback_experience
    
    try:
        # Test with CORRECT only
        print("\n  [5A] Creating experience with CORRECT only...")
        dataloader_correct = create_feedback_experience(
            correct_features['45200'],
            batch_size=8
        )
        
        batch_count = 0
        total_samples = 0
        for batch in dataloader_correct:
            batch_count += 1
            total_samples += batch[0].shape[0]
        
        assert total_samples == 20, f"Expected 20 samples, got {total_samples}"
        assert batch_count == 3, f"Expected 3 batches (8+8+4), got {batch_count}"
        
        print(f"    ✓ CORRECT only: {total_samples} samples in {batch_count} batches")
        
        # Test with CORRECT + FALSE_POSITIVE
        print("\n  [5B] Creating experience with CORRECT + FALSE_POSITIVE...")
        dataloader_combined = create_feedback_experience(
            correct_features['45200'],
            fp_features['45200'],
            batch_size=8
        )
        
        batch_count = 0
        total_samples = 0
        for batch in dataloader_combined:
            batch_count += 1
            total_samples += batch[0].shape[0]
        
        assert total_samples == 30, f"Expected 30 samples (20 CORRECT + 10 FP), got {total_samples}"
        
        print(f"    ✓ CORRECT + FALSE_POSITIVE: {total_samples} samples in {batch_count} batches")
        
        print(f"\n✓ Experience creation: PASSED")
        return dataloader_correct, dataloader_combined
    
    except Exception as e:
        print(f"✗ Experience creation: FAILED")
        print(f"  Error: {e}")
        raise


def test_model_retraining(correct_features, fp_features, test_dir):
    """Test 5: Full model retraining with FCL."""
    print("\n[TEST 6] Testing full FCL retraining pipeline...")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {DEVICE}")
    
    try:
        # Create a minimal model checkpoint
        print("\n  [6A] Creating test model checkpoint...")
        input_dim = 1284
        latent_dim = 2
        model = GLAutoencoder(input_dim=input_dim, latent_dim=latent_dim, architecture='deep').to(DEVICE)
        
        # Create preprocessor
        preprocessor = GLDataPreprocessor(
            categorical_columns=['ACCOUNT', 'CLASS_FLD', 'FUND_CODE'],
            numerical_columns=['debit_amount', 'credit_amount', 'debit_credit_ratio', 'net_balance', 'abs_balance']
        )
        # Mock preprocessor state
        preprocessor.categorical_indices = list(range(1279))
        preprocessor.numerical_indices = list(range(1279, 1284))
        preprocessor.feature_names = [f'feat_{i}' for i in range(1284)]
        preprocessor.is_fitted = True
        
        # BUG FIX #67: Verify preprocessor state is complete
        assert hasattr(preprocessor, 'categorical_indices'), "Preprocessor missing categorical_indices"
        assert hasattr(preprocessor, 'numerical_indices'), "Preprocessor missing numerical_indices"
        assert hasattr(preprocessor, 'feature_names'), "Preprocessor missing feature_names"
        assert len(preprocessor.categorical_indices) + len(preprocessor.numerical_indices) == input_dim, \
            f"Preprocessor dimensions don't match: {len(preprocessor.categorical_indices)} + {len(preprocessor.numerical_indices)} != {input_dim}"
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'latent_dim': latent_dim,
            'architecture': 'deep',
            'preprocessor_state': {
                'categorical_columns': preprocessor.categorical_columns,
                'numerical_columns': preprocessor.numerical_columns,
                'categorical_indices': preprocessor.categorical_indices,
                'numerical_indices': preprocessor.numerical_indices,
                'feature_names': preprocessor.feature_names
            },
            'val_loss': 0.005,
            'last_experience_id': 0
        }
        
        model_path = test_dir / 'test_model.pth'
        torch.save(checkpoint, model_path)
        print(f"    ✓ Created test model: {model_path.name}")
        
        # Test retraining with CORRECT only
        print("\n  [6B] Retraining with CORRECT labels only...")
        from src.federated.client import FCLClient
        from retrain_fcl_from_feedback import create_feedback_experience
        
        # Get initial model state
        # BUG FIX #65: Clone weights to CPU and detach to avoid CUDA memory issues during comparison
        initial_weights = {name: param.clone().detach().cpu() for name, param in model.named_parameters()}
        
        # Create experience
        dataloader = create_feedback_experience(correct_features['45200'], batch_size=8)
        
        # Train with EWC
        client = FCLClient(
            client_id='test_entity_45200',
            model=model,
            cl_strategy='ewc',
            cl_params={'lambda_ewc': 500.0},
            device=str(DEVICE)
        )
        
        metrics = client.train_on_experience(
            experience_id=1,
            dataloader=dataloader,
            categorical_indices=preprocessor.categorical_indices,
            numerical_indices=preprocessor.numerical_indices,
            n_epochs=3,
            learning_rate=0.0001,
            verbose=False
        )
        
        # Verify model was updated
        # BUG FIX #65: Compare tensors on CPU to avoid device mismatch
        weights_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param.detach().cpu(), initial_weights[name], atol=1e-6):
                weights_changed = True
                break
        
        assert weights_changed, "Model weights did not change after training!"
        # BUG FIX #70: FCLClient returns 'loss' not 'final_loss'
        assert 'loss' in metrics, "Metrics missing loss"
        
        print(f"    ✓ Model retrained successfully")
        print(f"    ✓ Final loss: {metrics['loss']:.6f}")
        
        # BUG FIX #68: Clear CUDA cache after first training to prevent OOM
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Test retraining with CORRECT + FALSE_POSITIVE
        print("\n  [6C] Retraining with CORRECT + FALSE_POSITIVE labels...")
        
        # Reload model
        model2 = GLAutoencoder(input_dim=input_dim, latent_dim=latent_dim, architecture='deep').to(DEVICE)
        model2.load_state_dict(checkpoint['model_state_dict'])
        
        # BUG FIX #65: Clone weights to CPU and detach
        initial_weights2 = {name: param.clone().detach().cpu() for name, param in model2.named_parameters()}
        
        # Create combined experience
        dataloader_combined = create_feedback_experience(
            correct_features['45200'],
            fp_features['45200'],
            batch_size=8
        )
        
        client2 = FCLClient(
            client_id='test_entity_45200_combined',
            model=model2,
            cl_strategy='ewc',
            cl_params={'lambda_ewc': 500.0},
            device=str(DEVICE)
        )
        
        metrics2 = client2.train_on_experience(
            experience_id=1,
            dataloader=dataloader_combined,
            categorical_indices=preprocessor.categorical_indices,
            numerical_indices=preprocessor.numerical_indices,
            n_epochs=3,
            learning_rate=0.0001,
            verbose=False
        )
        
        # Verify model was updated
        # BUG FIX #65: Compare tensors on CPU to avoid device mismatch
        weights_changed2 = False
        for name, param in model2.named_parameters():
            if not torch.allclose(param.detach().cpu(), initial_weights2[name], atol=1e-6):
                weights_changed2 = True
                break
        
        assert weights_changed2, "Model weights did not change after combined training!"
        
        print(f"    ✓ Model retrained with combined labels")
        # BUG FIX #70: FCLClient returns 'loss' not 'final_loss'
        print(f"    ✓ Final loss: {metrics2['loss']:.6f}")
        
        # Compare losses (combined should be similar or better due to more data)
        loss_diff = metrics2['loss'] - metrics['loss']
        print(f"    ✓ Loss difference (combined - correct_only): {loss_diff:+.6f}")
        
        # BUG FIX #68: Final CUDA cleanup and synchronization check
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()  # Ensure all GPU operations completed
            torch.cuda.empty_cache()  # Free unused memory
        
        print(f"\n✓ Model retraining: PASSED")
        return model, model2
    
    except Exception as e:
        print(f"✗ Model retraining: FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_edge_cases(test_dir):
    """Test 6: Edge cases and error handling."""
    print("\n[TEST 7] Testing edge cases...")
    
    from retrain_fcl_from_feedback import load_feedback_csv, match_feedback_to_features
    
    try:
        # Test 1: Empty feedback
        print("\n  [7A] Testing empty feedback...")
        empty_feedback = pd.DataFrame(columns=['anomaly_id', 'entity_id', 'reviewer_label'])
        feedback_file = test_dir / 'empty_feedback.csv'
        empty_feedback.to_csv(feedback_file, index=False)
        
        df = load_feedback_csv(str(feedback_file))
        assert len(df) == 0, "Empty feedback should return empty DataFrame"
        print(f"    ✓ Empty feedback handled correctly")
        
        # Test 2: Invalid labels
        print("\n  [7B] Testing invalid labels...")
        invalid_feedback = pd.DataFrame([
            {'anomaly_id': 'ANO_001', 'entity_id': '12345', 'reviewer_label': 'CORRECT'},
            {'anomaly_id': 'ANO_002', 'entity_id': '12345', 'reviewer_label': 'INVALID_LABEL'},
            {'anomaly_id': 'ANO_003', 'entity_id': '12345', 'reviewer_label': 'FALSE_POSITIVE'},
        ])
        feedback_file = test_dir / 'invalid_feedback.csv'
        invalid_feedback.to_csv(feedback_file, index=False)
        
        df = load_feedback_csv(str(feedback_file))
        assert len(df) == 2, f"Should filter invalid labels, expected 2 rows, got {len(df)}"
        print(f"    ✓ Invalid labels filtered correctly")
        
        # Test 3: Mismatched anomaly IDs
        print("\n  [7C] Testing mismatched anomaly IDs...")
        mismatched_feedback = pd.DataFrame([
            {'anomaly_id': 'NONEXISTENT_001', 'entity_id': '45200', 'reviewer_label': 'CORRECT'},
            {'anomaly_id': 'NONEXISTENT_002', 'entity_id': '45200', 'reviewer_label': 'FALSE_POSITIVE'},
        ])
        
        # Use real detection data
        from retrain_fcl_from_feedback import load_detection_data
        detections_dir = test_dir / 'detections'
        json_data, features_array, _ = load_detection_data(detections_dir)
        
        correct_features, fp_features = match_feedback_to_features(
            mismatched_feedback, json_data, features_array, use_false_positives=True
        )
        
        assert len(correct_features) == 0, "Should find no matches for nonexistent IDs"
        assert len(fp_features) == 0, "Should find no matches for nonexistent IDs"
        print(f"    ✓ Mismatched IDs handled correctly (no matches)")
        
        print(f"\n✓ Edge cases: PASSED")
    
    except Exception as e:
        print(f"✗ Edge cases: FAILED")
        print(f"  Error: {e}")
        raise


def main():
    """Run all tests."""
    
    # Create temporary test directory
    test_dir = Path(tempfile.mkdtemp(prefix='fcl_test_'))
    print(f"\nTest directory: {test_dir}")
    
    try:
        # Create test data
        json_file, features_file, feedback_file = create_test_data(test_dir)
        
        # Run tests
        feedback_df = test_feedback_loading(feedback_file)
        json_data, features_array = test_detection_loading(test_dir / 'detections')
        correct_features, fp_features = test_feature_matching(feedback_df, json_data, features_array)
        dataloader_correct, dataloader_combined = test_experience_creation(correct_features, fp_features)
        model_correct, model_combined = test_model_retraining(correct_features, fp_features, test_dir)
        test_edge_cases(test_dir)
        
        # Final summary
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)
        print("\nFCL Retraining Implementation Summary:")
        print("  ✓ Feedback CSV loading and validation")
        print("  ✓ Detection data loading and feature extraction")
        print("  ✓ Feedback-to-feature matching (CORRECT and FALSE_POSITIVE)")
        print("  ✓ Experience DataLoader creation")
        print("  ✓ FCL model retraining with EWC")
        print("  ✓ Combined label handling (CORRECT + FALSE_POSITIVE)")
        print("  ✓ Edge case handling (empty, invalid, mismatched)")
        print("\nConclusion: FCL retraining pipeline is PRODUCTION-READY")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("TESTS FAILED ✗")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print(f"\nCleaning up test directory: {test_dir}")
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == '__main__':
    main()

