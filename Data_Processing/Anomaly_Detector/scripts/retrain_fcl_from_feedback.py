#!/usr/bin/env python3
"""
Retrain FCL Autoencoder from Human Feedback

Uses human-verified anomalies to fine-tune the autoencoder with continual learning.
Separate from ADT threshold learning - this updates the model weights themselves.

Usage:
    python retrain_fcl_from_feedback.py \
        --feedback data/feedback/reviews_batch_001.csv \
        --detections data/detections/detections_20251101_120000.json \
        --n-epochs 5 \
        --learning-rate 0.0001
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
import json
import argparse
from typing import Dict, List, Tuple

from src.models.autoencoder import GLAutoencoder
from src.data.preprocessing import GLDataPreprocessor
from src.federated.client import FCLClient

print("=" * 80)
print("FCL RETRAINING FROM HUMAN FEEDBACK")
print("=" * 80)


def load_feedback_csv(feedback_file: str) -> pd.DataFrame:
    """
    Load and validate human feedback CSV.
    
    Expected format:
        anomaly_id,entity_id,reviewer_label
        ANO_20251101_000045,45200,CORRECT
        ANO_20251101_000123,45200,FALSE_POSITIVE
    
    Args:
        feedback_file: Path to feedback CSV
    
    Returns:
        DataFrame with validated feedback
    
    Raises:
        FileNotFoundError: If feedback file doesn't exist
        ValueError: If required columns are missing
    """
    if not Path(feedback_file).exists():
        raise FileNotFoundError(f"Feedback file not found: {feedback_file}")
    
    df = pd.read_csv(feedback_file)
    
    # Validate required columns
    required_cols = ['anomaly_id', 'entity_id', 'reviewer_label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Feedback CSV missing required columns: {missing_cols}")
    
    # Validate labels
    valid_labels = {'CORRECT', 'FALSE_POSITIVE'}
    invalid_labels = set(df['reviewer_label'].unique()) - valid_labels
    if invalid_labels:
        print(f"[WARNING] Found invalid labels: {invalid_labels}. Valid labels: {valid_labels}")
        df = df[df['reviewer_label'].isin(valid_labels)]
    
    # Force entity_id to string (consistency with detection exports)
    df['entity_id'] = df['entity_id'].astype(str)
    
    print(f"[OK] Loaded {len(df)} feedback records from: {feedback_file}")
    print(f"  CORRECT: {(df['reviewer_label'] == 'CORRECT').sum()}")
    print(f"  FALSE_POSITIVE: {(df['reviewer_label'] == 'FALSE_POSITIVE').sum()}")
    print(f"  Entities: {df['entity_id'].nunique()}")
    
    return df


def load_detection_data(
    detections_dir: Path,
    detection_file: str = None
) -> Tuple[Dict, np.ndarray, str]:
    """
    Load detection JSON and corresponding .npy features.
    
    Args:
        detections_dir: Path to detections directory
        detection_file: Specific detection JSON file (if None, uses most recent)
    
    Returns:
        Tuple of (json_data, features_array, timestamp)
    
    Raises:
        FileNotFoundError: If no detection files found
    """
    if not detections_dir.exists():
        raise FileNotFoundError(f"Detections directory not found: {detections_dir}")
    
    # Find detection file
    if detection_file is None:
        # Use most recent detection file
        detection_files = sorted(detections_dir.glob('detections_*.json'))
        if not detection_files:
            raise FileNotFoundError(f"No detection files found in {detections_dir}")
        detection_file = detection_files[-1]
        print(f"[INFO] Using most recent detection file: {detection_file.name}")
    else:
        detection_file = detections_dir / detection_file
        if not detection_file.exists():
            raise FileNotFoundError(f"Detection file not found: {detection_file}")
    
    # Load JSON
    with open(detection_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Extract timestamp and features file from metadata
    metadata = json_data.get('metadata', {})
    features_filename = metadata.get('features_file')
    if not features_filename:
        raise ValueError("Detection JSON missing 'features_file' in metadata")
    
    # Load features
    features_file = detections_dir / features_filename
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}")
    
    features_array = np.load(features_file)
    
    # Extract timestamp from filename (detections_YYYYMMDD_HHMMSS.json)
    timestamp = detection_file.stem.replace('detections_', '')
    
    print(f"[OK] Loaded detection data from: {detection_file.name}")
    print(f"  Features file: {features_filename}")
    print(f"  Feature shape: {features_array.shape}")
    print(f"  Total anomalies: {metadata.get('total_anomalies', 'unknown')}")
    
    return json_data, features_array, timestamp


def match_feedback_to_features(
    feedback_df: pd.DataFrame,
    json_data: Dict,
    features_array: np.ndarray,
    use_false_positives: bool = False
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Match feedback anomaly_ids to feature vectors.
    
    Args:
        feedback_df: DataFrame with feedback
        json_data: Detection JSON data
        features_array: Array of anomaly features [n_anomalies, input_dim]
        use_false_positives: Whether to use FALSE_POSITIVE labels as normal examples
    
    Returns:
        Tuple of (correct_features_by_entity, false_positive_features_by_entity)
        Each is a dict: {entity_id: np.ndarray of features}
    """
    # Build anomaly_id -> index mapping from JSON
    anomaly_id_to_index = {}
    anomaly_id_to_entity = {}
    
    idx = 0
    for entity_id, entity_data in json_data['anomalies_by_entity'].items():
        for anomaly_record in entity_data['anomalies']:
            anomaly_id = anomaly_record['anomaly_id']
            anomaly_id_to_index[anomaly_id] = idx
            anomaly_id_to_entity[anomaly_id] = entity_id
            idx += 1
    
    print(f"\n[INFO] Matching {len(feedback_df)} feedback records to features...")
    
    # Match feedback to features
    correct_features = {}
    false_positive_features = {}
    matched_count = 0
    unmatched_count = 0
    
    for _, row in feedback_df.iterrows():
        anomaly_id = row['anomaly_id']
        entity_id = row['entity_id']
        label = row['reviewer_label']
        
        if anomaly_id not in anomaly_id_to_index:
            unmatched_count += 1
            continue
        
        # Get feature vector with bounds checking
        feature_idx = anomaly_id_to_index[anomaly_id]
        if feature_idx >= len(features_array):
            print(f"[WARNING] Feature index {feature_idx} out of bounds for anomaly {anomaly_id} (features array has {len(features_array)} entries)")
            unmatched_count += 1
            continue
        
        feature_vector = features_array[feature_idx]
        
        # Group by entity
        if label == 'CORRECT':
            if entity_id not in correct_features:
                correct_features[entity_id] = []
            correct_features[entity_id].append(feature_vector)
            matched_count += 1
        
        elif label == 'FALSE_POSITIVE' and use_false_positives:
            if entity_id not in false_positive_features:
                false_positive_features[entity_id] = []
            false_positive_features[entity_id].append(feature_vector)
            matched_count += 1
    
    # Convert lists to arrays
    correct_features = {k: np.array(v) for k, v in correct_features.items()}
    false_positive_features = {k: np.array(v) for k, v in false_positive_features.items()}
    
    print(f"[OK] Matched {matched_count} feedback records")
    if unmatched_count > 0:
        print(f"[WARNING] {unmatched_count} feedback records not found in detection data")
    print(f"  Entities with CORRECT feedback: {len(correct_features)}")
    if use_false_positives:
        print(f"  Entities with FALSE_POSITIVE feedback: {len(false_positive_features)}")
    
    return correct_features, false_positive_features


def create_feedback_experience(
    correct_features: np.ndarray,
    false_positive_features: np.ndarray = None,
    batch_size: int = 32
) -> DataLoader:
    """
    Create FCL experience DataLoader from feedback features.
    
    Args:
        correct_features: Features for CORRECT anomalies [n_correct, input_dim]
        false_positive_features: Features for FALSE_POSITIVE (optional) [n_fp, input_dim]
        batch_size: Batch size for DataLoader
    
    Returns:
        DataLoader for training
    """
    # Combine CORRECT and FALSE_POSITIVE features
    if false_positive_features is not None and len(false_positive_features) > 0:
        all_features = np.vstack([correct_features, false_positive_features])
        print(f"[INFO] Created experience with {len(correct_features)} CORRECT + {len(false_positive_features)} FALSE_POSITIVE")
    else:
        all_features = correct_features
        print(f"[INFO] Created experience with {len(correct_features)} CORRECT")
    
    # Create DataLoader
    dataset = TensorDataset(torch.FloatTensor(all_features))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def retrain_model(args):
    """Main retraining logic."""
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {DEVICE}")
    
    # ========================================================================
    # LOAD FEEDBACK
    # ========================================================================
    print("\n" + "=" * 80)
    print("LOADING HUMAN FEEDBACK")
    print("=" * 80)
    
    feedback_df = load_feedback_csv(args.feedback)
    
    # ========================================================================
    # LOAD DETECTION DATA
    # ========================================================================
    print("\n" + "=" * 80)
    print("LOADING DETECTION DATA")
    print("=" * 80)
    
    detections_dir = Path(args.detections_dir)
    json_data, features_array, timestamp = load_detection_data(
        detections_dir,
        args.detections
    )
    
    # ========================================================================
    # MATCH FEEDBACK TO FEATURES
    # ========================================================================
    print("\n" + "=" * 80)
    print("MATCHING FEEDBACK TO FEATURES")
    print("=" * 80)
    
    correct_features, false_positive_features = match_feedback_to_features(
        feedback_df,
        json_data,
        features_array,
        use_false_positives=args.use_false_positives
    )
    
    if len(correct_features) == 0:
        print("\n[ERROR] No CORRECT feedback matched. Cannot retrain.")
        return
    
    # ========================================================================
    # LOAD EXISTING MODEL
    # ========================================================================
    print("\n" + "=" * 80)
    print("LOADING EXISTING MODEL")
    print("=" * 80)
    
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Infer latent_dim from encoder weights (handles any checkpoint)
    encoder_state = {k: v for k, v in checkpoint['model_state_dict'].items() if k.startswith('encoder.')}
    last_encoder_key = sorted([k for k in encoder_state.keys() if '.weight' in k])[-1]
    latent_dim = encoder_state[last_encoder_key].shape[0]
    
    architecture = checkpoint.get('architecture', 'deep')
    input_dim = checkpoint.get('input_dim')
    if input_dim is None:
        # Infer from first encoder layer
        first_encoder_key = sorted([k for k in encoder_state.keys() if '.weight' in k])[0]
        input_dim = encoder_state[first_encoder_key].shape[1]
    
    print(f"[INFO] Model architecture: {architecture}")
    print(f"[INFO] Input dim: {input_dim}, Latent dim: {latent_dim}")
    
    # Validate feature dimensions match
    if features_array.shape[1] != input_dim:
        raise ValueError(
            f"Feature dimension mismatch: "
            f"model expects {input_dim}, detection features have {features_array.shape[1]}"
        )
    
    # Load preprocessor
    if 'preprocessor_state' not in checkpoint:
        raise ValueError("Checkpoint missing preprocessor_state. Cannot retrain.")
    
    # Reconstruct preprocessor from checkpoint state
    preprocessor_state = checkpoint['preprocessor_state']
    preprocessor = GLDataPreprocessor(
        categorical_columns=preprocessor_state['categorical_columns'],
        numerical_columns=preprocessor_state['numerical_columns']
    )
    # Restore fitted state
    preprocessor.categorical_indices = preprocessor_state['categorical_indices']
    preprocessor.numerical_indices = preprocessor_state['numerical_indices']
    preprocessor.feature_names = preprocessor_state['feature_names']
    preprocessor.is_fitted = True
    
    print(f"[OK] Loaded preprocessor:")
    print(f"  Categorical features: {len(preprocessor.categorical_indices)}")
    print(f"  Numerical features: {len(preprocessor.numerical_indices)}")
    
    # Create model
    model = GLAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        architecture=architecture
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[OK] Loaded model from: {model_path.name}")
    
    # ========================================================================
    # RETRAIN PER ENTITY
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"RETRAINING WITH {args.cl_strategy.upper()} CONTINUAL LEARNING")
    print("=" * 80)
    
    # Combine all entities with feedback
    all_entity_ids = set(correct_features.keys())
    if args.use_false_positives:
        all_entity_ids.update(false_positive_features.keys())
    
    print(f"\n[INFO] Retraining on {len(all_entity_ids)} entities with feedback")
    print(f"  CL Strategy: {args.cl_strategy}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.n_epochs}")
    print(f"  Batch size: {args.batch_size}")
    
    # Get next experience ID (increment from checkpoint)
    next_experience_id = checkpoint.get('last_experience_id', 0) + 1
    print(f"  Experience ID: {next_experience_id}")
    
    for entity_idx, entity_id in enumerate(sorted(all_entity_ids), 1):
        print(f"\n[{entity_idx}/{len(all_entity_ids)}] Entity {entity_id}")
        
        # Get features for this entity
        entity_correct = correct_features.get(entity_id, np.array([]))
        entity_fp = false_positive_features.get(entity_id, np.array([])) if args.use_false_positives else np.array([])
        
        if len(entity_correct) == 0:
            print(f"  [SKIP] No CORRECT feedback for entity {entity_id}")
            continue
        
        # Create experience dataloader
        experience_dataloader = create_feedback_experience(
            entity_correct,
            entity_fp if len(entity_fp) > 0 else None,
            batch_size=args.batch_size
        )
        
        # Create FCL client
        cl_params = {}
        if args.cl_strategy == 'ewc':
            cl_params['lambda_ewc'] = args.lambda_ewc
        elif args.cl_strategy == 'replay':
            cl_params['buffer_size'] = args.buffer_size
            cl_params['replay_batch_size'] = args.batch_size
        elif args.cl_strategy == 'lwf':
            cl_params['lambda_lwf'] = args.lambda_lwf
        
        client = FCLClient(
            client_id=f"entity_{entity_id}",
            model=model,
            cl_strategy=args.cl_strategy,
            cl_params=cl_params,
            device=str(DEVICE)
        )
        
        # Train on feedback experience
        metrics = client.train_on_experience(
            experience_id=next_experience_id,
            dataloader=experience_dataloader,
            categorical_indices=preprocessor.categorical_indices,
            numerical_indices=preprocessor.numerical_indices,
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            verbose=True
        )
        
        # Update model (client may have modified weights)
        model = client.model
        
        print(f"  [OK] Final loss: {metrics.get('final_loss', 0.0):.6f}")
    
    # ========================================================================
    # SAVE UPDATED MODEL
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING UPDATED MODEL")
    print("=" * 80)
    
    # Update checkpoint
    checkpoint['model_state_dict'] = model.state_dict()
    checkpoint['last_fcl_feedback_update'] = datetime.now().isoformat()
    checkpoint['last_experience_id'] = next_experience_id
    checkpoint['fcl_feedback_history'] = checkpoint.get('fcl_feedback_history', [])
    checkpoint['fcl_feedback_history'].append({
        'timestamp': datetime.now().isoformat(),
        'feedback_file': args.feedback,
        'detection_timestamp': timestamp,
        'n_entities': len(all_entity_ids),
        'n_correct': sum(len(v) for v in correct_features.values()),
        'n_false_positive': sum(len(v) for v in false_positive_features.values()) if args.use_false_positives else 0,
        'cl_strategy': args.cl_strategy,
        'n_epochs': args.n_epochs
    })
    
    # Determine output path
    output_path = Path(args.output) if args.output else model_path
    
    # BUG FIX #58: Save to temporary file first to prevent corruption if save fails
    import tempfile
    import shutil
    
    temp_fd, temp_path = tempfile.mkstemp(suffix='.pth', dir=output_path.parent)
    try:
        # Close the file descriptor, torch.save will reopen it
        import os
        os.close(temp_fd)
        
        # Save to temporary file
        torch.save(checkpoint, temp_path)
        
        # Atomic rename (overwrites target safely)
        shutil.move(temp_path, output_path)
        
        print(f"[OK] Updated model saved to: {output_path}")
    except Exception as e:
        # Clean up temp file if save failed
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        raise RuntimeError(f"Failed to save checkpoint: {e}")
    print(f"  Experience ID: {next_experience_id}")
    print(f"  Entities retrained: {len(all_entity_ids)}")
    print(f"  Total feedback samples: {sum(len(v) for v in correct_features.values())}")
    if args.use_false_positives:
        print(f"  FALSE_POSITIVE samples used: {sum(len(v) for v in false_positive_features.values())}")
    
    print("\n" + "=" * 80)
    print("FCL RETRAINING COMPLETE!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Retrain FCL autoencoder from human feedback',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python retrain_fcl_from_feedback.py \\
        --feedback data/feedback/reviews_batch_001.csv \\
        --n-epochs 5 \\
        --learning-rate 0.0001
        """
    )
    
    parser.add_argument(
        '--feedback',
        required=True,
        help='Path to feedback CSV (anomaly_id,entity_id,reviewer_label)'
    )
    parser.add_argument(
        '--detections',
        help='Specific detection JSON file (default: most recent in detections_dir)'
    )
    parser.add_argument(
        '--detections-dir',
        default='data/detections',
        help='Directory containing detection files (default: data/detections)'
    )
    parser.add_argument(
        '--model',
        default='data/models/federated_optimized_rtx4070.pth',
        help='Input model checkpoint (default: data/models/federated_optimized_rtx4070.pth)'
    )
    parser.add_argument(
        '--output',
        help='Output model path (default: overwrite input model)'
    )
    parser.add_argument(
        '--cl-strategy',
        default='ewc',
        choices=['ewc', 'replay', 'lwf'],
        help='Continual learning strategy (default: ewc)'
    )
    parser.add_argument(
        '--lambda-ewc',
        type=float,
        default=500.0,
        help='EWC regularization strength (default: 500.0)'
    )
    parser.add_argument(
        '--lambda-lwf',
        type=float,
        default=1.2,
        help='LwF distillation strength (default: 1.2)'
    )
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=1000,
        help='Replay buffer size (default: 1000)'
    )
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=5,
        help='Training epochs per entity (default: 5)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.0001,
        help='Learning rate for fine-tuning (default: 0.0001)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--use-false-positives',
        action='store_true',
        help='Include FALSE_POSITIVE labels as normal examples'
    )
    
    args = parser.parse_args()
    
    try:
        retrain_model(args)
    except Exception as e:
        print(f"\n[ERROR] Retraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

