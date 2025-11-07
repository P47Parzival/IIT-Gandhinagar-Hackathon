"""
Process Human Feedback for ADT Training
Loads human review labels and updates ADT controllers
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from src.models.autoencoder import GLAutoencoder, AnomalyDetector
from src.models.threshold_manager import ThresholdManager


def load_feedback_csv(feedback_file):
    """
    Load human feedback from CSV.
    
    Expected CSV format:
        anomaly_id,entity_id,reviewer_label,review_timestamp,reviewer_name
        ANO_20251101_000045,AGY_45200,CORRECT,2025-11-01 10:30:00,John Doe
        ANO_20251101_000123,AGY_45200,FALSE_POSITIVE,2025-11-01 10:31:00,John Doe
        ...
    
    Args:
        feedback_file: Path to feedback CSV
    
    Returns:
        List of feedback records (dicts)
    """
    print(f"Loading feedback from: {feedback_file}")
    df = pd.read_csv(feedback_file)
    
    # Validate required columns
    required_cols = ['anomaly_id', 'entity_id', 'reviewer_label']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in feedback CSV: {missing}")
    
    # Validate labels
    valid_labels = ['CORRECT', 'FALSE_POSITIVE']
    invalid_labels = df[~df['reviewer_label'].isin(valid_labels)]
    if len(invalid_labels) > 0:
        print(f"[WARNING] {len(invalid_labels)} records have invalid labels, skipping...")
        df = df[df['reviewer_label'].isin(valid_labels)]
    
    print(f"  Loaded {len(df)} feedback records")
    print(f"  CORRECT: {(df['reviewer_label'] == 'CORRECT').sum()}")
    print(f"  FALSE_POSITIVE: {(df['reviewer_label'] == 'FALSE_POSITIVE').sum()}")
    
    records = df.to_dict('records')
    return records


def update_adt_from_feedback(detector, feedback_records):
    """
    Update ADT controllers based on human feedback.
    
    Args:
        detector: AnomalyDetector with threshold_manager
        feedback_records: List of feedback dicts from load_feedback_csv()
    
    Returns:
        Dictionary with update statistics
    """
    if not detector.threshold_manager.enable_adt:
        print("[WARNING] ADT not enabled! Call detector.threshold_manager.enable_adt_learning() first.")
        return
    
    print("\n" + "="*80)
    print("UPDATING ADT FROM HUMAN FEEDBACK")
    print("="*80)
    
    # Group feedback by entity
    feedback_by_entity = {}
    for record in feedback_records:
        entity = record['entity_id']
        if entity not in feedback_by_entity:
            feedback_by_entity[entity] = []
        
        feedback_by_entity[entity].append({
            'is_correct': record['reviewer_label'] == 'CORRECT',
            'anomaly_id': record['anomaly_id']
        })
    
    # Update each entity's ADT
    stats = {
        'entities_updated': 0,
        'total_feedback': len(feedback_records),
        'precision_by_entity': {}
    }
    
    for entity_id, batch in feedback_by_entity.items():
        # Skip empty batches (shouldn't happen but be safe)
        if len(batch) == 0:
            print(f"  [WARNING] Skipping {entity_id}: empty feedback batch")
            continue
        
        # Calculate precision for this entity
        n_correct = sum(1 for f in batch if f['is_correct'])
        precision = n_correct / len(batch)
        
        # Estimate alert rate (use default if not available)
        # In production, this should come from detection logs
        alert_rate = 0.05  # Default 5%
        
        # Update ADT
        detector.threshold_manager.update_from_feedback(entity_id, batch, alert_rate)
        
        stats['entities_updated'] += 1
        stats['precision_by_entity'][entity_id] = precision
        
        print(f"  Updated {entity_id}: {len(batch)} reviews, precision={precision:.1%}")
    
    return stats


def main():
    """Main script for processing feedback"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process human feedback for ADT training')
    parser.add_argument('--feedback', type=str, required=True,
                       help='Path to feedback CSV file')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model checkpoint (default: auto-detect)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save updated model (default: overwrite input)')
    parser.add_argument('--enable-adt', action='store_true',
                       help='Enable ADT if not already enabled')
    
    args = parser.parse_args()
    
    print("="*80)
    print("PROCESS HUMAN FEEDBACK FOR ADT")
    print("="*80)
    
    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = data_dir / 'models' / 'federated_optimized_rtx4070.pth'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = model_path  # Overwrite
    
    feedback_path = Path(args.feedback)
    if not feedback_path.exists():
        raise FileNotFoundError(f"Feedback file not found: {feedback_path}")
    
    # Load feedback
    print("\n" + "="*80)
    print("LOADING FEEDBACK")
    print("="*80)
    feedback = load_feedback_csv(feedback_path)
    
    # Load model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")
    print(f"Loading: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    input_dim = checkpoint['model_state_dict']['encoder.0.weight'].shape[1]
    
    # Extract latent_dim from checkpoint (either from config or infer from weights)
    if 'latent_dim' in checkpoint:
        latent_dim = checkpoint['latent_dim']
    else:
        # Infer from encoder final layer output shape
        final_encoder_weight = None
        max_layer_num = -1
        for key in checkpoint['model_state_dict'].keys():
            if 'encoder' in key and 'weight' in key:
                try:
                    # BUG FIX #59: Safe layer number extraction with error handling
                    parts = key.split('.')
                    if len(parts) >= 2 and parts[1].isdigit():
                        layer_num = int(parts[1])
                        if layer_num > max_layer_num:
                            max_layer_num = layer_num
                            final_encoder_weight = key
                except (ValueError, IndexError):
                    continue  # Skip malformed keys
        
        if final_encoder_weight:
            latent_dim = checkpoint['model_state_dict'][final_encoder_weight].shape[0]
        else:
            latent_dim = 32  # Fallback default
    
    # Extract architecture (default to 'deep' if not present)
    architecture = checkpoint.get('architecture', 'deep')
    
    print(f"Checkpoint config: latent_dim={latent_dim}, architecture={architecture}")
    
    model = GLAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        architecture=architecture
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create detector with threshold manager
    detector = AnomalyDetector(model, categorical_indices=[], numerical_indices=[])
    
    # Load threshold manager (with correct device for ADT)
    if 'threshold_manager_state' in checkpoint:
        detector.threshold_manager = ThresholdManager.from_state(
            checkpoint['threshold_manager_state'],
            device=str(DEVICE)
        )
        print(f"Loaded SPOT thresholds for {len(detector.threshold_manager.entity_spots)} entities")
    else:
        raise ValueError("Model checkpoint missing threshold_manager_state!")
    
    # Enable ADT if requested
    if args.enable_adt and not detector.threshold_manager.enable_adt:
        print("\n[INFO] Enabling ADT...")
        detector.threshold_manager.enable_adt_learning(device=str(DEVICE))
    
    if not detector.threshold_manager.enable_adt:
        print("\n[WARNING] ADT not enabled! Use --enable-adt flag to enable.")
        return
    
    # Update ADT from feedback
    stats = update_adt_from_feedback(detector, feedback)
    
    # Save updated model
    print("\n" + "="*80)
    print("SAVING UPDATED MODEL")
    print("="*80)
    
    checkpoint['threshold_manager_state'] = detector.threshold_manager.get_state()
    checkpoint['last_feedback_update'] = datetime.now().isoformat()
    checkpoint['feedback_stats'] = stats
    
    # BUG FIX #58: Save to temporary file first to prevent corruption if save fails
    import tempfile
    import shutil
    
    temp_fd, temp_path = tempfile.mkstemp(suffix='.pth', dir=output_path.parent)
    try:
        import os
        os.close(temp_fd)
        torch.save(checkpoint, temp_path)
        shutil.move(temp_path, output_path)
        print(f"Model saved to: {output_path}")
    except Exception as e:
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        raise RuntimeError(f"Failed to save checkpoint: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("UPDATE COMPLETE!")
    print("="*80)
    print(f"\nFeedback processed: {stats['total_feedback']} records")
    print(f"Entities updated: {stats['entities_updated']}")
    print(f"\nPrecision by entity:")
    for entity, precision in sorted(stats['precision_by_entity'].items(), key=lambda x: x[1]):
        print(f"  {entity}: {precision:.1%}")
    
    avg_precision = np.mean(list(stats['precision_by_entity'].values()))
    print(f"\nAverage precision: {avg_precision:.1%}")


if __name__ == "__main__":
    main()

