"""
Simulate Human Feedback Training for ADT
Generates synthetic labels for demo/training ADT before real human feedback is available
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from src.models.autoencoder import GLAutoencoder, AnomalyDetector
from src.data.preprocessing import GLDataPreprocessor


def simulate_human_labels(anomalies_df, threshold_col='threshold_used'):
    """
    Simulate human labels based on error-to-threshold ratio heuristics.
    
    Heuristic rules:
    - error > 3x threshold → likely CORRECT (90% chance)
    - error < 1.5x threshold → likely FALSE_POSITIVE (80% chance)
    - 1.5x-3x threshold → uncertain (50/50)
    
    Args:
        anomalies_df: DataFrame with anomalies (must have 'reconstruction_error' and threshold column)
        threshold_col: Column name for threshold values
    
    Returns:
        DataFrame with columns: anomaly_id, entity_id, is_correct, confidence, ratio
    """
    labels = []
    
    for idx, row in anomalies_df.iterrows():
        error = row['reconstruction_error']
        threshold = row.get(threshold_col, row.get('q_alpha', 0.1))  # Fallback
        
        if threshold == 0:
            threshold = 0.1  # Prevent division by zero
        
        ratio = error / threshold
        
        # Heuristic labeling based on ratio
        if ratio > 3.0:
            # Very high error → likely genuine anomaly
            is_correct = np.random.random() < 0.90
            confidence = 0.9
        elif ratio < 1.5:
            # Barely above threshold → likely false positive
            is_correct = np.random.random() < 0.20
            confidence = 0.8
        else:
            # Uncertain zone
            is_correct = np.random.random() < 0.50
            confidence = 0.5
        
        labels.append({
            'anomaly_id': f"ANO_{idx:06d}",
            'entity_id': row['entity_id'],
            'is_correct': is_correct,
            'confidence': confidence,
            'ratio': ratio,
            'simulated': True
        })
    
    return pd.DataFrame(labels)


def train_adt_with_simulation(detector, validation_data, entity_ids, validation_df, n_episodes=50, device='cpu'):
    """
    Train ADT with simulated human feedback for n episodes.
    
    Args:
        detector: AnomalyDetector instance with threshold_manager
        validation_data: Tensor of validation features
        entity_ids: Array of entity IDs for validation data
        validation_df: Original dataframe (for GL details)
        n_episodes: Number of training episodes
        device: 'cpu' or 'cuda'
    
    Returns:
        Dictionary with training history
    """
    print("\n" + "="*80)
    print("ADT TRAINING WITH SIMULATED FEEDBACK")
    print("="*80)
    print(f"Episodes: {n_episodes}")
    print(f"Device: {device}")
    print(f"Validation samples: {len(validation_data):,}")
    
    # Enable ADT
    if not detector.threshold_manager.enable_adt:
        detector.threshold_manager.enable_adt_learning(device=device)
    
    # Track metrics
    history = {
        'episode': [],
        'total_anomalies': [],
        'precision': [],
        'alert_rate': [],
        'mean_delta': [],
        'mean_epsilon': []
    }
    
    # Training loop
    for episode in range(1, n_episodes + 1):
        print(f"\n{'='*80}")
        print(f"EPISODE {episode}/{n_episodes}")
        print(f"{'='*80}")
        
        # Detect anomalies with current ADT policy
        predictions, _ = detector.predict_with_entity(validation_data, entity_ids)
        errors = detector.compute_reconstruction_errors(validation_data)
        
        # Extract anomalies
        anomaly_mask = predictions == 1
        if anomaly_mask.sum() == 0:
            print("[WARNING] No anomalies detected! Skipping episode...")
            continue
        
        anomalies_df = validation_df[anomaly_mask].copy()
        anomalies_df.loc[:, 'reconstruction_error'] = errors[anomaly_mask]
        
        # Add threshold info (vectorized to avoid SettingWithCopyWarning)
        threshold_values = []
        for idx, row in anomalies_df.iterrows():
            entity = str(row['entity_id'])  # entity_id is now consistent (no AGY_ prefix)
            if entity in detector.threshold_manager.entity_spots:
                spot = detector.threshold_manager.entity_spots[entity]
                threshold_values.append(spot.get_threshold())
            else:
                threshold_values.append(detector.threshold_manager.global_threshold)
        
        anomalies_df.loc[:, 'q_alpha'] = threshold_values
        
        # Simulate human feedback
        feedback = simulate_human_labels(anomalies_df, threshold_col='q_alpha')
        
        # Update ADT per entity
        for entity_id in feedback['entity_id'].unique():
            entity_feedback = feedback[feedback['entity_id'] == entity_id]
            
            feedback_batch = [
                {'is_correct': row['is_correct'], 'anomaly_id': row['anomaly_id']}
                for _, row in entity_feedback.iterrows()
            ]
            
            # Calculate alert rate for this entity
            entity_mask = entity_ids == entity_id
            entity_alert_rate = predictions[entity_mask].mean() if entity_mask.sum() > 0 else 0.0
            
            # Update ADT
            detector.threshold_manager.update_from_feedback(
                entity_id,
                feedback_batch,
                alert_rate=float(entity_alert_rate)
            )
        
        # Calculate episode metrics
        precision = feedback['is_correct'].mean()
        alert_rate = predictions.mean()
        
        # Get mean ADT delta across entities (with NaN handling)
        deltas = []
        epsilons = []
        for entity, controller in detector.threshold_manager.adt_controllers.items():
            # Only add if finite (skip NaN/Inf)
            if np.isfinite(controller.current_delta):
                deltas.append(controller.current_delta)
            if np.isfinite(controller.epsilon):
                epsilons.append(controller.epsilon)
        
        mean_delta = np.mean(deltas) if len(deltas) > 0 else 0.0
        mean_epsilon = np.mean(epsilons) if len(epsilons) > 0 else 0.1
        
        # Record history
        history['episode'].append(episode)
        history['total_anomalies'].append(int(predictions.sum()))
        history['precision'].append(precision)
        history['alert_rate'].append(alert_rate)
        history['mean_delta'].append(mean_delta)
        history['mean_epsilon'].append(mean_epsilon)
        
        # Print episode summary
        print(f"\nEpisode {episode} Results:")
        print(f"  Anomalies detected: {int(predictions.sum()):,}")
        print(f"  Simulated precision: {precision:.1%}")
        print(f"  Alert rate: {alert_rate:.2%}")
        print(f"  Mean ADT delta: {mean_delta:+.4f}")
        print(f"  Mean exploration (ε): {mean_epsilon:.4f}")
        print(f"  Entities with ADT: {len(detector.threshold_manager.adt_controllers)}")
    
    return history


def plot_training_history(history, output_dir):
    """
    Plot ADT training curves.
    
    Args:
        history: Dictionary from train_adt_with_simulation()
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ADT Training Progress (Simulated Feedback)', fontsize=16)
    
    # Plot 1: Precision over episodes
    axes[0, 0].plot(history['episode'], history['precision'], 'b-', linewidth=2)
    axes[0, 0].axhline(y=0.9, color='g', linestyle='--', label='Target (90%)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision Improvement')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Alert rate over episodes
    axes[0, 1].plot(history['episode'], history['alert_rate'], 'r-', linewidth=2)
    axes[0, 1].axhline(y=0.05, color='g', linestyle='--', label='Target (5%)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Alert Rate')
    axes[0, 1].set_title('Alert Volume Control')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: ADT delta adjustment over episodes
    axes[1, 0].plot(history['episode'], history['mean_delta'], 'purple', linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Mean Threshold Adjustment (δ)')
    axes[1, 0].set_title('ADT Threshold Learning')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Exploration rate (epsilon) over episodes
    axes[1, 1].plot(history['episode'], history['mean_epsilon'], 'orange', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Mean Exploration (ε)')
    axes[1, 1].set_title('Exploration Decay')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'adt_training_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n[PLOT] Training curves saved to: {plot_path}")
    plt.close()


def main():
    """Main training script"""
    print("="*80)
    print("ADT SIMULATED FEEDBACK TRAINING")
    print("="*80)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {DEVICE}")
    
    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    model_path = data_dir / 'models' / 'federated_optimized_rtx4070.pth'
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    raw_files = list((data_dir / 'raw').glob('*.csv'))
    if not raw_files:
        raise FileNotFoundError(f"No data files found in {data_dir / 'raw'}")
    
    data_file = str(raw_files[0])
    print(f"Loading: {data_file}")
    df = pd.read_csv(data_file, low_memory=False)
    print(f"Loaded {len(df):,} transactions")
    
    # Preprocess (same as detect_anomalies.py)
    print("\n" + "="*80)
    print("PREPROCESSING")
    print("="*80)
    
    df['net_amount'] = pd.to_numeric(df['POSTED_TOTAL_AMT'], errors='coerce').fillna(0.0)
    df['debit_amount'] = df['net_amount'].apply(lambda x: max(0, x))
    df['credit_amount'] = df['net_amount'].apply(lambda x: max(0, -x))
    df['debit_credit_ratio'] = df['debit_amount'] / (df['credit_amount'] + 1e-6)
    df['debit_credit_ratio'] = df['debit_credit_ratio'].clip(upper=1e6)
    df['net_balance'] = df['debit_amount'] - df['credit_amount']
    df['abs_balance'] = np.abs(df['net_balance'])
    df['gl_account'] = df['ACCOUNT'].astype(str)
    df['cost_center'] = df['CLASS_FLD'].astype(str).fillna('UNKNOWN')
    df['profit_center'] = df['FUND_CODE'].astype(str)
    df['entity_id'] = df['AGENCYNBR'].astype(str)  # Keep consistent with finalize_trained_model.py (no AGY_ prefix)
    df['entity_name'] = df['AGENCYNAME'].astype(str)
    
    categorical_cols = ['gl_account', 'cost_center', 'profit_center']
    numerical_cols = ['debit_amount', 'credit_amount', 'debit_credit_ratio', 'net_balance', 'abs_balance']
    
    preprocessor = GLDataPreprocessor(categorical_cols, numerical_cols)
    features = preprocessor.fit_transform(df)
    
    print(f"Feature shape: {features.shape}")
    
    # Take subset for faster training (20% of data)
    sample_size = int(len(df) * 0.2)
    indices = np.random.choice(len(df), sample_size, replace=False)
    df_sample = df.iloc[indices].reset_index(drop=True)
    features_sample = features[indices]
    
    print(f"Training on {len(df_sample):,} samples (20% of full dataset)")
    
    # Load model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    
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
    
    detector = AnomalyDetector(
        model,
        categorical_indices=preprocessor.categorical_indices,
        numerical_indices=preprocessor.numerical_indices
    )
    
    # Load SPOT thresholds (with correct device for ADT)
    if 'threshold_manager_state' in checkpoint:
        from src.models.threshold_manager import ThresholdManager
        detector.threshold_manager = ThresholdManager.from_state(
            checkpoint['threshold_manager_state'],
            device=str(DEVICE)
        )
        print(f"Loaded SPOT thresholds for {len(detector.threshold_manager.entity_spots)} entities")
    else:
        raise ValueError("Model checkpoint missing threshold_manager_state! Run finalize_trained_model.py first.")
    
    # Train ADT
    X_tensor = torch.FloatTensor(features_sample.astype(np.float32)).to(DEVICE)
    entity_ids = df_sample['AGENCYNBR'].astype(str).values  # Must match finalize_trained_model.py format
    
    history = train_adt_with_simulation(
        detector,
        X_tensor,
        entity_ids,
        df_sample,
        n_episodes=50,
        device=str(DEVICE)
    )
    
    # Plot results
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    plot_training_history(history, data_dir / 'results')
    
    # Save trained model with ADT
    print("\n" + "="*80)
    print("SAVING TRAINED ADT MODEL")
    print("="*80)
    
    checkpoint['threshold_manager_state'] = detector.threshold_manager.get_state()
    output_path = data_dir / 'models' / 'federated_optimized_rtx4070_adt.pth'
    
    # BUG FIX #58: Save to temporary file first to prevent corruption if save fails
    import tempfile
    import shutil
    
    temp_fd, temp_path = tempfile.mkstemp(suffix='.pth', dir=output_path.parent)
    try:
        import os
        os.close(temp_fd)
        torch.save(checkpoint, temp_path)
        shutil.move(temp_path, output_path)
        print(f"Model with trained ADT saved to: {output_path}")
    except Exception as e:
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        raise RuntimeError(f"Failed to save checkpoint: {e}")
    
    # Print final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal Results (Episode {len(history['episode'])}):")
    print(f"  Precision: {history['precision'][-1]:.1%}")
    print(f"  Alert rate: {history['alert_rate'][-1]:.2%}")
    print(f"  Mean ADT delta: {history['mean_delta'][-1]:+.4f}")
    print(f"  Improvement from start:")
    print(f"    Precision: {history['precision'][0]:.1%} → {history['precision'][-1]:.1%} "
          f"({(history['precision'][-1] - history['precision'][0])*100:+.1f}%)")
    print(f"    Alert rate: {history['alert_rate'][0]:.2%} → {history['alert_rate'][-1]:.2%} "
          f"({(history['alert_rate'][-1] - history['alert_rate'][0])*100:+.1f}%)")


if __name__ == "__main__":
    main()

