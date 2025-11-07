"""
Finalize Already-Trained Model
Loads best_model_optuna.pth and completes threshold calculation and ONNX export
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import json
from datetime import datetime
from pathlib import Path

from src.models.autoencoder import GLAutoencoder, AnomalyDetector
from src.data.preprocessing import GLDataPreprocessor

print("=" * 80)
print("FINALIZING TRAINED MODEL")
print("=" * 80)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {DEVICE}")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

script_dir = Path(__file__).parent
data_dir = script_dir.parent / 'data'

# Load raw data
raw_files = list((data_dir / 'raw').glob('*.csv'))
if not raw_files:
    raise FileNotFoundError(f"No data files found in {data_dir / 'raw'}")

data_file = str(raw_files[0])
print(f"Loading: {data_file}")
df = pd.read_csv(data_file, low_memory=False)
print(f"Loaded {len(df):,} transactions")

# ============================================================================
# PREPROCESS
# ============================================================================
print("\n" + "=" * 80)
print("PREPROCESSING")
print("=" * 80)

# Create features (same as training)
df['net_amount'] = pd.to_numeric(df['POSTED_TOTAL_AMT'], errors='coerce').fillna(0.0)
df['debit_amount'] = df['net_amount'].apply(lambda x: max(0, x))
df['credit_amount'] = df['net_amount'].apply(lambda x: max(0, -x))

# Compute debit/credit ratio with clipping to prevent extreme values
df['debit_credit_ratio'] = df['debit_amount'] / (df['credit_amount'] + 1e-6)
df['debit_credit_ratio'] = df['debit_credit_ratio'].clip(upper=1e6)  # Cap at 1M to prevent overflow

df['net_balance'] = df['debit_amount'] - df['credit_amount']
df['abs_balance'] = np.abs(df['net_balance'])

# Check for inf/nan values that could corrupt the model
if df[['debit_amount', 'credit_amount', 'debit_credit_ratio', 'net_balance', 'abs_balance']].isnull().any().any():
    print("\n[WARNING] NaN values detected! Filling with 0...")
    df.fillna(0.0, inplace=True)

if np.isinf(df[['debit_amount', 'credit_amount', 'debit_credit_ratio', 'net_balance', 'abs_balance']]).any().any():
    print("\n[WARNING] Inf values detected! Clipping...")
    df['debit_credit_ratio'] = df['debit_credit_ratio'].replace([np.inf, -np.inf], 1e6)

df['gl_account'] = df['ACCOUNT'].astype(str)
df['cost_center'] = df['CLASS_FLD'].astype(str).fillna('UNKNOWN')
df['profit_center'] = df['FUND_CODE'].astype(str)

categorical_cols = ['gl_account', 'cost_center', 'profit_center']
numerical_cols = ['debit_amount', 'credit_amount', 'debit_credit_ratio',
                 'net_balance', 'abs_balance']

preprocessor = GLDataPreprocessor(categorical_cols, numerical_cols)
features = preprocessor.fit_transform(df)

print(f"Feature shape: {features.shape}")

# Train/val split (sequential)
n_samples = len(features)
train_size = int(0.8 * n_samples)
X_val = features[train_size:]

print(f"Validation samples: {len(X_val):,}")

# ============================================================================
# LOAD TRAINED MODEL
# ============================================================================
print("\n" + "=" * 80)
print("LOADING TRAINED MODEL")
print("=" * 80)

# Try to load best_model_optuna.pth first, fallback to production model
best_model_path = data_dir / 'models' / 'best_model_optuna.pth'
production_model_path = data_dir / 'models' / 'federated_optimized_rtx4070.pth'

if best_model_path.exists():
    model_path = best_model_path
elif production_model_path.exists():
    model_path = production_model_path
    print(f"[INFO] Using existing production model: {model_path.name}")
else:
    raise FileNotFoundError(f"No trained model found. Please train the model first.\n"
                          f"  Expected: {best_model_path}\n"
                          f"  Or: {production_model_path}")

checkpoint = torch.load(str(model_path), map_location=DEVICE, weights_only=False)
if 'epoch' in checkpoint:
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"Validation loss: {checkpoint['val_loss']:.4f}")

# Recreate model with same architecture
input_dim = features.shape[1]

# Extract config (handle both training and production checkpoints)
if 'model_config' in checkpoint:
    config = checkpoint['model_config']
    latent_dim = config.get('latent_dim', 2)
    architecture = config.get('architecture', 'deep')
else:
    latent_dim = 2  # Paper config default
    architecture = 'deep'

model = GLAutoencoder(
    input_dim=input_dim,
    latent_dim=latent_dim,
    architecture=architecture
).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded successfully")
print(f"Input dim: {input_dim}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# CALCULATE THRESHOLD (SPOT Adaptive Per-Entity)
# ============================================================================
print("\n" + "=" * 80)
print("CALCULATING ADAPTIVE THRESHOLDS (SPOT + ADWIN)")
print("=" * 80)

detector = AnomalyDetector(
    model=model,
    categorical_indices=preprocessor.categorical_indices,
    numerical_indices=preprocessor.numerical_indices,
    threshold_percentile=95.0,  # Fallback only
    theta=2/3
)

X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)

# Extract entity IDs for per-entity thresholding
# Force string type to ensure consistent key matching (101.0 → "101.0" vs 101 → "101")
entity_ids_val = df['AGENCYNBR'].astype(str).values[train_size:]

# Fit SPOT adaptive thresholds per entity
detector.fit_threshold_spot(X_val_tensor, entity_ids_val)

# Test on validation set with SPOT
predictions, drift_flags = detector.predict_with_entity(X_val_tensor, entity_ids_val)
anomaly_rate = float(predictions.sum()) / len(predictions) * 100  # Explicit float to prevent overflow

# Get errors for stats
errors = detector.compute_reconstruction_errors(X_val_tensor)

print(f"\n" + "="*80)
print(f"SPOT THRESHOLD SUMMARY")
print(f"="*80)
summary = detector.threshold_manager.get_entity_summary()
print(f"Total entities: {summary['n_entities']}")
print(f"Anomaly rate: {anomaly_rate:.2f}%")
if len(errors[predictions == 0]) > 0:
    print(f"Mean error (normal): {errors[predictions == 0].mean():.4f}")
if len(errors[predictions == 1]) > 0:
    print(f"Mean error (anomaly): {errors[predictions == 1].mean():.4f}")
print(f"Drift flags: {drift_flags.sum()} samples")
print(f"="*80)

# ============================================================================
# SAVE PRODUCTION MODEL
# ============================================================================
print("\n" + "=" * 80)
print("SAVING PRODUCTION MODEL")
print("=" * 80)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
models_dir = data_dir / 'models'

# Save complete model with SPOT threshold manager state
production_path = str(models_dir / 'federated_optimized_rtx4070.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'preprocessor_state': {
        'categorical_columns': preprocessor.categorical_columns,
        'numerical_columns': preprocessor.numerical_columns,
        'categorical_indices': preprocessor.categorical_indices,
        'numerical_indices': preprocessor.numerical_indices,
    },
    'model_config': {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'architecture': architecture,
    },
    'threshold': detector.threshold,  # Legacy fallback
    'threshold_manager_state': detector.threshold_manager.get_state(),  # SPOT state
    'val_loss': checkpoint['val_loss'],
    'anomaly_rate': anomaly_rate,
    'device': 'RTX 4070 Laptop',
    'timestamp': timestamp,
}, production_path)

print(f"Production model saved: {production_path}")
print(f"  Includes SPOT adaptive thresholds for {summary['n_entities']} entities")

# ============================================================================
# EXPORT TO ONNX
# ============================================================================
print("\n" + "=" * 80)
print("EXPORTING TO ONNX")
print("=" * 80)

try:
    model.eval()
    dummy_input = torch.randn(1, input_dim).to(DEVICE)
    onnx_path = str(models_dir / f'federated_optimized_{timestamp}.onnx')
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX model saved: {onnx_path}")
    print("  [OK] Ready for production deployment")
except Exception as e:
    print(f"  [WARNING] ONNX export failed: {e}")
    onnx_path = "N/A"

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINALIZATION COMPLETE!")
print("=" * 80)

print("\nProduction Model:")
print(f"  Path: {production_path}")
print(f"  Validation Loss: {checkpoint['val_loss']:.4f}")
print(f"  Anomaly Rate: {anomaly_rate:.2f}% (SPOT adaptive)")
print(f"  Entities with SPOT: {summary['n_entities']}")
if summary['n_entities'] > 0:
    thresholds_list = list(summary['thresholds'].values())
    print(f"  Threshold range: {min(thresholds_list):.4f} - {max(thresholds_list):.4f}")

print("\nConfiguration:")
print(f"  Architecture: {architecture} (paper config)")
print(f"  Latent Dim: {latent_dim}")
print(f"  Input Features: {input_dim}")
print(f"  Device: RTX 4070 Laptop GPU")

print("\n" + "=" * 80)

