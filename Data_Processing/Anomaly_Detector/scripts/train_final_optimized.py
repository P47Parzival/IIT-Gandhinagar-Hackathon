"""
Final Training with Optuna-Optimized Parameters
Uses best parameters from Optuna study
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import json
from datetime import datetime

from src.models.autoencoder import GLAutoencoder, combined_loss, AnomalyDetector
from src.data.preprocessing import GLDataPreprocessor

print("=" * 80)
print("FINAL TRAINING - Optuna-Optimized Parameters")
print("=" * 80)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

# Try to load processed data, fallback to raw data
import glob
from pathlib import Path

# Get script directory and find data files
script_dir = Path(__file__).parent
data_dir = script_dir.parent / 'data'

processed_files = list((data_dir / 'processed').glob('*.csv'))
raw_files = list((data_dir / 'raw').glob('*.csv'))

if processed_files:
    data_file = str(processed_files[0])
    print(f"Loading processed data: {data_file}")
elif raw_files:
    data_file = str(raw_files[0])
    print(f"Loading raw data: {data_file}")
else:
    raise FileNotFoundError(f"No data files found in {data_dir / 'processed'} or {data_dir / 'raw'}")

df = pd.read_csv(data_file, low_memory=False)
print(f"Loaded {len(df):,} transactions from {data_file}")
print(f"Available columns: {list(df.columns)[:10]}...")

# ============================================================================
# PREPROCESS OKLAHOMA LEDGER DATA
# ============================================================================
print("\n" + "=" * 80)
print("PREPROCESSING OKLAHOMA LEDGER DATA")
print("=" * 80)

# Map Oklahoma ledger columns to our model's expected format
print("Mapping columns and creating features...")

# Create derived numerical features (FIXED to match Optuna preprocessing)
df['net_amount'] = pd.to_numeric(df['POSTED_TOTAL_AMT'], errors='coerce').fillna(0.0)
df['debit_amount'] = df['net_amount'].apply(lambda x: max(0, x))
df['credit_amount'] = df['net_amount'].apply(lambda x: max(0, -x))  # FIXED: Use max(0, -x) not abs(min(0, x))
df['debit_credit_ratio'] = df['debit_amount'] / (df['credit_amount'] + 1e-6)  # FIXED: Use 1e-6 not 1e-10
df['net_balance'] = df['debit_amount'] - df['credit_amount']  # FIXED: debit - credit, not just net_amount
df['abs_balance'] = np.abs(df['net_balance'])  # FIXED: abs of net_balance

# Map categorical columns (FIXED to match Optuna columns)
df['gl_account'] = df['ACCOUNT'].astype(str)
df['cost_center'] = df['CLASS_FLD'].astype(str).fillna('UNKNOWN')  # FIXED: Use CLASS_FLD (349 values) to match Optuna!
df['profit_center'] = df['FUND_CODE'].astype(str)

print(f"  Created debit_amount: {df['debit_amount'].describe()}")
print(f"  Created credit_amount: {df['credit_amount'].describe()}")
print(f"  Mapped gl_account: {df['gl_account'].nunique()} unique values")
print(f"  Mapped cost_center: {df['cost_center'].nunique()} unique values")
print(f"  Mapped profit_center: {df['profit_center'].nunique()} unique values")

# Now use standard preprocessing
categorical_cols = ['gl_account', 'cost_center', 'profit_center']
numerical_cols = ['debit_amount', 'credit_amount', 'debit_credit_ratio',
                 'net_balance', 'abs_balance']

preprocessor = GLDataPreprocessor(categorical_cols, numerical_cols)
features = preprocessor.fit_transform(df)

print(f"\nFeature shape: {features.shape}")
print(f"Categorical features: {len(preprocessor.categorical_indices)}")
print(f"Numerical features: {len(preprocessor.numerical_indices)}")

# ============================================================================
# OPTUNA BEST PARAMETERS (from study results)
# ============================================================================
print("\n" + "=" * 80)
print("OPTUNA BEST PARAMETERS")
print("=" * 80)

# Best parameters from Optuna study
BEST_PARAMS = {
    'config_type': 'paper',
    'batch_size': 16,
    'learning_rate': 0.001,
    'latent_dim': 2,
    'architecture': 'deep',
    'weight_decay': 0.0,  # FIXED: Paper doesn't use weight decay (was 1e-5)
    'n_epochs': 50,
    'patience': 10,
    'val_loss': 0.003460547757318472,  # Exact from Optuna
    'anomaly_rate': 5.00,
    'threshold': 0.009420202160254031  # Exact from Optuna
}

print("Paper Config (Best):")
for key, val in BEST_PARAMS.items():
    print(f"  {key}: {val}")

# ============================================================================
# TRAIN/VAL SPLIT
# ============================================================================
# CRITICAL FIX: Use sequential split (not random) to match Optuna training
n_samples = len(features)
train_size = int(0.8 * n_samples)

# Sequential split (matches Optuna)
X_train = features[:train_size]
X_val = features[train_size:]

print(f"\nTrain samples: {len(X_train):,}")
print(f"Val samples: {len(X_val):,}")

# Create DataLoaders
train_dataset = TensorDataset(torch.FloatTensor(X_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val))

train_loader = DataLoader(train_dataset, batch_size=BEST_PARAMS['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BEST_PARAMS['batch_size'], shuffle=False)

# ============================================================================
# BUILD MODEL
# ============================================================================
print("\n" + "=" * 80)
print("BUILDING MODEL")
print("=" * 80)

input_dim = features.shape[1]
model = GLAutoencoder(
    input_dim=input_dim,
    latent_dim=BEST_PARAMS['latent_dim'],
    architecture=BEST_PARAMS['architecture']
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nArchitecture: {BEST_PARAMS['architecture']}")
print(f"Input dim: {input_dim}")
print(f"Latent dim: {BEST_PARAMS['latent_dim']}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# TRAINING SETUP
# ============================================================================
# FIXED: Use AdamW (not Adam) to match Optuna training
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=BEST_PARAMS['learning_rate'],
    weight_decay=BEST_PARAMS['weight_decay'],
    betas=(0.9, 0.999)  # Paper's beta values
)

# FIXED: Add min_lr to match Optuna training
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
    verbose=False,  # Changed to False to reduce output noise
    min_lr=1e-6  # CRITICAL: Prevents LR from going to 0
)

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

print("\nProgress:")
for epoch in range(BEST_PARAMS['n_epochs']):
    # Training
    model.train()
    epoch_train_losses = []
    for batch_x, in train_loader:
        batch_x = batch_x.to(DEVICE)
        optimizer.zero_grad()
        reconstructed = model(batch_x)
        loss = combined_loss(
            batch_x, reconstructed,
            preprocessor.categorical_indices,
            preprocessor.numerical_indices,
            theta=2/3
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_train_losses.append(loss.item())
    
    train_loss = np.mean(epoch_train_losses)
    train_losses.append(train_loss)
    
    # Validation
    model.eval()
    epoch_val_losses = []
    with torch.no_grad():
        for batch_x, in val_loader:
            batch_x = batch_x.to(DEVICE)
            reconstructed = model(batch_x)
            loss = combined_loss(
                batch_x, reconstructed,
                preprocessor.categorical_indices,
                preprocessor.numerical_indices,
                theta=2/3
            )
            epoch_val_losses.append(loss.item())
    
    val_loss = np.mean(epoch_val_losses)
    val_losses.append(val_loss)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Track best model
    status = ""
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        status = " ** BEST"
        
        # Save best model
        best_model_path = data_dir / 'models' / 'best_model_optuna.pth'
        best_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
        }, str(best_model_path))
    else:
        patience_counter += 1
    
    # Print progress every 5 epochs or if best
    if (epoch + 1) % 5 == 0 or status:
        print(f"  Epoch {epoch+1:2d}/{BEST_PARAMS['n_epochs']} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}{status}")
    
    # Early stopping
    if patience_counter >= BEST_PARAMS['patience']:
        print(f"  [EARLY STOP] Stopped at epoch {epoch+1} (patience={BEST_PARAMS['patience']})")
        break

print(f"\nBest validation loss: {best_val_loss:.4f}")

# ============================================================================
# CALCULATE THRESHOLD
# ============================================================================
print("\n" + "=" * 80)
print("CALCULATING ANOMALY THRESHOLD")
print("=" * 80)

# Load best model
best_model_path = data_dir / 'models' / 'best_model_optuna.pth'
checkpoint = torch.load(str(best_model_path), weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

# Create detector
detector = AnomalyDetector(
    model=model,
    categorical_indices=preprocessor.categorical_indices,
    numerical_indices=preprocessor.numerical_indices,
    threshold_percentile=95.0,
    theta=2/3
)

# Calculate threshold on validation set
X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
detector.fit_threshold(X_val_tensor)

print(f"\nAnomaly threshold: {detector.threshold:.4f}")
print(f"Expected anomaly rate: ~5%")

# Test on validation set
errors, predictions = detector.detect_anomalies(X_val_tensor)
anomaly_rate = predictions.sum() / len(predictions) * 100

print(f"Actual anomaly rate: {anomaly_rate:.2f}%")
print(f"Mean error (normal): {errors[predictions == 0].mean():.4f}")
print(f"Mean error (anomaly): {errors[predictions == 1].mean():.4f}")

# ============================================================================
# SAVE FINAL MODEL
# ============================================================================
print("\n" + "=" * 80)
print("SAVING FINAL MODEL")
print("=" * 80)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save model weights
models_dir = data_dir / 'models'
models_dir.mkdir(parents=True, exist_ok=True)
model_path = str(models_dir / f'final_model_optuna_{timestamp}.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'preprocessor_state': {
        'categorical_columns': preprocessor.categorical_columns,
        'numerical_columns': preprocessor.numerical_columns,
        'categorical_indices': preprocessor.categorical_indices,
        'numerical_indices': preprocessor.numerical_indices,
    },
    'best_params': BEST_PARAMS,
    'threshold': detector.threshold,
    'val_loss': best_val_loss,
    'anomaly_rate': anomaly_rate,
    'training_history': {
        'train_losses': train_losses,
        'val_losses': val_losses,
    }
}, model_path)

print(f"Model saved: {model_path}")

# Save training summary
summary = {
    'timestamp': timestamp,
    'best_params': BEST_PARAMS,
    'final_metrics': {
        'best_val_loss': float(best_val_loss),
        'final_epoch': len(train_losses),
        'threshold': float(detector.threshold),
        'anomaly_rate': float(anomaly_rate),
    },
    'dataset_info': {
        'total_samples': n_samples,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'features': input_dim,
    },
    'model_info': {
        'architecture': BEST_PARAMS['architecture'],
        'latent_dim': BEST_PARAMS['latent_dim'],
        'total_parameters': total_params,
    }
}

summary_path = str(models_dir / f'training_summary_optuna_{timestamp}.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved: {summary_path}")

# ============================================================================
# EXPORT TO ONNX (for production deployment)
# ============================================================================
print("\n" + "=" * 80)
print("EXPORTING TO ONNX")
print("=" * 80)

try:
    model.eval()
    dummy_input = torch.randn(1, input_dim).to(DEVICE)
    onnx_path = str(models_dir / f'final_model_optuna_{timestamp}.onnx')
    
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

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)

print("\nFinal Results:")
print(f"  Best Val Loss: {best_val_loss:.4f}")
print(f"  Anomaly Rate: {anomaly_rate:.2f}%")
print(f"  Threshold: {detector.threshold:.4f}")
print(f"  Total Epochs: {len(train_losses)}")

print("\nBest Configuration:")
print(f"  Architecture: {BEST_PARAMS['architecture']}")
print(f"  Batch Size: {BEST_PARAMS['batch_size']}")
print(f"  Learning Rate: {BEST_PARAMS['learning_rate']}")
print(f"  Latent Dim: {BEST_PARAMS['latent_dim']}")
print(f"  Weight Decay: {BEST_PARAMS['weight_decay']}")

print("\nModel Files:")
print(f"  PyTorch: {model_path}")
print(f"  ONNX: {onnx_path}")
print(f"  Summary: {summary_path}")

print("\n" + "=" * 80)
print("Next Steps:")
print("  1. Use this model for inference")
print("  2. Deploy ONNX model to production")
print("  3. Update model_path in run_complete_pipeline.py")
print("=" * 80)

