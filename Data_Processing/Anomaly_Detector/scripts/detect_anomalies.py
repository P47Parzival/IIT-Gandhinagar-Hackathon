"""
Detect Anomalies Using Finalized Model
Loads the production model and displays top anomalies grouped by entity
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime

from src.models.autoencoder import GLAutoencoder, AnomalyDetector
from src.data.preprocessing import GLDataPreprocessor

print("=" * 80)
print("ANOMALY DETECTION - PRODUCTION MODEL")
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

# Validate required columns exist
required_cols = ['POSTED_TOTAL_AMT', 'ACCOUNT', 'CLASS_FLD', 'FUND_CODE', 'AGENCYNBR', 'AGENCYNAME']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"\n[ERROR] Missing required columns: {missing_cols}")
    print(f"Available columns: {list(df.columns)[:10]}...")
    raise ValueError(f"Dataset missing required columns: {missing_cols}")

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
    print("\n[WARNING] NaN values detected in numerical features! Filling with 0...")
    df.fillna(0.0, inplace=True)

if np.isinf(df[['debit_amount', 'credit_amount', 'debit_credit_ratio', 'net_balance', 'abs_balance']]).any().any():
    print("\n[WARNING] Inf values detected in numerical features! Clipping...")
    df['debit_credit_ratio'] = df['debit_credit_ratio'].replace([np.inf, -np.inf], 1e6)

df['gl_account'] = df['ACCOUNT'].astype(str)
df['cost_center'] = df['CLASS_FLD'].astype(str).fillna('UNKNOWN')
df['profit_center'] = df['FUND_CODE'].astype(str)

# Validate AGENCYNBR has no NaN (prevents "AGY_nan" entities)
if df['AGENCYNBR'].isna().any():
    nan_count = df['AGENCYNBR'].isna().sum()
    print(f"\n[WARNING] {nan_count} rows have missing AGENCYNBR, dropping them...")
    df = df.dropna(subset=['AGENCYNBR'])
    print(f"  Remaining transactions: {len(df):,}")

# Keep original columns for reporting
df['entity_id'] = 'AGY_' + df['AGENCYNBR'].astype(str)
df['entity_name'] = df['AGENCYNAME'].astype(str)

categorical_cols = ['gl_account', 'cost_center', 'profit_center']
numerical_cols = ['debit_amount', 'credit_amount', 'debit_credit_ratio',
                 'net_balance', 'abs_balance']

preprocessor = GLDataPreprocessor(categorical_cols, numerical_cols)
features = preprocessor.fit_transform(df)

print(f"Feature shape: {features.shape}")
print(f"Total transactions: {len(df):,}")

# Validate preprocessing output
if features.shape[0] == 0:
    raise ValueError("Preprocessing produced empty feature matrix!")

if features.shape[0] != len(df):
    print(f"[WARNING] Feature count ({features.shape[0]}) doesn't match dataframe length ({len(df)})")
    print(f"  Some rows may have been filtered during preprocessing")

# Final NaN/Inf check after preprocessing
if np.isnan(features).any():
    nan_count = np.isnan(features).sum()
    print(f"[ERROR] {nan_count} NaN values in features after preprocessing!")
    raise ValueError("Features contain NaN after preprocessing - check input data quality")

if np.isinf(features).any():
    inf_count = np.isinf(features).sum()
    print(f"[ERROR] {inf_count} Inf values in features after preprocessing!")
    raise ValueError("Features contain Inf after preprocessing - check for extreme values")

# ============================================================================
# LOAD PRODUCTION MODEL
# ============================================================================
print("\n" + "=" * 80)
print("LOADING PRODUCTION MODEL")
print("=" * 80)

production_model_path = data_dir / 'models' / 'federated_optimized_rtx4070.pth'

if not production_model_path.exists():
    raise FileNotFoundError(f"Production model not found: {production_model_path}\nPlease run finalize_trained_model.py first.")

checkpoint = torch.load(str(production_model_path), map_location=DEVICE, weights_only=False)
print(f"Loaded production model")
print(f"  Validation Loss: {checkpoint['val_loss']:.4f}")

# Check if SPOT threshold manager is available
has_spot = 'threshold_manager_state' in checkpoint
if has_spot:
    print(f"  Using SPOT adaptive thresholds (per-entity)")
else:
    print(f"  Using legacy percentile threshold: {checkpoint['threshold']:.4f}")

# CRITICAL: Load preprocessor state from checkpoint, don't fit new one!
preprocessor_state = checkpoint['preprocessor_state']
model_config = checkpoint['model_config']

# Verify dimensions match
if features.shape[1] != model_config['input_dim']:
    print(f"\n[ERROR] Feature dimension mismatch!")
    print(f"  Model expects: {model_config['input_dim']} features")
    print(f"  Data has: {features.shape[1]} features")
    print(f"\n  This means the data has different categories than training data.")
    print(f"  The model cannot process this data safely.")
    raise ValueError("Feature dimension mismatch - data incompatible with model")

# Validate data quality
print(f"\nValidating input data quality...")
if features.shape[0] == 0:
    raise ValueError("No data to process!")

# Check for all-zero features (indicates preprocessing error)
zero_features = (features == 0).all(axis=0).sum()
if zero_features > 0:
    print(f"  [WARNING] {zero_features} features are all zeros (may indicate missing data)")

# Check for NaN or Inf in features
if np.isnan(features).any():
    print(f"  [ERROR] NaN values detected in features! Cannot proceed.")
    raise ValueError("Features contain NaN values")
    
if np.isinf(features).any():
    print(f"  [ERROR] Inf values detected in features! Cannot proceed.")
    raise ValueError("Features contain Inf values")

print(f"  [OK] Data quality checks passed")

# Recreate model
model = GLAutoencoder(
    input_dim=model_config['input_dim'],
    latent_dim=model_config['latent_dim'],
    architecture=model_config['architecture']
).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create detector with SAVED indices from training (not new preprocessor!)
detector = AnomalyDetector(
    model=model,
    categorical_indices=preprocessor_state['categorical_indices'],
    numerical_indices=preprocessor_state['numerical_indices'],
    threshold_percentile=95.0,
    theta=2/3
)

# Load threshold manager if available (SPOT), otherwise use legacy threshold
if has_spot:
    from src.models.threshold_manager import ThresholdManager
    detector.threshold_manager = ThresholdManager.from_state(
        checkpoint['threshold_manager_state'],
        device=DEVICE
    )
    summary = detector.threshold_manager.get_entity_summary()
    print(f"  Loaded SPOT thresholds for {summary['n_entities']} entities")
    if summary['n_surge'] > 0:
        print(f"  [WARNING] {summary['n_surge']} entities in SURGE MODE (drift detected)")
        print(f"            Affected entities: {summary['surge_entities']}")
else:
    detector.threshold = checkpoint['threshold']

print(f"Model ready for inference")

# ============================================================================
# DETECT ANOMALIES
# ============================================================================
print("\n" + "=" * 80)
print("DETECTING ANOMALIES")
print("=" * 80)

try:
    # Convert to float32 explicitly to avoid precision issues
    # FloatTensor expects float32, but numpy might be float64
    if features.dtype != np.float32:
        features = features.astype(np.float32)
    
    X_tensor = torch.FloatTensor(features).to(DEVICE)
    
    # Detect anomalies - use SPOT if available, otherwise legacy method
    if has_spot:
        # Extract entity IDs for per-entity thresholding
        # Force string type to ensure consistent key matching with training
        entity_ids = df['AGENCYNBR'].astype(str).values
        predictions, drift_flags = detector.predict_with_entity(X_tensor, entity_ids)
        errors = detector.compute_reconstruction_errors(X_tensor)  # For stats
    else:
        # Legacy percentile-based detection
        errors, predictions = detector.detect_anomalies(X_tensor)
        drift_flags = np.zeros(len(predictions), dtype=bool)  # No drift detection
        
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print(f"\n[ERROR] GPU out of memory! Try reducing data size or use CPU.")
        print(f"  Current data size: {len(features):,} samples")
        raise
    else:
        print(f"\n[ERROR] Model inference failed: {e}")
        raise
except Exception as e:
    print(f"\n[ERROR] Unexpected error during anomaly detection: {e}")
    raise

# Validate array lengths match before assignment
if len(errors) != len(df):
    raise ValueError(f"Length mismatch! Model returned {len(errors)} predictions but dataframe has {len(df)} rows. "
                     f"This indicates preprocessing filtered some rows.")

if len(predictions) != len(df):
    raise ValueError(f"Length mismatch! Model returned {len(predictions)} anomaly flags but dataframe has {len(df)} rows.")

# Add predictions to dataframe (use .loc to avoid SettingWithCopyWarning)
df.loc[:, 'reconstruction_error'] = errors
df.loc[:, 'is_anomaly'] = predictions
df.loc[:, 'drift_detected'] = drift_flags

# Add surge mode flag for SPOT
if has_spot:
    df.loc[:, 'surge_mode'] = df['AGENCYNBR'].apply(
        lambda x: detector.threshold_manager.surge_mode.get(str(x), False)
    )
else:
    df.loc[:, 'surge_mode'] = False

total_anomalies = int(df['is_anomaly'].sum())
anomaly_rate = (float(total_anomalies) / len(df)) * 100 if len(df) > 0 else 0.0  # Explicit float
drift_count = int(drift_flags.sum())

print(f"Total anomalies detected: {total_anomalies:,}")
print(f"Anomaly rate: {anomaly_rate:.2f}%")
if has_spot:
    surge_anomalies = int(df[df['surge_mode'] == True]['is_anomaly'].sum())
    print(f"Drift flags: {drift_count} samples")
    print(f"Surge mode anomalies: {surge_anomalies} (require priority review)")

# Calculate mean errors with safety checks
normal_mask = df['is_anomaly'] == 0
anomaly_mask = df['is_anomaly'] == 1

if normal_mask.sum() > 0:
    print(f"Mean error (normal): {df[normal_mask]['reconstruction_error'].mean():.4f}")
else:
    print("Mean error (normal): N/A (no normal samples)")

if anomaly_mask.sum() > 0:
    print(f"Mean error (anomaly): {df[anomaly_mask]['reconstruction_error'].mean():.4f}")
else:
    print("Mean error (anomaly): N/A (no anomalies detected)")

# ============================================================================
# SURGE MODE ALERTS (DRIFT DETECTION)
# ============================================================================
if has_spot:
    surge_anomalies = df[(df['surge_mode'] == True) & (df['is_anomaly'] == 1)]
    
    if len(surge_anomalies) > 0:
        print("\n" + "=" * 80)
        print("[WARNING] SURGE MODE ANOMALIES DETECTED - POTENTIAL ATTACK")
        print("=" * 80)
        
        surge_entities = surge_anomalies['entity_id'].unique()
        print(f"\n{len(surge_anomalies)} anomalies detected in SURGE MODE")
        print(f"Affected entities: {len(surge_entities)}")
        print(f"\nDrift detected in error distributions - possible data poisoning attack!")
        print(f"Thresholds have been frozen and tightened for affected entities.")
        print(f"\n[ACTION REQUIRED] Manual review needed to confirm drift legitimacy:")
        
        for entity in surge_entities[:5]:  # Show top 5
            entity_surge = surge_anomalies[surge_anomalies['entity_id'] == entity]
            print(f"  - {entity}: {len(entity_surge)} anomalies")
        
        if len(surge_entities) > 5:
            print(f"  ... and {len(surge_entities) - 5} more entities")
        
        print("\n" + "=" * 80)

# ============================================================================
# TOP ANOMALIES BY ENTITY
# ============================================================================
print("\n" + "=" * 80)
print("TOP ANOMALIES DETECTED")
print("=" * 80)

# Filter anomalies
anomalies_df = df[df['is_anomaly'] == 1].copy()

if len(anomalies_df) == 0:
    print("\nNo anomalies detected!")
else:
    # Sort by reconstruction error (highest first)
    anomalies_df = anomalies_df.sort_values('reconstruction_error', ascending=False)
    
    # Group by entity
    entities = anomalies_df['entity_id'].unique()
    
    if len(entities) == 0:
        print("\nNo entities found in anomalies!")
    else:
        # Show top 5 entities with most/highest anomalies
        entity_anomaly_scores = anomalies_df.groupby('entity_id', as_index=False).agg({
            'reconstruction_error': ['sum', 'count', 'mean']
        })
        # Flatten multi-level columns
        entity_anomaly_scores.columns = ['entity_id', 'total_error', 'count', 'mean_error']
        entity_anomaly_scores = entity_anomaly_scores.sort_values('total_error', ascending=False)
        
        top_entities = entity_anomaly_scores.head(10)['entity_id'].tolist()
        
        for entity_id in top_entities:
            entity_anomalies = anomalies_df[anomalies_df['entity_id'] == entity_id]
            
            if len(entity_anomalies) == 0:
                continue  # Skip if no anomalies for this entity
            
            # Safe access with fallback
            try:
                entity_name = entity_anomalies['entity_name'].iloc[0]
            except (IndexError, KeyError):
                entity_name = f"Unknown Entity ({entity_id})"
            
            print(f"\n{entity_name} ({entity_id}):")
            print("-" * 80)
            
            # Show top 5 anomalies for this entity
            top_anomalies = entity_anomalies.nlargest(5, 'reconstruction_error')
            
            for idx, row in top_anomalies.iterrows():
                gl = str(row['gl_account'])[:20]  # Truncate long GL codes
                debit = float(row['debit_amount'])
                credit = float(row['credit_amount'])
                error = float(row['reconstruction_error'])
                
                # Format with safety checks for very large numbers
                debit_str = f"${debit:,.2f}" if abs(debit) < 1e12 else f"${debit:.2e}"
                credit_str = f"${credit:,.2f}" if abs(credit) < 1e12 else f"${credit:.2e}"
                
                print(f"  GL {gl:<20s} | Debit: {debit_str:>18s} | Credit: {credit_str:>18s} | Error: {error:.4f}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("ANOMALY SUMMARY BY ENTITY")
print("=" * 80)

if len(anomalies_df) > 0:
    entity_summary = anomalies_df.groupby('entity_id', as_index=False).agg({
        'is_anomaly': 'count',
        'reconstruction_error': 'mean',
        'debit_amount': 'sum',
        'credit_amount': 'sum'
    })
    
    entity_summary.columns = ['Entity ID', 'Anomaly Count', 'Avg Error', 'Total Debit', 'Total Credit']
    
    # Convert to float64 to prevent overflow in large sums
    entity_summary['Total Debit'] = entity_summary['Total Debit'].astype(np.float64)
    entity_summary['Total Credit'] = entity_summary['Total Credit'].astype(np.float64)
    
    entity_summary = entity_summary.sort_values('Anomaly Count', ascending=False)
    
    print(f"\nTop 10 Entities by Anomaly Count:")
    print("-" * 80)
    
    for idx, row in entity_summary.head(10).iterrows():
        print(f"{row['Entity ID']:30s} | {int(row['Anomaly Count']):5d} anomalies | Avg Error: {row['Avg Error']:.4f}")

# ============================================================================
# EXPORT ANOMALIES
# ============================================================================
print("\n" + "=" * 80)
print("EXPORTING RESULTS")
print("=" * 80)

# Always create timestamped detection files (even if no anomalies)
# This ensures FCL retraining can always find detection files
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
detections_dir = data_dir / 'detections'
detections_dir.mkdir(exist_ok=True)

json_output_path = detections_dir / f'detections_{timestamp}.json'
features_output_path = detections_dir / f'features_{timestamp}.npy'

if len(anomalies_df) > 0:
    # Select relevant columns for export (CSV for legacy compatibility)
    export_cols = ['entity_id', 'entity_name', 'gl_account', 'cost_center', 'profit_center',
                   'debit_amount', 'credit_amount', 'net_balance', 'reconstruction_error']
    
    anomalies_export = anomalies_df[export_cols].copy()
    anomalies_export = anomalies_export.sort_values('reconstruction_error', ascending=False)
    
    output_path = data_dir / 'results' / 'detected_anomalies.csv'
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Use UTF-8 encoding with BOM for Excel compatibility on Windows
        anomalies_export.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Anomalies exported to: {output_path}")
        print(f"Total rows exported: {len(anomalies_export):,}")
    except PermissionError:
        print(f"[WARNING] Permission denied writing to {output_path}")
        print(f"  File may be open in Excel or locked by another process")
    except UnicodeEncodeError as e:
        print(f"[WARNING] Unicode encoding error in data: {e}")
        print(f"  Some special characters may not be supported. Trying ASCII fallback...")
        try:
            anomalies_export.to_csv(output_path, index=False, encoding='ascii', errors='replace')
            print(f"  Exported with ASCII encoding (special chars replaced)")
        except:
            print(f"  ASCII fallback also failed. Export skipped.")
    except Exception as e:
        print(f"[WARNING] Failed to export anomalies: {e}")
    
    # Export to JSON with FULL THRESHOLD PROVENANCE (enhanced for ADT)
    
    try:
        # Use detect_anomalies_detailed() if SPOT is available for full threshold snapshots
        if has_spot:
            print("\n[INFO] Exporting detailed anomalies with threshold provenance...")
            detailed_anomalies = detector.detect_anomalies_detailed(
                X_tensor,
                entity_ids,
                df
            )
            
            # Extract anomaly indices and save features for FCL retraining
            anomaly_indices = [i for i, pred in enumerate(predictions) if pred == 1]
            if len(anomaly_indices) > 0:
                anomaly_features = X_tensor.cpu().numpy()[anomaly_indices]
                np.save(features_output_path, anomaly_features)
                print(f"[INFO] Saved {len(anomaly_indices)} anomaly features to: {features_output_path.name}")
            else:
                anomaly_features = np.array([])
                print("[INFO] No anomaly features to save")
            
            # Group by entity
            anomalies_by_entity = {}
            for record in detailed_anomalies:
                entity_id = record['entity_id']
                if entity_id not in anomalies_by_entity:
                    anomalies_by_entity[entity_id] = {
                        'entity_name': record['entity_name'],
                        'anomaly_count': 0,
                        'anomalies': []
                    }
                anomalies_by_entity[entity_id]['anomaly_count'] += 1
                anomalies_by_entity[entity_id]['anomalies'].append(record)
            
            json_data = {
                "metadata": {
                    "total_transactions": len(df),
                    "total_anomalies": len(detailed_anomalies),
                    "anomaly_rate": f"{(len(detailed_anomalies) / len(df) * 100):.2f}%",
                    "entities_with_anomalies": len(anomalies_by_entity),
                    "detection_method": "SPOT+ADWIN" if has_spot else "percentile",
                    "adt_enabled": detector.threshold_manager.enable_adt if has_spot else False,
                    "timestamp": datetime.now().isoformat(),
                    "features_file": f"features_{timestamp}.npy",
                    "feature_shape": list(anomaly_features.shape) if len(anomaly_features) > 0 else [0, 0]
                },
                "anomalies_by_entity": anomalies_by_entity
            }
        else:
            # Legacy export without threshold snapshots
            # Still save features for potential FCL retraining
            anomaly_indices = list(anomalies_df.index)
            if len(anomaly_indices) > 0:
                anomaly_features = X_tensor.cpu().numpy()[anomaly_indices]
                np.save(features_output_path, anomaly_features)
                print(f"[INFO] Saved {len(anomaly_indices)} anomaly features to: {features_output_path.name}")
            else:
                anomaly_features = np.array([])
            
            json_data = {
                "metadata": {
                    "total_transactions": len(df),
                    "total_anomalies": len(anomalies_df),
                    "anomaly_rate": f"{(len(anomalies_df) / len(df) * 100):.2f}%",
                    "entities_with_anomalies": anomalies_df['entity_id'].nunique(),
                    "detection_method": "percentile",
                    "timestamp": datetime.now().isoformat(),
                    "features_file": f"features_{timestamp}.npy",
                    "feature_shape": list(anomaly_features.shape) if len(anomaly_features) > 0 else [0, 0]
                },
                "anomalies_by_entity": {}
            }
            
            # Legacy grouping
            for entity_id, entity_group in anomalies_df.groupby('entity_id'):
                entity_group = entity_group.sort_values('reconstruction_error', ascending=False)
                json_data["anomalies_by_entity"][str(entity_id)] = {
                    "entity_name": entity_group.iloc[0]['entity_name'] if 'entity_name' in entity_group.columns else "UNKNOWN",
                    "anomaly_count": len(entity_group),
                    "anomalies": [
                        {
                            "gl_account": str(row['gl_account']),
                            "amount": float(row['net_balance']),
                            "reconstruction_error": float(row['reconstruction_error'])
                        }
                        for _, row in entity_group.iterrows()
                    ]
                }
        
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"JSON export saved to: {json_output_path}")
        print(f"  {len(json_data['anomalies_by_entity'])} entities with anomalies")
        if has_spot:
            print(f"  [INFO] Includes full threshold provenance for audit trails")
    except Exception as e:
        print(f"[WARNING] Failed to export JSON: {e}")
else:
    # No anomalies detected - still create empty detection files for FCL retraining consistency
    print("\nNo anomalies detected - creating empty detection archive...")
    
    # Create empty .npy file
    empty_features = np.array([]).reshape(0, features.shape[1])
    np.save(features_output_path, empty_features)
    
    # Create empty JSON
    json_data = {
        "metadata": {
            "total_transactions": len(df),
            "total_anomalies": 0,
            "anomaly_rate": "0.00%",
            "entities_with_anomalies": 0,
            "detection_method": "SPOT+ADWIN" if has_spot else "percentile",
            "adt_enabled": detector.threshold_manager.enable_adt if has_spot else False,
            "timestamp": datetime.now().isoformat(),
            "features_file": f"features_{timestamp}.npy",
            "feature_shape": [0, features.shape[1]]
        },
        "anomalies_by_entity": {}
    }
    
    try:
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Empty detection archive saved to: {json_output_path}")
        print(f"  Features file: {features_output_path.name}")
    except Exception as e:
        print(f"[WARNING] Failed to create empty detection archive: {e}")

print("\n" + "=" * 80)
print("DETECTION COMPLETE!")
print("=" * 80)

