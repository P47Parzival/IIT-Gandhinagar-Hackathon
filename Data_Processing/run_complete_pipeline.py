#!/usr/bin/env python3
"""
Complete BalanceGuard AI Pipeline
Paper 1 (Anomaly Detection) → Paper 2 (Validation & Explanation)
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add paths
anomaly_detector_path = os.path.join(os.path.dirname(__file__), 'Anomaly_Detector')
anomaly_validator_path = os.path.join(os.path.dirname(__file__), 'Anomaly_Validator', 'src')
sys.path.insert(0, anomaly_detector_path)
sys.path.insert(0, anomaly_validator_path)
sys.path.insert(0, os.path.join(anomaly_detector_path, 'src'))  # Add Anomaly_Detector/src

# Load environment variables from .env file
env_file = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

print("\n" + "="*90)
print("BALANCEGUARD AI - COMPLETE PIPELINE".center(90))
print("Paper 1 (FCL Anomaly Detection) → Paper 2 (LLM Multi-Agent Validation)".center(90))
print("="*90 + "\n")

# ============================================================================
# PHASE 1: ANOMALY DETECTION (Paper 1)
# ============================================================================

print("="*90)
print("PHASE 1: ANOMALY DETECTION (Federated Continual Learning)".center(90))
print("="*90 + "\n")

try:
    from src.models.autoencoder import AnomalyDetector
    from src.data.preprocessing import GLDataPreprocessor
except ImportError:
    from models.autoencoder import AnomalyDetector
    from data.preprocessing import GLDataPreprocessor

# 1.1 Load trained model
print("📦 Loading trained model...")
model_path = 'Anomaly_Detector/data/models/full_dataset_production.pth'

if not os.path.exists(model_path):
    print(f"[ERROR] Model not found at: {model_path}")
    print("   Please run: python Anomaly_Detector/scripts/train_full_dataset_tuned.py")
    sys.exit(1)

# Detect device (GPU if available, CPU otherwise)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device} {'(GPU acceleration enabled)' if device.type == 'cuda' else '(CPU mode)'}")

# Load checkpoint to appropriate device
checkpoint = torch.load(model_path, map_location=device)
print(f"[OK] Production model loaded from checkpoint")
print(f"  Val Loss: {checkpoint['best_val_loss']:.4f}")
print(f"  Threshold: {checkpoint['threshold']:.4f}")

# 1.2 Load and preprocess data
print("\n📊 Loading ledger data...")
df_raw = pd.read_csv('Anomaly_Detector/data/raw/ledger_fy25_qtr4.csv', low_memory=False)
print(f"✓ Loaded: {len(df_raw):,} transactions")

# Prepare data (match EXACTLY the training script)
df = pd.DataFrame()
df['entity_id'] = 'AGY_' + df_raw['AGENCYNBR'].astype(str)
df['entity_name'] = df_raw['AGENCYNAME']
df['period'] = df_raw['FISCAL_YEAR'].astype(str) + '-' + df_raw['ACCOUNTING_PERIOD'].astype(str).str.zfill(2)
df['gl_account'] = df_raw['ACCOUNT'].astype(str)
df['gl_name'] = df_raw['ACCTDESCR']
df['cost_center'] = df_raw['DEPTID'].fillna('UNKNOWN')
df['profit_center'] = df_raw['FUND_CODE'].astype(str)
df['document_number'] = df_raw.get('PROJECT_ID', '').fillna('').astype(str)
df['document_type'] = df_raw.get('ACTIVITY', 'UNK').fillna('UNK')
df['net_amount'] = df_raw['POSTED_TOTAL_AMT'].fillna(0.0)
df['debit_amount'] = df['net_amount'].apply(lambda x: max(0, x))
df['credit_amount'] = df['net_amount'].apply(lambda x: max(0, -x))
df['debit_credit_ratio'] = df['debit_amount'] / (df['credit_amount'] + 1e-6)
df['net_balance'] = df['debit_amount'] - df['credit_amount']
df['abs_balance'] = np.abs(df['net_balance'])
df['fiscal_year'] = df_raw['FISCAL_YEAR']

print(f"✓ Preprocessed {len(df):,} records")

# 1.3 Create preprocessor and detector
print("\n🔧 Initializing detector...")
# Use the SAME columns as training script
categorical_cols = ['gl_account', 'cost_center', 'profit_center']
numerical_cols = ['debit_amount', 'credit_amount', 'debit_credit_ratio',
                 'net_balance', 'abs_balance']
preprocessor = GLDataPreprocessor(categorical_cols, numerical_cols)

# Fit on full dataset
X = preprocessor.fit_transform(df)
input_dim = X.shape[1]
print(f"✓ Input dimension: {input_dim}")

# Load model - create autoencoder first and move to device
from Anomaly_Detector.src.models.autoencoder import GLAutoencoder
autoencoder = GLAutoencoder(input_dim=input_dim, latent_dim=32, architecture='deep')
autoencoder.load_state_dict(checkpoint['model_state_dict'])
autoencoder.to(device)  # Move model to GPU/CPU
autoencoder.eval()

# Create detector with the loaded model
detector = AnomalyDetector(
    model=autoencoder,
    categorical_indices=preprocessor.categorical_indices,
    numerical_indices=preprocessor.numerical_indices
)
detector.threshold = checkpoint.get('threshold', 0.05)

print(f"✓ Anomaly threshold: {detector.threshold:.4f}")
print(f"✓ Model ready on {device}")

# 1.4 Detect anomalies
print("\n🔍 Detecting anomalies...")
X_tensor = torch.FloatTensor(X).to(device)  # Move data to device
reconstruction_errors, is_anomaly = detector.detect_anomalies(X_tensor)

n_anomalies = np.sum(is_anomaly)
print(f"✓ Detected {n_anomalies:,} anomalies ({100*n_anomalies/len(df):.2f}% of transactions)")

# 1.5 Export anomalies for validation
print("\n📤 Exporting anomalies...")
anomaly_records = []

for idx in np.where(is_anomaly)[0]:
    record = {
        'anomaly_id': f"ANO_{datetime.now().strftime('%Y%m%d%H%M%S')}_{idx}",
        'detection_timestamp': datetime.now().isoformat(),
        'entity_id': df.iloc[idx]['entity_id'],
        'entity_name': df.iloc[idx]['entity_name'],
        'period': df.iloc[idx]['period'],
        'fiscal_year': int(df.iloc[idx]['fiscal_year']),
        'gl_account': df.iloc[idx]['gl_account'],
        'gl_name': df.iloc[idx]['gl_name'],
        'document_number': df.iloc[idx]['document_number'],
        'document_type': df.iloc[idx]['document_type'],
        'debit_amount': float(df.iloc[idx]['debit_amount']),
        'credit_amount': float(df.iloc[idx]['credit_amount']),
        'amount': float(df.iloc[idx]['debit_amount'] - df.iloc[idx]['credit_amount']),
        'reconstruction_error': float(reconstruction_errors[idx]),
        'threshold': float(detector.threshold),
        'anomaly_score': float(reconstruction_errors[idx] / detector.threshold),
        'severity': 'CRITICAL' if reconstruction_errors[idx] > detector.threshold * 2 else 
                   'HIGH' if reconstruction_errors[idx] > detector.threshold * 1.5 else 'MEDIUM'
    }
    anomaly_records.append(record)

print(f"✓ Exported {len(anomaly_records)} anomaly records")

# Save detected anomalies
output_dir = Path('data/detected_anomalies')
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

with open(output_file, 'w') as f:
    json.dump(anomaly_records, f, indent=2)

print(f"✓ Saved to: {output_file}")

# ============================================================================
# PHASE 2: ANOMALY VALIDATION (Paper 2)
# ============================================================================

print("\n" + "="*90)
print("PHASE 2: ANOMALY VALIDATION (LLM Multi-Agent System)".center(90))
print("="*90 + "\n")

from pipeline.validator_pipeline import ValidatorPipeline

# 2.1 Initialize validator
print("⚙️  Initializing validation pipeline...")
pipeline = ValidatorPipeline(
    config_path="Anomaly_Validator/config/validator_config.yaml"
)

# 2.2 Select top anomalies for validation (limit to avoid API quota)
print("\n📋 Selecting top anomalies for validation...")
top_anomalies = sorted(anomaly_records, key=lambda x: x['anomaly_score'], reverse=True)[:1]  # Just 1 for testing

print(f"✓ Selected {len(top_anomalies)} highest-priority anomalies:\n")
for i, anom in enumerate(top_anomalies, 1):
    print(f"   {i}. {anom['entity_name']}")
    print(f"      GL: {anom['gl_account']} | Amount: ${anom['amount']:,.2f}")
    print(f"      Score: {anom['anomaly_score']:.2f} | Severity: {anom['severity']}")
    print()

# 2.3 Start real-time processing
pipeline.start_realtime_processing()
print("✓ Real-time validation started\n")

# 2.4 Submit anomalies for validation
print("🚀 Submitting anomalies for validation...\n")
validation_results = []

for anom in top_anomalies:
    print(f"{'='*90}")
    print(f"Validating: {anom['entity_name']}")
    print(f"GL Account: {anom['gl_account']} ({anom['gl_name']})")
    print(f"{'='*90}\n")
    
    try:
        # Submit to validation pipeline
        result = pipeline.validate_single(anom)
        validation_results.append({
            'anomaly': anom,
            'validation': result
        })
        
        # Print result
        decision = result.get('decision', 'UNKNOWN')
        confidence = result.get('confidence', 0.0)
        
        # Fix confidence display (stored as 0-1, display as percentage)
        if isinstance(confidence, (int, float)) and confidence <= 1.0:
            confidence_pct = confidence * 100
        else:
            confidence_pct = confidence
        
        flag_emoji = "🔴" if decision == "RED_FLAG" else "🟡" if decision == "YELLOW_FLAG" else "⚪"
        print(f"\n{flag_emoji} RESULT: {decision} (Confidence: {confidence_pct:.1f}%)")
        
        if 'summary' in result:
            print(f"📝 {result['summary']}")
        
        print("\n")
        
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}\n")
        validation_results.append({
            'anomaly': anom,
            'validation': {'error': str(e), 'decision': 'ERROR'}
        })

# ============================================================================
# PHASE 3: GENERATE SUMMARY REPORT
# ============================================================================

print("\n" + "="*90)
print("PHASE 3: SUMMARY REPORT".center(90))
print("="*90 + "\n")

# Count results
red_flags = sum(1 for r in validation_results if r['validation'].get('decision') == 'RED_FLAG')
yellow_flags = sum(1 for r in validation_results if r['validation'].get('decision') == 'YELLOW_FLAG')
errors = sum(1 for r in validation_results if r['validation'].get('decision') == 'ERROR')

print(f"📊 Validation Summary:")
print(f"   Total Anomalies Detected: {n_anomalies:,}")
print(f"   Validated: {len(validation_results)}")
print(f"   🔴 RED FLAGS: {red_flags} (Require urgent review)")
print(f"   🟡 YELLOW FLAGS: {yellow_flags} (Explained with documentation)")
print(f"   ❌ ERRORS: {errors}")
print()

# Save complete results
results_dir = Path('data/validation_results')
results_dir.mkdir(parents=True, exist_ok=True)
results_file = results_dir / f"complete_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

complete_results = {
    'pipeline_run': {
        'timestamp': datetime.now().isoformat(),
        'model': model_path,
        'data_file': 'Anomaly_Detector/data/raw/ledger_fy25_qtr4.csv'
    },
    'detection_summary': {
        'total_transactions': len(df),
        'anomalies_detected': n_anomalies,
        'detection_rate': float(n_anomalies / len(df)),
        'threshold': float(detector.threshold)
    },
    'validation_summary': {
        'validated': len(validation_results),
        'red_flags': red_flags,
        'yellow_flags': yellow_flags,
        'errors': errors
    },
    'results': validation_results
}

with open(results_file, 'w') as f:
    json.dump(complete_results, f, indent=2)

print(f"💾 Complete results saved to: {results_file}")

print("\n" + "="*90)
print("✅ PIPELINE COMPLETE!".center(90))
print("="*90 + "\n")

print("Next Steps:")
print("  1. Review RED FLAG anomalies in validation results")
print("  2. Check YELLOW FLAG explanations for business context")
print("  3. Investigate top anomalies with supporting documents")
print()

