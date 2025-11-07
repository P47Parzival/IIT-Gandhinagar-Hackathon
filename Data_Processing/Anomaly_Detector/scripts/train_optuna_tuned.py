"""
Optuna Hyperparameter Tuning for Anomaly Detection
Compares Paper Parameters vs Optimized Parameters
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.trial import TrialState
import json
from datetime import datetime

from src.models.autoencoder import GLAutoencoder, combined_loss, AnomalyDetector
from src.data.preprocessing import GLDataPreprocessor

print("=" * 80)
print("OPTUNA HYPERPARAMETER TUNING - Paper vs Optimized")
print("=" * 80)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load Data ONCE (reuse across trials)
print("\n" + "=" * 80)
print("LOADING DATASET")
print("-" * 80)

df_raw = pd.read_csv('../data/raw/ledger_fy25_qtr4.csv', low_memory=False)
print(f"[OK] Loaded: {len(df_raw):,} transactions")

df = pd.DataFrame()
df['entity_id'] = 'AGY_' + df_raw['AGENCYNBR'].astype(str)
df['entity_name'] = df_raw['AGENCYNAME']
df['period'] = df_raw['FISCAL_YEAR'].astype(str) + '-' + df_raw['ACCOUNTING_PERIOD'].astype(str).str.zfill(2)
df['gl_account'] = df_raw['ACCOUNT'].astype(str)
df['gl_name'] = df_raw['ACCTDESCR']
df['cost_center'] = df_raw['CLASS_FLD'].astype(str).fillna('UNKNOWN')  # FIXED: Use CLASS_FLD (349 values) not DEPTID (1585 values)
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

print(f"[OK] Preprocessed {len(df):,} records")

# Preprocessing
print("\n" + "=" * 80)
print("PREPROCESSING")
print("-" * 80)

categorical_cols = ['gl_account', 'cost_center', 'profit_center']
numerical_cols = ['debit_amount', 'credit_amount', 'debit_credit_ratio',
                 'net_balance', 'abs_balance']

preprocessor = GLDataPreprocessor(categorical_cols, numerical_cols)
X_all = preprocessor.fit_transform(df)
input_dim = X_all.shape[1]

print(f"[OK] Input dimension: {input_dim}")
print(f"[OK] Total samples: {len(X_all):,}")

# Split train/val
n_samples = len(X_all)
n_train = int(0.8 * n_samples)

X_train = X_all[:n_train]
X_val = X_all[n_train:]

print(f"[OK] Train: {len(X_train):,} | Val: {len(X_val):,}")

# Convert to tensors (reuse across trials)
X_train_tensor = torch.FloatTensor(X_train)
X_val_tensor = torch.FloatTensor(X_val)


def objective(trial):
    """
    Optuna objective function to optimize hyperparameters.
    
    Returns validation loss (lower is better).
    """
    
    # Suggest hyperparameters
    config_type = trial.suggest_categorical('config_type', ['paper', 'optimized', 'custom'])
    
    if config_type == 'paper':
        # Paper's parameters from arXiv:2210.15051 (Appendix A.3-A.5)
        batch_size = 16  # Paper uses 16 (γ=16)
        learning_rate = 0.001  # Paper doesn't specify, using Adam defaults
        latent_dim = 2  # Paper uses 2 for visualization
        weight_decay = 0.0  # Paper doesn't use weight decay
        architecture = 'deep'  # Paper uses deep for local anomalies
        
    elif config_type == 'optimized':
        # Current best-known parameters
        batch_size = 512
        learning_rate = 0.0001
        latent_dim = 32
        weight_decay = 1e-5
        architecture = 'deep'
        
    else:  # custom - let Optuna explore
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256, 512])
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        latent_dim = trial.suggest_categorical('latent_dim', [2, 4, 8, 16, 32, 64])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        architecture = trial.suggest_categorical('architecture', ['shallow', 'deep'])
    
    # Fixed hyperparameters (matching production settings)
    n_epochs = 50  # Increased for proper convergence (production uses 50)
    patience = 10  # Increased to allow more exploration (production uses 10)
    
    # Print trial info
    print(f"\n[Trial {trial.number}] Testing config: {config_type}")
    if config_type == 'paper':
        print(f"  Parameters: batch={batch_size}, lr={learning_rate}, latent={latent_dim}, arch={architecture}")
    elif config_type == 'optimized':
        print(f"  Parameters: batch={batch_size}, lr={learning_rate}, latent={latent_dim}, arch={architecture}")
    else:
        print(f"  Parameters: batch={batch_size}, lr={learning_rate:.6f}, latent={latent_dim}, arch={architecture}")
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = GLAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        architecture=architecture
    ).to(DEVICE)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)  # Paper's β values
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=False, min_lr=1e-6
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("  Training:")
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_x, in train_loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            
            reconstructed = model(batch_x)
            loss = combined_loss(
                batch_x, reconstructed,
                preprocessor.categorical_indices,
                preprocessor.numerical_indices,
                theta=2/3  # Paper's θ value
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
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
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Track best
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            status = " ** BEST"
        else:
            patience_counter += 1
        
        # Print progress every epoch
        print(f"    Epoch {epoch+1:2d}/{n_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.6f}{status}")
        
        # Report intermediate value for pruning
        trial.report(val_loss, epoch)
        
        # Handle pruning
        if trial.should_prune():
            print(f"  [PRUNED] Trial stopped early at epoch {epoch+1}")
            raise optuna.TrialPruned()
        
        # Early stopping
        if patience_counter >= patience:
            print(f"  [EARLY STOP] Stopped at epoch {epoch+1} (patience={patience})")
            break
    
    # Evaluate anomaly detection performance
    detector = AnomalyDetector(
        model=model,
        categorical_indices=preprocessor.categorical_indices,
        numerical_indices=preprocessor.numerical_indices,
        threshold_percentile=95.0
    )
    
    sample_size = min(50000, len(X_val))
    X_sample = torch.FloatTensor(X_val[:sample_size]).to(DEVICE)
    detector.fit_threshold(X_sample)
    
    errors, anomalies = detector.detect_anomalies(X_sample)
    anomaly_rate = anomalies.mean() * 100
    
    # Store additional metrics
    trial.set_user_attr('anomaly_rate', float(anomaly_rate))
    trial.set_user_attr('threshold', float(detector.threshold))
    trial.set_user_attr('final_epoch', epoch + 1)
    
    print(f"  [COMPLETE] Best Val Loss: {best_val_loss:.4f} | Anomaly Rate: {anomaly_rate:.2f}%\n")
    
    return best_val_loss


def run_paper_baseline():
    """
    Run the paper's exact configuration as a baseline.
    """
    print("\n" + "=" * 80)
    print("PAPER BASELINE (arXiv:2210.15051)")
    print("=" * 80)
    print("\nConfiguration from Appendix A.3-A.5:")
    print("  - Batch size: 16 (γ=16)")
    print("  - Latent dim: 2 (for visualization)")
    print("  - Architecture: Deep (Table 3)")
    print("  - Loss: θ=2/3 (Equation 4)")
    print("  - Optimizer: Adam (β1=0.9, β2=0.999)")
    
    # Paper's exact parameters
    batch_size = 16
    learning_rate = 0.001
    latent_dim = 2
    n_epochs = 50  # Increased for fair comparison with Optuna trials
    patience = 10
    
    train_dataset = TensorDataset(X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = GLAutoencoder(input_dim=input_dim, latent_dim=latent_dim, architecture='deep').to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("\nTraining progress:")
    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        for batch_x, in train_loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = combined_loss(batch_x, reconstructed,
                               preprocessor.categorical_indices,
                               preprocessor.numerical_indices,
                               theta=2/3)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, in val_loader:
                batch_x = batch_x.to(DEVICE)
                reconstructed = model(batch_x)
                loss = combined_loss(batch_x, reconstructed,
                                   preprocessor.categorical_indices,
                                   preprocessor.numerical_indices,
                                   theta=2/3)
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            status = " ** BEST"
        else:
            patience_counter += 1
        
        print(f"  Epoch {epoch+1:2d}/{n_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}{status}")
        
        # Early stopping for paper baseline too
        if patience_counter >= patience:
            print(f"  [EARLY STOP] Stopped at epoch {epoch+1} (patience={patience})")
            break
    
    print(f"\n[PAPER] Best Val Loss: {best_val_loss:.4f}")
    return best_val_loss


if __name__ == "__main__":
    # Run paper baseline first
    paper_loss = run_paper_baseline()
    
    # Run Optuna study
    print("\n" + "=" * 80)
    print("OPTUNA HYPERPARAMETER SEARCH")
    print("=" * 80)
    print("\nSearching for optimal parameters...")
    print("This will compare:")
    print("  1. Paper's configuration (batch=16, latent=2)")
    print("  2. Current optimized (batch=512, latent=32)")
    print("  3. Custom Optuna exploration")
    
    study = optuna.create_study(
        study_name="anomaly_detector_tuning",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(
        objective,
        n_trials=30,  # Try 30 different configurations
        timeout=3600,  # 1 hour max
        show_progress_bar=True,
        n_jobs=1  # Sequential (can't parallelize on single GPU)
    )
    
    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print(f"\nStudy statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")
    
    print(f"\nBEST TRIAL:")
    trial = study.best_trial
    print(f"  Val Loss: {trial.value:.4f}")
    print(f"  Anomaly Rate: {trial.user_attrs['anomaly_rate']:.2f}%")
    print(f"  Threshold: {trial.user_attrs['threshold']:.4f}")
    print(f"  Epochs: {trial.user_attrs['final_epoch']}")
    print(f"\n  Parameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Compare with paper
    print(f"\nCOMPARISON:")
    print(f"  Paper Baseline:  Val Loss = {paper_loss:.4f}")
    print(f"  Best Optuna:     Val Loss = {trial.value:.4f}")
    improvement = ((paper_loss - trial.value) / paper_loss) * 100
    print(f"  Improvement:     {improvement:+.2f}%")
    
    if trial.value < paper_loss:
        print(f"  [SUCCESS] Optuna found better parameters!")
    else:
        print(f"  [INFO] Paper parameters are competitive")
    
    # Top 5 trials
    print(f"\nTOP 5 TRIALS:")
    for i, t in enumerate(study.best_trials[:5], 1):
        config = t.params.get('config_type', 'custom')
        print(f"\n  {i}. Val Loss: {t.value:.4f} | Config: {config}")
        if config == 'custom':
            print(f"     Batch: {t.params.get('batch_size')}, "
                  f"LR: {t.params.get('learning_rate'):.6f}, "
                  f"Latent: {t.params.get('latent_dim')}, "
                  f"Arch: {t.params.get('architecture')}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'paper_baseline': float(paper_loss),
        'best_trial': {
            'value': float(trial.value),
            'params': trial.params,
            'user_attrs': trial.user_attrs
        },
        'all_trials': [
            {
                'number': t.number,
                'value': float(t.value) if t.value is not None else None,
                'params': t.params,
                'state': str(t.state)
            }
            for t in study.trials
        ]
    }
    
    os.makedirs('../data/models', exist_ok=True)
    results_file = f"../data/models/optuna_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Visualization
    print(f"\nGenerating optimization visualizations...")
    try:
        import plotly
        
        # Optimization history
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_html('../data/models/optuna_history.html')
        print(f"  [OK] Optimization history: ../data/models/optuna_history.html")
        
        # Parameter importances
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_html('../data/models/optuna_importances.html')
        print(f"  [OK] Parameter importances: ../data/models/optuna_importances.html")
        
        # Parallel coordinate plot
        fig3 = optuna.visualization.plot_parallel_coordinate(study)
        fig3.write_html('../data/models/optuna_parallel.html')
        print(f"  [OK] Parallel coordinates: ../data/models/optuna_parallel.html")
        
    except Exception as e:
        print(f"  [WARNING] Visualization failed: {e}")
    
    print("\n" + "=" * 80)
    print("OPTUNA TUNING COMPLETE!")
    print("=" * 80)
    
    print("\nRECOMMENDATION:")
    if trial.params.get('config_type') == 'paper':
        print("  Use PAPER parameters - they work best for your data!")
    elif trial.params.get('config_type') == 'optimized':
        print("  Use OPTIMIZED parameters - current tuning is best!")
    else:
        print("  Use CUSTOM parameters found by Optuna:")
        print(f"    python train_full_dataset_tuned.py \\")
        print(f"      --batch-size {trial.params.get('batch_size')} \\")
        print(f"      --learning-rate {trial.params.get('learning_rate'):.6f} \\")
        print(f"      --latent-dim {trial.params.get('latent_dim')} \\")
        print(f"      --weight-decay {trial.params.get('weight_decay'):.6f} \\")
        print(f"      --architecture {trial.params.get('architecture')}")
    
    print()

