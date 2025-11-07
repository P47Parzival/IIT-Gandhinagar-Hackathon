"""
Federated Continual Learning Training Script

Implements the complete FCL algorithm from the paper (Algorithms 1 & 2).

This script demonstrates the TRUE federated continual learning approach where:
1. Multiple clients (companies) train on sequential experiences (time periods)
2. Each client uses continual learning to prevent catastrophic forgetting
3. Clients share model updates (not raw data) via federated aggregation
4. Performance is measured across all past experiences to track forgetting

Usage:
    python scripts/train_federated_continual.py --cl_strategy ewc --fl_strategy fedavg
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict

from src.models.autoencoder import GLAutoencoder
from src.data.preprocessing import GLDataPreprocessor, create_gl_dataset, inject_global_anomalies
from src.federated.client import FCLClient
from src.federated.server import FCLServer
from src.continual.metrics import ContinualLearningMetrics, evaluate_anomaly_detection


def load_experience_data(
    entity_id: str,
    period: str,
    preprocessor: GLDataPreprocessor,
    batch_size: int = 32
) -> tuple:
    """
    Load and preprocess data for one experience.

    Args:
        entity_id: Entity identifier
        period: Period identifier (e.g., '2024-01')
        preprocessor: Fitted preprocessor
        batch_size: Batch size

    Returns:
        Tuple of (dataloader, labels)
    """
    data_file = f'data/raw/trial_balance_{entity_id}_{period}.csv'

    if not os.path.exists(data_file):
        return None, None

    # Load data
    df_raw = pd.read_csv(data_file)
    df = create_gl_dataset(df_raw, entity_id, period)

    # Inject anomalies for testing
    df_anomalous, labels = inject_global_anomalies(df, n_anomalies=20, random_state=42)

    # Preprocess
    X = preprocessor.transform(df_anomalous)

    # Create dataloader
    dataset = TensorDataset(torch.FloatTensor(X))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, labels


def main():
    parser = argparse.ArgumentParser(description='Federated Continual Learning Training')

    # Experiment configuration
    parser.add_argument('--n_clients', type=int, default=5,
                       help='Number of clients (companies)')
    parser.add_argument('--n_experiences', type=int, default=3,
                       help='Number of experiences (time periods)')

    # FL configuration
    parser.add_argument('--fl_strategy', type=str, default='fedavg',
                       choices=['fedavg', 'fedprox', 'scaffold'],
                       help='Federated learning strategy')
    parser.add_argument('--n_rounds', type=int, default=5,
                       help='Communication rounds per experience')
    parser.add_argument('--client_fraction', type=float, default=1.0,
                       help='Fraction of clients per round')

    # CL configuration
    parser.add_argument('--cl_strategy', type=str, default='ewc',
                       choices=['ewc', 'replay', 'lwf', 'joint', 'naive'],
                       help='Continual learning strategy')
    parser.add_argument('--lambda_ewc', type=float, default=500.0,
                       help='EWC regularization strength (paper: 500)')
    parser.add_argument('--buffer_size', type=int, default=1000,
                       help='Replay buffer size (paper: 1000)')
    parser.add_argument('--lambda_lwf', type=float, default=1.2,
                       help='LwF distillation weight (paper: 1.2)')

    # Training configuration
    parser.add_argument('--local_epochs', type=int, default=5,
                       help='Local training epochs per round')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size (paper: 16)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')

    # Model configuration
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent dimension')
    parser.add_argument('--architecture', type=str, default='deep',
                       choices=['shallow', 'deep'],
                       help='Autoencoder architecture')

    # Other
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to train on')
    parser.add_argument('--output_dir', type=str, default='data/fcl_results',
                       help='Output directory for results')

    args = parser.parse_args()

    print("=" * 80)
    print("FEDERATED CONTINUAL LEARNING - Full Paper Implementation")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Clients: {args.n_clients}")
    print(f"  Experiences: {args.n_experiences}")
    print(f"  FL Strategy: {args.fl_strategy}")
    print(f"  CL Strategy: {args.cl_strategy}")
    print(f"  Communication Rounds: {args.n_rounds}")
    print(f"  Local Epochs: {args.local_epochs}")
    print(f"  Architecture: {args.architecture}")
    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================================================
    # STEP 1: Prepare Data and Preprocessor
    # ========================================================================

    print("STEP 1: Preparing Data")
    print("-" * 80)

    # Entity and period configuration
    entities = [f'E{i+1:03d}' for i in range(args.n_clients)]
    periods = ['2024-01', '2024-02', '2024-03'][:args.n_experiences]

    print(f"Entities: {entities}")
    print(f"Periods: {periods}")

    # Fit preprocessor on first entity's first experience
    data_file = f'data/raw/trial_balance_{entities[0]}_{periods[0]}.csv'

    if not os.path.exists(data_file):
        print(f"\nError: Sample data not found!")
        print("Please run: python scripts/generate_sample_data.py")
        return

    df_raw = pd.read_csv(data_file)
    df = create_gl_dataset(df_raw, entities[0], periods[0])

    categorical_cols = ['gl_account', 'cost_center', 'profit_center']
    numerical_cols = ['debit_amount', 'credit_amount', 'debit_credit_ratio',
                     'net_balance', 'abs_balance']

    preprocessor = GLDataPreprocessor(categorical_cols, numerical_cols)
    X_sample = preprocessor.fit_transform(df)
    input_dim = X_sample.shape[1]

    print(f"\nPreprocessor fitted:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Categorical features: {len(preprocessor.categorical_indices)}")
    print(f"  Numerical features: {len(preprocessor.numerical_indices)}")

    # ========================================================================
    # STEP 2: Initialize Global Model and Server
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 2: Initializing Global Model and Server")
    print("-" * 80)

    global_model = GLAutoencoder(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        architecture=args.architecture
    )

    fl_params = {}
    if args.fl_strategy == 'fedprox':
        fl_params['mu'] = 1.2  # Paper specification (Appendix A.5)

    server = FCLServer(
        global_model=global_model,
        fl_strategy=args.fl_strategy,
        fl_params=fl_params,
        device=args.device
    )

    print(f"Global model created:")
    print(f"  Parameters: {sum(p.numel() for p in global_model.parameters()):,}")
    print(f"  FL Strategy: {args.fl_strategy}")

    # ========================================================================
    # STEP 3: Initialize Clients
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 3: Initializing Clients")
    print("-" * 80)

    clients = []
    cl_params = {}

    if args.cl_strategy == 'ewc':
        cl_params['lambda_ewc'] = args.lambda_ewc
    elif args.cl_strategy == 'replay':
        cl_params['buffer_size'] = args.buffer_size
        cl_params['replay_batch_size'] = args.batch_size
    elif args.cl_strategy == 'lwf':
        cl_params['lambda_lwf'] = args.lambda_lwf

    for entity_id in entities:
        # Create client with its own model copy
        client_model = GLAutoencoder(
            input_dim=input_dim,
            latent_dim=args.latent_dim,
            architecture=args.architecture
        )

        client = FCLClient(
            client_id=entity_id,
            model=client_model,
            cl_strategy=args.cl_strategy,
            cl_params=cl_params,
            device=args.device
        )

        # Load data for all experiences
        for exp_id, period in enumerate(periods):
            dataloader, labels = load_experience_data(
                entity_id,
                period,
                preprocessor,
                batch_size=args.batch_size
            )

            if dataloader is not None:
                client.experience_dataloaders[exp_id] = dataloader

        clients.append(client)
        print(f"  Created client: {entity_id} ({len(client.experience_dataloaders)} experiences)")

    # ========================================================================
    # STEP 4: Initialize Metrics Tracker
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 4: Initializing Metrics Tracker")
    print("-" * 80)

    metrics_tracker = ContinualLearningMetrics(n_experiences=args.n_experiences)

    print(f"Metrics tracker initialized for {args.n_experiences} experiences")
    print("  Will track: Average Accuracy, BWT, FWT, Forgetting")

    # ========================================================================
    # STEP 5: Federated Continual Learning Training Loop
    # (This is Algorithm 1 & 2 from the paper)
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 5: FEDERATED CONTINUAL LEARNING TRAINING")
    print("=" * 80)

    for exp_id, period in enumerate(periods):
        print(f"\n{'='*80}")
        print(f"EXPERIENCE {exp_id} (Period: {period})")
        print(f"{'='*80}\n")

        # Train on current experience (Algorithm 1: Federated Learning)
        server.train_experience(
            clients=clients,
            experience_id=exp_id,
            categorical_indices=preprocessor.categorical_indices,
            numerical_indices=preprocessor.numerical_indices,
            n_rounds=args.n_rounds,
            local_epochs=args.local_epochs,
            learning_rate=args.learning_rate,
            client_fraction=args.client_fraction,
            verbose=True
        )

        # Evaluate on ALL past experiences (Algorithm 2: Continual Evaluation)
        print(f"\n  Evaluating on all experiences up to {exp_id}...")
        print("  " + "-" * 76)

        for test_exp_id in range(exp_id + 1):
            test_period = periods[test_exp_id]

            # Evaluate each client
            all_ap_scores = []

            for client in clients:
                if test_exp_id not in client.experience_dataloaders:
                    continue

                # Distribute global model to client
                client.set_parameters(server.get_global_parameters())

                # Get test data
                dataloader, labels = load_experience_data(
                    client.client_id,
                    test_period,
                    preprocessor,
                    batch_size=args.batch_size
                )

                if dataloader is None:
                    continue

                # Evaluate anomaly detection
                results = evaluate_anomaly_detection(
                    client.model,
                    dataloader,
                    labels,
                    preprocessor.categorical_indices,
                    preprocessor.numerical_indices,
                    device=args.device
                )

                all_ap_scores.append(results['average_precision'])

            # Average AP across clients
            avg_ap = np.mean(all_ap_scores) if all_ap_scores else 0.0

            # Update metrics tracker
            metrics_tracker.update(exp_id, test_exp_id, avg_ap)

            print(f"    Experience {test_exp_id} (Period {test_period}): AP = {avg_ap:.4f}")

    # ========================================================================
    # STEP 6: Compute and Display Final Metrics
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 6: FINAL RESULTS")
    print("=" * 80)

    # Print metrics summary
    metrics_tracker.print_summary()

    # Save results
    results = {
        'config': vars(args),
        'metrics': metrics_tracker.get_summary(),
        'performance_matrix': metrics_tracker.get_performance_matrix().tolist(),
        'training_history': server.training_history
    }

    import json
    results_file = os.path.join(args.output_dir,
                               f'fcl_results_{args.fl_strategy}_{args.cl_strategy}.json')

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Save global model
    model_file = os.path.join(args.output_dir,
                             f'global_model_{args.fl_strategy}_{args.cl_strategy}.pth')
    server.save_global_model(model_file)

    # ========================================================================
    # STEP 7: Interpretation and Recommendations
    # ========================================================================

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    summary = metrics_tracker.get_summary()

    print("\nContinual Learning Performance:")
    print(f"  Backward Transfer (BWT): {summary['backward_transfer']:.4f}")

    if summary['backward_transfer'] < -0.1:
        print("    ⚠️  Significant catastrophic forgetting detected!")
        print("    Recommendation: Increase λ_EWC or buffer size")
    elif summary['backward_transfer'] < 0:
        print("    ⚡ Minor forgetting (acceptable)")
    else:
        print("    ✓  Positive backward transfer (learning improves old tasks!)")

    print(f"\n  Forward Transfer (FWT): {summary['forward_transfer']:.4f}")

    if summary['forward_transfer'] > 0:
        print("    ✓  Positive knowledge transfer to new tasks!")
    else:
        print("    ⚠️  No positive transfer (tasks might be too different)")

    print(f"\n  Average Accuracy: {summary['average_accuracy']:.4f}")
    print(f"  Average Forgetting: {summary['forgetting']:.4f}")

    print("\nFederated Learning Performance:")
    print(f"  Strategy: {args.fl_strategy}")
    print(f"  Final Round Loss: {server.training_history['avg_loss'][-1]:.4f}")

    print("\n" + "=" * 80)
    print("✓ FEDERATED CONTINUAL LEARNING TRAINING COMPLETE!")
    print("=" * 80)

    print("\nNext Steps:")
    print("  1. Visualize results: streamlit run scripts/simple_dashboard.py")
    print("  2. Try different strategies:")
    print("     - python scripts/train_federated_continual.py --cl_strategy replay")
    print("     - python scripts/train_federated_continual.py --fl_strategy fedprox")
    print("  3. Scale to more clients/experiences:")
    print("     - python scripts/train_federated_continual.py --n_clients 10 --n_experiences 5")


if __name__ == "__main__":
    main()
