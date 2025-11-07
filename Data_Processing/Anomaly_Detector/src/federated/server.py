"""
Federated Continual Learning Server

Coordinates federated learning across multiple clients over sequential experiences.
Implements the server-side logic for Algorithms 1 & 2 from the paper.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import copy

from src.federated.strategies import get_federated_strategy
from src.federated.client import FCLClient
from src.continual.metrics import ContinualLearningMetrics


class FCLServer:
    """
    Federated Continual Learning Server.

    Manages:
    - Global model aggregation
    - Communication rounds across experiences
    - Evaluation and metrics tracking
    """

    def __init__(
        self,
        global_model: nn.Module,
        fl_strategy: str = 'fedavg',
        fl_params: Optional[Dict] = None,
        device: str = 'cpu'
    ):
        """
        Initialize FCL server.

        Args:
            global_model: Global model architecture
            fl_strategy: Federated learning strategy ('fedavg', 'fedprox', 'scaffold')
            fl_params: Parameters for FL strategy
            device: Device to run on
        """
        self.global_model = global_model.to(device)
        self.device = device

        # Initialize FL strategy
        if fl_params is None:
            fl_params = {}

        self.fl_strategy = get_federated_strategy(fl_strategy, **fl_params)
        self.fl_strategy_name = fl_strategy

        # Metrics tracking
        self.training_history = {
            'experiences': [],
            'rounds': [],
            'avg_loss': [],
            'participating_clients': []
        }

    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get global model parameters.

        Returns:
            Dictionary of parameter tensors
        """
        return {
            name: param.data.clone()
            for name, param in self.global_model.state_dict().items()
        }

    def set_global_parameters(self, parameters: Dict[str, torch.Tensor]):
        """
        Set global model parameters.

        Args:
            parameters: Dictionary of parameter tensors
        """
        self.global_model.load_state_dict(parameters)

    def select_clients(
        self,
        clients: List[FCLClient],
        fraction: float = 1.0
    ) -> List[FCLClient]:
        """
        Select clients for current round.

        Args:
            clients: List of all clients
            fraction: Fraction of clients to select (1.0 = all)

        Returns:
            Selected clients
        """
        import random

        n_clients = max(1, int(len(clients) * fraction))

        if n_clients == len(clients):
            return clients
        else:
            return random.sample(clients, n_clients)

    def aggregate_clients(
        self,
        clients: List[FCLClient],
        experience_id: int
    ) -> None:
        """
        Aggregate client models into global model.

        Args:
            clients: List of client models to aggregate
            experience_id: Current experience ID
        """
        # Get client models and weights
        client_models = [client.model for client in clients]
        client_weights = [
            client.get_dataset_size(experience_id)
            for client in clients
        ]

        # Handle case where weights are all zero
        if sum(client_weights) == 0:
            client_weights = [1.0] * len(clients)

        # Aggregate based on FL strategy
        if self.fl_strategy_name == 'fedavg':
            self.global_model = self.fl_strategy.aggregate(
                self.global_model,
                client_models,
                client_weights
            )

        elif self.fl_strategy_name == 'fedprox':
            # FedProx uses same aggregation as FedAvg
            self.global_model = self.fl_strategy.aggregate(
                self.global_model,
                client_models,
                client_weights
            )

        elif self.fl_strategy_name == 'scaffold':
            # Scaffold needs additional info
            # For now, use simplified version (same as FedAvg)
            # Full Scaffold implementation requires tracking per-round updates
            self.global_model = self.fl_strategy.aggregate(
                self.global_model,
                client_models,
                client_weights
            )

    def train_one_round(
        self,
        clients: List[FCLClient],
        experience_id: int,
        categorical_indices: List[int],
        numerical_indices: List[int],
        local_epochs: int = 5,
        learning_rate: float = 0.001,
        client_fraction: float = 1.0,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Perform one round of federated training.

        Args:
            clients: List of clients
            experience_id: Current experience ID
            categorical_indices: Indices of categorical features
            numerical_indices: Indices of numerical features
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for local training
            client_fraction: Fraction of clients to select
            verbose: Whether to print progress

        Returns:
            Dictionary with round metrics
        """
        # Select clients for this round
        selected_clients = self.select_clients(clients, client_fraction)

        if verbose:
            print(f"    Round: {len(selected_clients)} clients selected")

        # Distribute global model to selected clients
        global_params = self.get_global_parameters()

        for client in selected_clients:
            client.set_parameters(global_params)

        # Local training on each client
        client_losses = []

        for client in selected_clients:
            # Get client's dataloader for this experience
            if experience_id not in client.experience_dataloaders:
                continue

            dataloader = client.experience_dataloaders[experience_id]

            # Train locally
            metrics = client.train_on_experience(
                experience_id,
                dataloader,
                categorical_indices,
                numerical_indices,
                n_epochs=local_epochs,
                learning_rate=learning_rate,
                verbose=False
            )

            client_losses.append(metrics['loss'])

        # Aggregate client models
        self.aggregate_clients(selected_clients, experience_id)

        # Compute average loss
        avg_loss = sum(client_losses) / len(client_losses) if client_losses else 0.0

        if verbose:
            print(f"    Round complete: avg_loss={avg_loss:.4f}")

        return {
            'avg_loss': avg_loss,
            'n_clients': len(selected_clients)
        }

    def train_experience(
        self,
        clients: List[FCLClient],
        experience_id: int,
        categorical_indices: List[int],
        numerical_indices: List[int],
        n_rounds: int = 5,
        local_epochs: int = 5,
        learning_rate: float = 0.001,
        client_fraction: float = 1.0,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train on one experience across multiple rounds.

        Args:
            clients: List of clients
            experience_id: Current experience ID
            categorical_indices: Indices of categorical features
            numerical_indices: Indices of numerical features
            n_rounds: Number of communication rounds
            local_epochs: Local epochs per round
            learning_rate: Learning rate
            client_fraction: Fraction of clients per round
            verbose: Whether to print progress

        Returns:
            Training history for this experience
        """
        if verbose:
            print(f"\n  Experience {experience_id}: Starting {n_rounds} rounds...")

        experience_history = {
            'rounds': [],
            'losses': []
        }

        for round_num in range(n_rounds):
            if verbose:
                print(f"  [Experience {experience_id}, Round {round_num + 1}/{n_rounds}]")

            metrics = self.train_one_round(
                clients,
                experience_id,
                categorical_indices,
                numerical_indices,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                client_fraction=client_fraction,
                verbose=verbose
            )

            experience_history['rounds'].append(round_num + 1)
            experience_history['losses'].append(metrics['avg_loss'])

            # Update global history
            self.training_history['experiences'].append(experience_id)
            self.training_history['rounds'].append(round_num + 1)
            self.training_history['avg_loss'].append(metrics['avg_loss'])
            self.training_history['participating_clients'].append(metrics['n_clients'])

        if verbose:
            print(f"  Experience {experience_id} complete!\n")

        return experience_history

    def evaluate_all_clients(
        self,
        clients: List[FCLClient],
        experience_id: int,
        categorical_indices: List[int],
        numerical_indices: List[int],
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate all clients on a specific experience.

        Args:
            clients: List of clients
            experience_id: Experience to evaluate on
            categorical_indices: Indices of categorical features
            numerical_indices: Indices of numerical features
            verbose: Whether to print progress

        Returns:
            Dictionary with evaluation metrics
        """
        client_losses = []

        for client in clients:
            # Distribute global model
            client.set_parameters(self.get_global_parameters())

            # Evaluate if client has this experience
            if experience_id in client.experience_dataloaders:
                loss = client.evaluate_on_experience(
                    experience_id,
                    categorical_indices=categorical_indices,
                    numerical_indices=numerical_indices
                )
                client_losses.append(loss)

        avg_loss = sum(client_losses) / len(client_losses) if client_losses else float('inf')

        if verbose:
            print(f"    Evaluation on Experience {experience_id}: avg_loss={avg_loss:.4f}")

        return {
            'avg_loss': avg_loss,
            'n_clients': len(client_losses)
        }

    def save_global_model(self, filepath: str):
        """
        Save global model.

        Args:
            filepath: Path to save model
        """
        torch.save({
            'model_state_dict': self.global_model.state_dict(),
            'fl_strategy': self.fl_strategy_name,
            'training_history': self.training_history
        }, filepath)

        print(f"Global model saved to {filepath}")

    def load_global_model(self, filepath: str):
        """
        Load global model.

        Args:
            filepath: Path to model checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])

        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']

        print(f"Global model loaded from {filepath}")
