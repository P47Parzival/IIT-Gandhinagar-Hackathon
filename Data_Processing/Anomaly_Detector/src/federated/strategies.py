"""
Federated Learning Strategies

Implements FL aggregation strategies from Section 4.2 of the paper:
- FedAvg: Federated Averaging (McMahan et al., 2017)
- FedProx: Federated Proximal (Li et al., 2020)
- Scaffold: Stochastic Controlled Averaging (Karimireddy et al., 2020)

These strategies handle non-IID data distribution across clients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
import copy
import numpy as np


class FederatedAveraging:
    """
    FedAvg: Federated Averaging algorithm.

    Aggregates client models by weighted averaging based on dataset sizes.

    θ_global^{t+1} = Σ_i (n_i / n_total) * θ_client_i^{t+1}

    where n_i is the number of samples at client i.
    """

    def __init__(self):
        """Initialize FedAvg aggregator."""
        self.name = "FedAvg"

    def aggregate(
        self,
        global_model: nn.Module,
        client_models: List[nn.Module],
        client_weights: List[float]
    ) -> nn.Module:
        """
        Aggregate client models into global model.

        Args:
            global_model: Current global model
            client_models: List of client models
            client_weights: List of weights (typically proportional to dataset sizes)

        Returns:
            Updated global model
        """
        # Normalize weights
        total_weight = sum(client_weights)
        weights = [w / total_weight for w in client_weights]

        # Get global model state dict
        global_dict = global_model.state_dict()

        # Weighted average of client parameters
        for key in global_dict.keys():
            # Skip non-trainable parameters
            if not global_model.state_dict()[key].requires_grad:
                continue

            # Weighted sum
            weighted_sum = torch.zeros_like(global_dict[key])

            for client_model, weight in zip(client_models, weights):
                client_param = client_model.state_dict()[key]
                weighted_sum += weight * client_param

            global_dict[key] = weighted_sum

        # Load aggregated parameters
        global_model.load_state_dict(global_dict)

        return global_model


class FederatedProximal:
    """
    FedProx: Federated Proximal algorithm.

    Adds a proximal term to local objective to handle systems heterogeneity.

    Local objective at client i:
    L_i(θ) = F_i(θ) + (μ/2) * ||θ - θ_global||^2

    where:
    - F_i is the local loss function
    - μ is the proximal term coefficient
    - θ_global is the global model parameter
    """

    def __init__(self, mu: float = 1.2):
        """
        Initialize FedProx.

        Args:
            mu: Proximal term coefficient (paper uses 1.2, range: 0.001 to 1.5)
        """
        self.name = "FedProx"
        self.mu = mu

    def compute_proximal_loss(
        self,
        local_model: nn.Module,
        global_model: nn.Module
    ) -> torch.Tensor:
        """
        Compute proximal regularization term.

        Penalty = (μ/2) * ||θ_local - θ_global||^2

        Args:
            local_model: Current local model
            global_model: Global model from server

        Returns:
            Proximal penalty term
        """
        penalty = torch.tensor(0.0)

        for local_param, global_param in zip(
            local_model.parameters(),
            global_model.parameters()
        ):
            penalty += torch.norm(local_param - global_param.detach()) ** 2

        penalty = (self.mu / 2.0) * penalty

        return penalty

    def local_train_step(
        self,
        local_model: nn.Module,
        global_model: nn.Module,
        batch_x: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        categorical_indices: List[int],
        numerical_indices: List[int]
    ) -> float:
        """
        Perform one local training step with proximal term.

        Args:
            local_model: Local client model
            global_model: Global model (frozen)
            batch_x: Input batch
            optimizer: Optimizer for local model
            categorical_indices: Indices of categorical features
            numerical_indices: Indices of numerical features

        Returns:
            Total loss (task loss + proximal penalty)
        """
        local_model.train()
        optimizer.zero_grad()

        # Forward pass
        reconstructed = local_model(batch_x)

        # Task loss
        from src.models.autoencoder import combined_loss
        task_loss = combined_loss(
            batch_x,
            reconstructed,
            categorical_indices,
            numerical_indices
        )

        # Proximal penalty
        prox_penalty = self.compute_proximal_loss(local_model, global_model)

        # Total loss
        total_loss = task_loss + prox_penalty

        # Backward pass
        total_loss.backward()
        optimizer.step()

        return total_loss.item()

    def aggregate(
        self,
        global_model: nn.Module,
        client_models: List[nn.Module],
        client_weights: List[float]
    ) -> nn.Module:
        """
        Aggregate client models (same as FedAvg).

        Args:
            global_model: Current global model
            client_models: List of client models
            client_weights: List of weights

        Returns:
            Updated global model
        """
        # FedProx uses same aggregation as FedAvg
        fedavg = FederatedAveraging()
        return fedavg.aggregate(global_model, client_models, client_weights)


class Scaffold:
    """
    Scaffold: Stochastic Controlled Averaging for Federated Learning.

    Uses control variates to handle client drift due to data heterogeneity.

    Maintains control variates c_i for each client and c for server.
    Updates are corrected using: θ^{t+1} = θ^t - η * (∇F_i(θ) - c_i + c)

    More communication efficient than FedAvg for non-IID data.
    """

    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize Scaffold.

        Args:
            learning_rate: Global learning rate
        """
        self.name = "Scaffold"
        self.learning_rate = learning_rate

        # Server control variate (c)
        self.server_control = None

        # Client control variates (c_i for each client)
        self.client_controls = {}

    def initialize_controls(
        self,
        model: nn.Module,
        n_clients: int
    ):
        """
        Initialize control variates.

        Args:
            model: Model architecture (used to get parameter shapes)
            n_clients: Number of clients
        """
        # Initialize server control to zero
        self.server_control = {
            name: torch.zeros_like(param)
            for name, param in model.state_dict().items()
        }

        # Initialize client controls to zero
        for client_id in range(n_clients):
            self.client_controls[client_id] = {
                name: torch.zeros_like(param)
                for name, param in model.state_dict().items()
            }

    def compute_client_drift(
        self,
        client_id: int,
        initial_params: Dict[str, torch.Tensor],
        final_params: Dict[str, torch.Tensor],
        n_local_steps: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute client drift and update control variates.

        Args:
            client_id: Client identifier
            initial_params: Parameters before local training
            final_params: Parameters after local training
            n_local_steps: Number of local training steps

        Returns:
            Updated client control variate
        """
        if self.server_control is None:
            raise ValueError("Controls not initialized. Call initialize_controls() first.")

        # Compute option II update from Scaffold paper
        # Δc_i = (θ_global - θ_local^{t+1}) / (K * η) - c + c_i
        # where K is number of local steps, η is learning rate

        new_client_control = {}

        for name in initial_params.keys():
            theta_diff = initial_params[name] - final_params[name]

            # Simplified update: c_i^{new} = c_i - c + (θ_diff / (K * η))
            correction = theta_diff / (n_local_steps * self.learning_rate)

            new_client_control[name] = (
                self.client_controls[client_id][name] -
                self.server_control[name] +
                correction
            )

        return new_client_control

    def update_server_control(
        self,
        client_controls_delta: List[Dict[str, torch.Tensor]],
        n_clients: int
    ):
        """
        Update server control variate.

        c^{new} = c + (1/N) * Σ_i Δc_i

        Args:
            client_controls_delta: List of client control updates
            n_clients: Total number of clients
        """
        for name in self.server_control.keys():
            delta_sum = torch.zeros_like(self.server_control[name])

            for client_delta in client_controls_delta:
                if name in client_delta:
                    delta_sum += client_delta[name]

            self.server_control[name] += delta_sum / n_clients

    def local_train_step(
        self,
        local_model: nn.Module,
        client_id: int,
        batch_x: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        categorical_indices: List[int],
        numerical_indices: List[int]
    ) -> float:
        """
        Perform one local training step with Scaffold correction.

        Args:
            local_model: Local client model
            client_id: Client identifier
            batch_x: Input batch
            optimizer: Optimizer
            categorical_indices: Indices of categorical features
            numerical_indices: Indices of numerical features

        Returns:
            Loss value
        """
        local_model.train()
        optimizer.zero_grad()

        # Forward pass
        reconstructed = local_model(batch_x)

        # Compute loss
        from src.models.autoencoder import combined_loss
        loss = combined_loss(
            batch_x,
            reconstructed,
            categorical_indices,
            numerical_indices
        )

        # Backward pass
        loss.backward()

        # Apply Scaffold correction: gradient -= (c_i - c)
        with torch.no_grad():
            for name, param in local_model.named_parameters():
                if param.grad is not None and name in self.client_controls[client_id]:
                    correction = (
                        self.client_controls[client_id][name] -
                        self.server_control[name]
                    )
                    param.grad -= correction

        optimizer.step()

        return loss.item()

    def aggregate(
        self,
        global_model: nn.Module,
        client_models: List[nn.Module],
        client_weights: List[float],
        client_ids: List[int],
        initial_params: Dict[str, torch.Tensor],
        n_local_steps: int
    ) -> nn.Module:
        """
        Aggregate client models with Scaffold.

        Args:
            global_model: Current global model
            client_models: List of client models
            client_weights: List of weights
            client_ids: List of client IDs
            initial_params: Parameters before local training
            n_local_steps: Number of local training steps

        Returns:
            Updated global model
        """
        # Standard weighted averaging
        fedavg = FederatedAveraging()
        global_model = fedavg.aggregate(global_model, client_models, client_weights)

        # Update client and server controls
        client_controls_delta = []

        for client_id, client_model in zip(client_ids, client_models):
            # Compute new client control
            new_control = self.compute_client_drift(
                client_id,
                initial_params,
                client_model.state_dict(),
                n_local_steps
            )

            # Compute delta
            delta = {
                name: new_control[name] - self.client_controls[client_id][name]
                for name in new_control.keys()
            }

            client_controls_delta.append(delta)

            # Update client control
            self.client_controls[client_id] = new_control

        # Update server control
        self.update_server_control(client_controls_delta, len(client_models))

        return global_model


def get_federated_strategy(
    strategy_name: str,
    **kwargs
) -> object:
    """
    Factory function to get federated learning strategy.

    Args:
        strategy_name: Name of strategy ('fedavg', 'fedprox', 'scaffold')
        **kwargs: Strategy-specific parameters

    Returns:
        Federated learning strategy instance
    """
    strategy_name = strategy_name.lower()

    if strategy_name == 'fedavg':
        return FederatedAveraging()

    elif strategy_name == 'fedprox':
        mu = kwargs.get('mu', 0.01)
        return FederatedProximal(mu=mu)

    elif strategy_name == 'scaffold':
        learning_rate = kwargs.get('learning_rate', 0.01)
        return Scaffold(learning_rate=learning_rate)

    else:
        raise ValueError(f"Unknown federated strategy: {strategy_name}")
