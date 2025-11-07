"""
Federated Continual Learning Client

Implements a client that combines FL and CL strategies.
Each client represents one legal entity (e.g., Adani Power, Adani Ports).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import copy

from src.continual.strategies import EWC, ExperienceReplay, LearningWithoutForgetting


class FCLClient:
    """
    Federated Continual Learning Client.

    Combines:
    - Federated Learning (trains locally, shares parameters with server)
    - Continual Learning (learns sequentially across experiences without forgetting)
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        cl_strategy: str = 'ewc',
        cl_params: Optional[Dict] = None,
        device: str = 'cpu'
    ):
        """
        Initialize FCL client.

        Args:
            client_id: Unique identifier (e.g., 'Adani_Power')
            model: Local model instance
            cl_strategy: Continual learning strategy ('ewc', 'replay', 'lwf', 'joint')
            cl_params: Parameters for CL strategy
            device: Device to train on
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.cl_strategy_name = cl_strategy
        self.device = device

        # Current experience number
        self.current_experience = 0

        # Initialize continual learning strategy
        if cl_params is None:
            cl_params = {}

        if cl_strategy == 'ewc':
            lambda_ewc = cl_params.get('lambda_ewc', 1000.0)
            self.cl_strategy = EWC(self.model, lambda_ewc=lambda_ewc, device=device)

        elif cl_strategy == 'replay':
            buffer_size = cl_params.get('buffer_size', 1000)
            replay_batch_size = cl_params.get('replay_batch_size', 32)
            self.cl_strategy = ExperienceReplay(
                self.model,
                buffer_size=buffer_size,
                replay_batch_size=replay_batch_size,
                device=device
            )

        elif cl_strategy == 'lwf':
            lambda_lwf = cl_params.get('lambda_lwf', 1.0)
            temperature = cl_params.get('temperature', 2.0)
            self.cl_strategy = LearningWithoutForgetting(
                self.model,
                lambda_lwf=lambda_lwf,
                temperature=temperature,
                device=device
            )

        elif cl_strategy == 'joint':
            # Joint training (no continual learning, train on all data)
            self.cl_strategy = None

        else:
            raise ValueError(f"Unknown CL strategy: {cl_strategy}")

        # Store data loaders for each experience
        self.experience_dataloaders = {}

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get model parameters (for sending to server).

        Returns:
            Dictionary of parameter tensors
        """
        return {
            name: param.data.clone()
            for name, param in self.model.state_dict().items()
        }

    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """
        Set model parameters (from global model).

        Args:
            parameters: Dictionary of parameter tensors
        """
        self.model.load_state_dict(parameters)

    def train_on_experience(
        self,
        experience_id: int,
        dataloader: DataLoader,
        categorical_indices: List[int],
        numerical_indices: List[int],
        n_epochs: int = 5,
        learning_rate: float = 0.001,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Train on a single experience with continual learning.

        Args:
            experience_id: Experience identifier
            dataloader: Data loader for this experience
            categorical_indices: Indices of categorical features
            numerical_indices: Indices of numerical features
            n_epochs: Number of local epochs
            learning_rate: Learning rate
            verbose: Whether to print progress

        Returns:
            Dictionary with training metrics
        """
        if verbose:
            print(f"  [{self.client_id}] Training on Experience {experience_id}...")

        self.current_experience = experience_id

        # Store dataloader for this experience
        self.experience_dataloaders[experience_id] = dataloader

        # Initialize optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Prepare continual learning strategy
        if self.cl_strategy_name == 'replay' and experience_id > 0:
            # Populate replay buffer with current experience
            self.cl_strategy.populate_buffer(dataloader, experience_id)

        # Training loop
        total_loss = 0.0
        n_batches = 0

        for epoch in range(n_epochs):
            epoch_loss = 0.0

            for batch_x, in dataloader:
                batch_x = batch_x.to(self.device)

                # Train step depends on CL strategy
                if self.cl_strategy_name == 'ewc':
                    loss = self.cl_strategy.train_step(
                        batch_x,
                        optimizer,
                        categorical_indices,
                        numerical_indices
                    )

                elif self.cl_strategy_name == 'replay':
                    loss = self.cl_strategy.train_step(
                        batch_x,
                        optimizer,
                        categorical_indices,
                        numerical_indices,
                        replay_ratio=0.5
                    )

                elif self.cl_strategy_name == 'lwf':
                    total_loss_val, task_loss, distill_loss = self.cl_strategy.train_step(
                        batch_x,
                        optimizer,
                        categorical_indices,
                        numerical_indices
                    )
                    loss = total_loss_val

                else:  # joint or no CL
                    self.model.train()
                    optimizer.zero_grad()

                    reconstructed = self.model(batch_x)

                    from src.models.autoencoder import combined_loss
                    loss_tensor = combined_loss(
                        batch_x,
                        reconstructed,
                        categorical_indices,
                        numerical_indices
                    )

                    loss_tensor.backward()
                    optimizer.step()
                    loss = loss_tensor.item()

                epoch_loss += loss
                n_batches += 1

            total_loss += epoch_loss

        avg_loss = total_loss / max(n_batches, 1)

        # Post-experience updates for CL strategies
        if self.cl_strategy_name == 'ewc':
            # Compute Fisher information after training on this experience
            self.cl_strategy.compute_fisher_information(
                dataloader,
                categorical_indices,
                numerical_indices,
                experience_id
            )

        elif self.cl_strategy_name == 'lwf':
            # Save model snapshot for next experience
            self.cl_strategy.save_model_snapshot()

        if verbose:
            print(f"  [{self.client_id}] Experience {experience_id} complete: avg_loss={avg_loss:.4f}")

        return {
            'loss': avg_loss,
            'n_samples': len(dataloader.dataset),
            'n_batches': n_batches
        }

    def evaluate_on_experience(
        self,
        experience_id: int,
        dataloader: Optional[DataLoader] = None,
        categorical_indices: List[int] = None,
        numerical_indices: List[int] = None
    ) -> float:
        """
        Evaluate on a specific experience.

        Args:
            experience_id: Experience to evaluate on
            dataloader: Data loader (if None, uses stored one)
            categorical_indices: Indices of categorical features
            numerical_indices: Indices of numerical features

        Returns:
            Average loss on this experience
        """
        if dataloader is None:
            if experience_id not in self.experience_dataloaders:
                raise ValueError(f"No dataloader stored for experience {experience_id}")
            dataloader = self.experience_dataloaders[experience_id]

        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch_x, in dataloader:
                batch_x = batch_x.to(self.device)
                reconstructed = self.model(batch_x)

                from src.models.autoencoder import combined_loss
                loss = combined_loss(
                    batch_x,
                    reconstructed,
                    categorical_indices,
                    numerical_indices
                )

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        return avg_loss

    def get_dataset_size(self, experience_id: int) -> int:
        """
        Get number of samples for an experience.

        Args:
            experience_id: Experience identifier

        Returns:
            Number of samples
        """
        if experience_id in self.experience_dataloaders:
            return len(self.experience_dataloaders[experience_id].dataset)
        return 0

    def save_checkpoint(self, filepath: str):
        """
        Save client checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'client_id': self.client_id,
            'model_state_dict': self.model.state_dict(),
            'current_experience': self.current_experience,
            'cl_strategy': self.cl_strategy_name,
        }

        # Save CL strategy state
        if self.cl_strategy_name == 'ewc':
            checkpoint['fisher_dict'] = self.cl_strategy.fisher_dict
            checkpoint['optpar_dict'] = self.cl_strategy.optpar_dict

        elif self.cl_strategy_name == 'replay':
            checkpoint['replay_buffer'] = {
                'buffer_x': self.cl_strategy.buffer.buffer_x,
                'buffer_exp': self.cl_strategy.buffer.buffer_exp,
            }

        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        """
        Load client checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_experience = checkpoint['current_experience']

        # Restore CL strategy state
        if self.cl_strategy_name == 'ewc' and 'fisher_dict' in checkpoint:
            self.cl_strategy.fisher_dict = checkpoint['fisher_dict']
            self.cl_strategy.optpar_dict = checkpoint['optpar_dict']

        elif self.cl_strategy_name == 'replay' and 'replay_buffer' in checkpoint:
            self.cl_strategy.buffer.buffer_x = checkpoint['replay_buffer']['buffer_x']
            self.cl_strategy.buffer.buffer_exp = checkpoint['replay_buffer']['buffer_exp']
