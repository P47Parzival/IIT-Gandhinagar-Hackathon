"""
Learning without Forgetting (LwF) for Continual Learning

Based on Li & Hoiem (2017) "Learning without Forgetting"
Implements the LwF strategy from Section 4.3 of the paper.

LwF uses knowledge distillation to preserve the knowledge of previous tasks
by minimizing divergence between current and previous model outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import copy


class LearningWithoutForgetting:
    """
    Learning without Forgetting (LwF) strategy.

    Uses knowledge distillation to preserve old task knowledge:
    L_LwF = L_new + λ * D_KL(p_old || p_new)

    where:
    - L_new is the loss on new task
    - D_KL is KL divergence between old and new model outputs
    - λ is the distillation weight
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_lwf: float = 1.2,
        temperature: float = 2.0,
        device: str = 'cpu'
    ):
        """
        Initialize LwF.

        Args:
            model: The neural network model
            lambda_lwf: Weight for distillation loss (λ in paper = 1.2)
            temperature: Temperature for softening probability distributions
            device: Device to run on
        """
        self.model = model
        self.lambda_lwf = lambda_lwf
        self.temperature = temperature
        self.device = device

        # Store previous model for distillation
        self.prev_model = None

    def save_model_snapshot(self):
        """
        Save a snapshot of the current model as the 'previous model'.

        Called at the end of each experience to prepare for the next one.
        """
        print("    Saving model snapshot for LwF...")
        self.prev_model = copy.deepcopy(self.model)
        self.prev_model.eval()

        # Freeze previous model parameters
        for param in self.prev_model.parameters():
            param.requires_grad = False

        print("    ✓ Model snapshot saved")

    def compute_distillation_loss(
        self,
        x: torch.Tensor,
        new_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.

        For autoencoders, we use MSE between old and new reconstructions.
        For classification, we'd use KL divergence between softmax outputs.

        Args:
            x: Input samples
            new_output: Output from current model

        Returns:
            Distillation loss
        """
        if self.prev_model is None:
            return torch.tensor(0.0, device=self.device)

        # Get old model's output
        with torch.no_grad():
            old_output = self.prev_model(x)

        # For autoencoders: MSE between reconstructions
        distillation_loss = F.mse_loss(new_output, old_output.detach())

        # Apply temperature scaling (makes distributions softer)
        # For regression (autoencoder), temperature just scales the loss
        distillation_loss = distillation_loss * (self.temperature ** 2)

        return distillation_loss

    def train_step(
        self,
        batch_x: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        categorical_indices: List[int],
        numerical_indices: List[int]
    ) -> tuple:
        """
        Perform one training step with LwF.

        Args:
            batch_x: Input batch
            optimizer: Optimizer
            categorical_indices: Indices of categorical features
            numerical_indices: Indices of numerical features

        Returns:
            Tuple of (total_loss, task_loss, distillation_loss)
        """
        self.model.train()
        optimizer.zero_grad()

        # Forward pass
        reconstructed = self.model(batch_x)

        # Task loss (reconstruction loss)
        from src.models.autoencoder import combined_loss
        task_loss = combined_loss(
            batch_x,
            reconstructed,
            categorical_indices,
            numerical_indices
        )

        # Distillation loss
        distillation_loss = self.compute_distillation_loss(batch_x, reconstructed)

        # Total loss
        total_loss = task_loss + self.lambda_lwf * distillation_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        return total_loss.item(), task_loss.item(), distillation_loss.item()


class LwFWithAttention:
    """
    LwF variant with attention mechanism.

    Weights the distillation loss by importance (e.g., based on reconstruction error).
    Samples with low error on old task get higher weight in distillation.
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_lwf: float = 1.0,
        temperature: float = 2.0,
        device: str = 'cpu'
    ):
        """
        Initialize LwF with Attention.

        Args:
            model: The neural network model
            lambda_lwf: Weight for distillation loss
            temperature: Temperature for softening
            device: Device to run on
        """
        self.model = model
        self.lambda_lwf = lambda_lwf
        self.temperature = temperature
        self.device = device
        self.prev_model = None

    def save_model_snapshot(self):
        """Save model snapshot."""
        print("    Saving model snapshot for LwF (with attention)...")
        self.prev_model = copy.deepcopy(self.model)
        self.prev_model.eval()

        for param in self.prev_model.parameters():
            param.requires_grad = False

        print("    ✓ Model snapshot saved")

    def compute_attention_weights(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention weights based on old model's performance.

        Samples with low reconstruction error on old model get higher weight.

        Args:
            x: Input samples

        Returns:
            Attention weights (batch_size,)
        """
        if self.prev_model is None:
            return torch.ones(x.size(0), device=self.device)

        with torch.no_grad():
            old_output = self.prev_model(x)
            # Reconstruction error
            error = F.mse_loss(old_output, x, reduction='none').mean(dim=1)

            # Convert to weights: low error = high weight
            # Use softmax with negative error
            weights = F.softmax(-error / self.temperature, dim=0)

        return weights

    def train_step(
        self,
        batch_x: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        categorical_indices: List[int],
        numerical_indices: List[int]
    ) -> tuple:
        """
        Perform training step with attention-weighted LwF.

        Args:
            batch_x: Input batch
            optimizer: Optimizer
            categorical_indices: Indices of categorical features
            numerical_indices: Indices of numerical features

        Returns:
            Tuple of (total_loss, task_loss, distillation_loss)
        """
        self.model.train()
        optimizer.zero_grad()

        # Forward pass
        reconstructed = self.model(batch_x)

        # Task loss
        from src.models.autoencoder import combined_loss
        task_loss = combined_loss(
            batch_x,
            reconstructed,
            categorical_indices,
            numerical_indices
        )

        # Distillation loss with attention
        if self.prev_model is not None:
            with torch.no_grad():
                old_output = self.prev_model(batch_x)
                attention = self.compute_attention_weights(batch_x)

            # Weighted MSE
            diff = (reconstructed - old_output.detach()) ** 2
            diff_weighted = diff * attention.unsqueeze(1)
            distillation_loss = diff_weighted.mean() * (self.temperature ** 2)
        else:
            distillation_loss = torch.tensor(0.0, device=self.device)

        # Total loss
        total_loss = task_loss + self.lambda_lwf * distillation_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        return total_loss.item(), task_loss.item(), distillation_loss.item()


class AdaptiveLwF:
    """
    Adaptive LwF that adjusts lambda based on task similarity.

    If new task is very different, reduce distillation weight.
    If similar, increase weight to preserve more knowledge.
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_lwf_min: float = 0.1,
        lambda_lwf_max: float = 10.0,
        temperature: float = 2.0,
        device: str = 'cpu'
    ):
        """
        Initialize Adaptive LwF.

        Args:
            model: The neural network model
            lambda_lwf_min: Minimum distillation weight
            lambda_lwf_max: Maximum distillation weight
            temperature: Temperature for softening
            device: Device to run on
        """
        self.model = model
        self.lambda_lwf_min = lambda_lwf_min
        self.lambda_lwf_max = lambda_lwf_max
        self.lambda_lwf = lambda_lwf_min  # Start conservative
        self.temperature = temperature
        self.device = device
        self.prev_model = None

    def save_model_snapshot(self):
        """Save model snapshot."""
        print("    Saving model snapshot for Adaptive LwF...")
        self.prev_model = copy.deepcopy(self.model)
        self.prev_model.eval()

        for param in self.prev_model.parameters():
            param.requires_grad = False

        print("    ✓ Model snapshot saved")

    def estimate_task_similarity(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> float:
        """
        Estimate similarity between current and previous task.

        Uses average reconstruction error on new data with old model.
        Low error = similar tasks, high error = different tasks.

        Args:
            dataloader: Data loader for new task

        Returns:
            Similarity score (0 = different, 1 = very similar)
        """
        if self.prev_model is None:
            return 0.5

        self.prev_model.eval()
        total_error = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch_x, in dataloader:
                batch_x = batch_x.to(self.device)
                old_output = self.prev_model(batch_x)

                # Reconstruction error
                error = F.mse_loss(old_output, batch_x)
                total_error += error.item() * batch_x.size(0)
                n_samples += batch_x.size(0)

        avg_error = total_error / n_samples

        # Convert to similarity: low error = high similarity
        # Use sigmoid to map to [0, 1]
        similarity = torch.sigmoid(torch.tensor(-avg_error)).item()

        return similarity

    def adapt_lambda(
        self,
        dataloader: torch.utils.data.DataLoader
    ):
        """
        Adapt lambda based on task similarity.

        Args:
            dataloader: Data loader for new task
        """
        similarity = self.estimate_task_similarity(dataloader)

        # Linear interpolation: high similarity = high lambda
        self.lambda_lwf = (
            self.lambda_lwf_min +
            similarity * (self.lambda_lwf_max - self.lambda_lwf_min)
        )

        print(f"    Adaptive LwF: similarity={similarity:.3f}, λ={self.lambda_lwf:.3f}")

    def train_step(
        self,
        batch_x: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        categorical_indices: List[int],
        numerical_indices: List[int]
    ) -> tuple:
        """
        Perform training step with adaptive LwF.

        Args:
            batch_x: Input batch
            optimizer: Optimizer
            categorical_indices: Indices of categorical features
            numerical_indices: Indices of numerical features

        Returns:
            Tuple of (total_loss, task_loss, distillation_loss)
        """
        self.model.train()
        optimizer.zero_grad()

        # Forward pass
        reconstructed = self.model(batch_x)

        # Task loss
        from src.models.autoencoder import combined_loss
        task_loss = combined_loss(
            batch_x,
            reconstructed,
            categorical_indices,
            numerical_indices
        )

        # Distillation loss
        if self.prev_model is not None:
            with torch.no_grad():
                old_output = self.prev_model(batch_x)

            distillation_loss = F.mse_loss(reconstructed, old_output.detach())
            distillation_loss = distillation_loss * (self.temperature ** 2)
        else:
            distillation_loss = torch.tensor(0.0, device=self.device)

        # Total loss with adaptive lambda
        total_loss = task_loss + self.lambda_lwf * distillation_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        return total_loss.item(), task_loss.item(), distillation_loss.item()
