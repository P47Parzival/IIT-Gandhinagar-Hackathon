"""
Elastic Weight Consolidation (EWC) for Continual Learning

Based on Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks"
Implements the EWC strategy from Section 4.3 of the paper.

EWC penalizes changes to important weights (high Fisher information) when learning new tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import copy


class EWC:
    """
    Elastic Weight Consolidation strategy.

    Adds a quadratic penalty term to the loss that prevents important weights
    from changing too much during new task learning.

    L_EWC = L_task + (λ/2) * Σ_i F_i (θ_i - θ*_i)^2

    where:
    - L_task is the task loss (reconstruction loss for autoencoder)
    - F_i is the Fisher information for parameter i
    - θ*_i is the optimal parameter from previous task
    - λ is the regularization strength
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 500.0,
        device: str = 'cpu'
    ):
        """
        Initialize EWC.

        Args:
            model: The neural network model
            lambda_ewc: Regularization strength (λ in paper = 500, range: 100-5000)
            device: Device to run on
        """
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.device = device

        # Store Fisher information and optimal parameters for each experience
        self.fisher_dict = {}  # {experience_id: {param_name: fisher_matrix}}
        self.optpar_dict = {}  # {experience_id: {param_name: optimal_param}}

    def compute_fisher_information(
        self,
        dataloader: torch.utils.data.DataLoader,
        categorical_indices: List[int],
        numerical_indices: List[int],
        experience_id: int
    ):
        """
        Compute Fisher Information Matrix for current experience.

        Uses empirical Fisher: F_i ≈ E[(∂L/∂θ_i)^2]

        Args:
            dataloader: Data loader for current experience
            categorical_indices: Indices of categorical features
            numerical_indices: Indices of numerical features
            experience_id: Current experience identifier
        """
        print(f"    Computing Fisher Information for Experience {experience_id}...")

        self.model.eval()

        # Initialize Fisher information dictionary
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)

        # Compute gradients for each batch
        n_samples = 0

        for batch_x, in dataloader:
            batch_x = batch_x.to(self.device)
            n_samples += batch_x.size(0)

            # Zero gradients
            self.model.zero_grad()

            # Forward pass
            reconstructed = self.model(batch_x)

            # Compute loss (using combined loss as in autoencoder)
            from src.models.autoencoder import combined_loss
            loss = combined_loss(
                batch_x,
                reconstructed,
                categorical_indices,
                numerical_indices
            )

            # Backward pass to get gradients
            loss.backward()

            # Accumulate squared gradients (empirical Fisher)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2

        # Average over all samples
        # BUG FIX #62: Prevent division by zero if dataloader is empty
        if n_samples == 0:
            print("[WARNING] EWC: Empty dataloader, cannot compute Fisher information!")
            return
        
        for name in fisher.keys():
            fisher[name] /= n_samples
            # BUG FIX #62: Clip Fisher values to prevent overflow in penalty computation
            # Fisher values can become extremely large, causing Inf in ewc_penalty
            fisher[name] = torch.clamp(fisher[name], max=1e6)

        # Store Fisher information and optimal parameters
        self.fisher_dict[experience_id] = fisher
        self.optpar_dict[experience_id] = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        print(f"    ✓ Fisher Information computed ({len(fisher)} parameters)")

    def compute_ewc_penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty term.

        Penalty = (λ/2) * Σ_experiences Σ_params F_i (θ_i - θ*_i)^2

        Returns:
            EWC penalty as a scalar tensor
        """
        penalty = torch.tensor(0.0, device=self.device)

        # Sum penalties from all previous experiences
        for experience_id in self.fisher_dict.keys():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.fisher_dict[experience_id]:
                    fisher = self.fisher_dict[experience_id][name]
                    optpar = self.optpar_dict[experience_id][name]

                    # Add quadratic penalty: F * (θ - θ*)^2
                    penalty += (fisher * (param - optpar) ** 2).sum()

        # Multiply by λ/2
        penalty = (self.lambda_ewc / 2.0) * penalty
        
        # BUG FIX #62: Validate penalty is finite before returning
        if not torch.isfinite(penalty):
            print(f"[WARNING] EWC: Non-finite penalty detected, returning zero")
            return torch.tensor(0.0, device=self.device)

        return penalty

    def train_step(
        self,
        batch_x: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        categorical_indices: List[int],
        numerical_indices: List[int]
    ) -> float:
        """
        Perform one training step with EWC penalty.

        Args:
            batch_x: Input batch
            optimizer: Optimizer
            categorical_indices: Indices of categorical features
            numerical_indices: Indices of numerical features

        Returns:
            Total loss (task loss + EWC penalty)
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

        # EWC penalty
        ewc_penalty = self.compute_ewc_penalty()

        # Total loss
        total_loss = task_loss + ewc_penalty

        # Backward pass
        total_loss.backward()
        optimizer.step()

        return total_loss.item()


class OnlineEWC(EWC):
    """
    Online EWC variant that updates Fisher information incrementally.

    More memory efficient for long sequences of experiences.
    Instead of storing Fisher for each experience separately, it maintains
    a running average: F_new = γ * F_old + (1-γ) * F_current
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 500.0,
        gamma: float = 0.9,
        device: str = 'cpu'
    ):
        """
        Initialize Online EWC.

        Args:
            model: The neural network model
            lambda_ewc: Regularization strength
            gamma: Decay factor for Fisher averaging (0 < γ < 1)
            device: Device to run on
        """
        super().__init__(model, lambda_ewc, device)
        self.gamma = gamma

        # Online EWC uses single Fisher matrix (not per experience)
        self.fisher = None
        self.optpar = None

    def compute_fisher_information(
        self,
        dataloader: torch.utils.data.DataLoader,
        categorical_indices: List[int],
        numerical_indices: List[int],
        experience_id: int
    ):
        """
        Compute and update Fisher Information incrementally.

        F_new = γ * F_old + (1-γ) * F_current
        """
        print(f"    Computing Online Fisher Information for Experience {experience_id}...")

        self.model.eval()

        # Compute current Fisher
        current_fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                current_fisher[name] = torch.zeros_like(param.data)

        n_samples = 0
        for batch_x, in dataloader:
            batch_x = batch_x.to(self.device)
            n_samples += batch_x.size(0)

            self.model.zero_grad()
            reconstructed = self.model(batch_x)

            from src.models.autoencoder import combined_loss
            loss = combined_loss(
                batch_x,
                reconstructed,
                categorical_indices,
                numerical_indices
            )

            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    current_fisher[name] += param.grad.data ** 2

        # Average over samples
        # BUG FIX #62: Prevent division by zero if dataloader is empty
        if n_samples == 0:
            print("[WARNING] OnlineEWC: Empty dataloader, cannot compute Fisher information!")
            return
        
        for name in current_fisher.keys():
            current_fisher[name] /= n_samples
            # BUG FIX #62: Clip Fisher values to prevent overflow
            current_fisher[name] = torch.clamp(current_fisher[name], max=1e6)

        # Update Fisher with exponential moving average
        if self.fisher is None:
            # First experience
            self.fisher = current_fisher
        else:
            # Update: F_new = γ * F_old + (1-γ) * F_current
            for name in self.fisher.keys():
                self.fisher[name] = (
                    self.gamma * self.fisher[name] +
                    (1 - self.gamma) * current_fisher[name]
                )
                # BUG FIX #62: Clip accumulated Fisher to prevent long-term drift to Inf
                self.fisher[name] = torch.clamp(self.fisher[name], max=1e6)

        # Update optimal parameters
        self.optpar = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        print(f"    ✓ Online Fisher updated (γ={self.gamma})")

    def compute_ewc_penalty(self) -> torch.Tensor:
        """Compute EWC penalty using online Fisher."""
        if self.fisher is None:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher:
                fisher = self.fisher[name]
                optpar = self.optpar[name]
                penalty += (fisher * (param - optpar) ** 2).sum()

        penalty = (self.lambda_ewc / 2.0) * penalty
        
        # BUG FIX #62: Validate penalty is finite before returning
        if not torch.isfinite(penalty):
            print(f"[WARNING] OnlineEWC: Non-finite penalty detected, returning zero")
            return torch.tensor(0.0, device=self.device)

        return penalty
