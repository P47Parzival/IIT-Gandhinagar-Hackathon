"""
Experience Replay for Continual Learning

Stores samples from previous experiences and replays them during new task learning.
Implements the Replay strategy from Section 4.3 of the paper.

Replay prevents catastrophic forgetting by maintaining a memory buffer of past examples.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader


class ReplayBuffer:
    """
    Memory buffer for storing samples from previous experiences.

    Uses reservoir sampling to maintain a fixed-size buffer with
    uniform probability across all experiences.
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        device: str = 'cpu'
    ):
        """
        Initialize replay buffer.

        Args:
            buffer_size: Maximum number of samples to store
            device: Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.device = device

        # Buffer storage
        self.buffer_x = []  # Input samples
        self.buffer_exp = []  # Experience IDs

        self.n_seen_samples = 0  # Total samples seen (for reservoir sampling)

    def add_samples(
        self,
        x: torch.Tensor,
        experience_id: int
    ):
        """
        Add samples to buffer using reservoir sampling.

        Reservoir sampling ensures uniform probability across all experiences
        even when number of samples per experience varies.

        Args:
            x: Samples to add (batch_size, feature_dim)
            experience_id: Experience identifier
        """
        batch_size = x.size(0)

        for i in range(batch_size):
            sample = x[i].cpu()

            if len(self.buffer_x) < self.buffer_size:
                # Buffer not full, just add
                self.buffer_x.append(sample)
                self.buffer_exp.append(experience_id)
            else:
                # Reservoir sampling: replace with probability buffer_size / n_seen
                self.n_seen_samples += 1
                j = np.random.randint(0, self.n_seen_samples)

                if j < self.buffer_size:
                    self.buffer_x[j] = sample
                    self.buffer_exp[j] = experience_id

    def get_samples(
        self,
        batch_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from buffer.

        Args:
            batch_size: Number of samples to retrieve (None = all)

        Returns:
            Tuple of (samples, experience_ids)
        """
        if len(self.buffer_x) == 0:
            return None, None

        if batch_size is None or batch_size >= len(self.buffer_x):
            # Return all samples
            indices = range(len(self.buffer_x))
        else:
            # Sample random batch
            indices = np.random.choice(len(self.buffer_x), batch_size, replace=False)

        samples = torch.stack([self.buffer_x[i] for i in indices]).to(self.device)
        exp_ids = torch.tensor([self.buffer_exp[i] for i in indices]).to(self.device)

        return samples, exp_ids

    def __len__(self):
        return len(self.buffer_x)

    def get_buffer_stats(self):
        """Get statistics about buffer contents."""
        if len(self.buffer_exp) == 0:
            return {}

        exp_ids = np.array(self.buffer_exp)
        unique_exp = np.unique(exp_ids)

        stats = {
            'total_samples': len(self.buffer_x),
            'n_experiences': len(unique_exp),
            'samples_per_experience': {
                int(exp_id): int((exp_ids == exp_id).sum())
                for exp_id in unique_exp
            }
        }

        return stats


class ExperienceReplay:
    """
    Experience Replay continual learning strategy.

    Combines current experience data with replayed samples from buffer
    during training.
    """

    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 1000,
        replay_batch_size: int = 32,
        device: str = 'cpu'
    ):
        """
        Initialize Experience Replay.

        Args:
            model: The neural network model
            buffer_size: Size of replay buffer
            replay_batch_size: Batch size for replay samples
            device: Device to run on
        """
        self.model = model
        self.buffer = ReplayBuffer(buffer_size, device)
        self.replay_batch_size = replay_batch_size
        self.device = device

    def populate_buffer(
        self,
        dataloader: torch.utils.data.DataLoader,
        experience_id: int
    ):
        """
        Populate buffer with samples from current experience.

        Args:
            dataloader: Data loader for current experience
            experience_id: Current experience identifier
        """
        print(f"    Populating replay buffer for Experience {experience_id}...")

        for batch_x, in dataloader:
            batch_x = batch_x.to(self.device)
            self.buffer.add_samples(batch_x, experience_id)

        stats = self.buffer.get_buffer_stats()
        print(f"    ✓ Buffer populated: {stats['total_samples']} samples from {stats['n_experiences']} experiences")

    def train_step(
        self,
        batch_x: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        categorical_indices: List[int],
        numerical_indices: List[int],
        replay_ratio: float = 0.5
    ) -> float:
        """
        Perform one training step with experience replay.

        Combines current batch with replay batch.

        Args:
            batch_x: Current experience batch
            optimizer: Optimizer
            categorical_indices: Indices of categorical features
            numerical_indices: Indices of numerical features
            replay_ratio: Ratio of replay samples (0.5 = 50% current, 50% replay)

        Returns:
            Total loss
        """
        self.model.train()
        optimizer.zero_grad()

        # Current batch
        current_batch_size = batch_x.size(0)

        # Get replay samples
        replay_batch_size = int(current_batch_size * replay_ratio)

        if len(self.buffer) > 0 and replay_batch_size > 0:
            replay_x, _ = self.buffer.get_samples(replay_batch_size)

            if replay_x is not None:
                # Combine current and replay batches
                combined_x = torch.cat([batch_x, replay_x], dim=0)
            else:
                combined_x = batch_x
        else:
            combined_x = batch_x

        # Forward pass
        reconstructed = self.model(combined_x)

        # Compute loss
        from src.models.autoencoder import combined_loss
        loss = combined_loss(
            combined_x,
            reconstructed,
            categorical_indices,
            numerical_indices
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        return loss.item()


class GenerativeReplay:
    """
    Generative Replay variant that generates synthetic samples instead of storing real ones.

    Uses a generative model (e.g., VAE) to generate pseudo-samples from previous experiences.
    More privacy-preserving but requires training a separate generator.
    """

    def __init__(
        self,
        model: nn.Module,
        generator: Optional[nn.Module] = None,
        device: str = 'cpu'
    ):
        """
        Initialize Generative Replay.

        Args:
            model: The main model (autoencoder)
            generator: Generative model (if None, uses the autoencoder's decoder)
            device: Device to run on
        """
        self.model = model
        self.generator = generator if generator is not None else model
        self.device = device

    def generate_samples(
        self,
        n_samples: int,
        latent_dim: int = 32
    ) -> torch.Tensor:
        """
        Generate synthetic samples from latent space.

        Args:
            n_samples: Number of samples to generate
            latent_dim: Dimension of latent space

        Returns:
            Generated samples
        """
        self.generator.eval()

        with torch.no_grad():
            # Sample from latent space (assuming Gaussian)
            z = torch.randn(n_samples, latent_dim).to(self.device)

            # Decode to generate samples
            if hasattr(self.generator, 'decoder'):
                generated = self.generator.decoder(z)
            else:
                # If using full autoencoder, sample and decode
                generated = self.generator(z)

        return generated

    def train_step(
        self,
        batch_x: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        categorical_indices: List[int],
        numerical_indices: List[int],
        n_generated: int = 32
    ) -> float:
        """
        Perform training step with generated replay samples.

        Args:
            batch_x: Current batch
            optimizer: Optimizer
            categorical_indices: Indices of categorical features
            numerical_indices: Indices of numerical features
            n_generated: Number of samples to generate

        Returns:
            Total loss
        """
        self.model.train()
        optimizer.zero_grad()

        # Generate pseudo-samples
        generated_x = self.generate_samples(
            n_generated,
            latent_dim=32  # Should match model's latent dimension
        )

        # Combine with current batch
        combined_x = torch.cat([batch_x, generated_x], dim=0)

        # Forward pass
        reconstructed = self.model(combined_x)

        # Compute loss
        from src.models.autoencoder import combined_loss
        loss = combined_loss(
            combined_x,
            reconstructed,
            categorical_indices,
            numerical_indices
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        return loss.item()
