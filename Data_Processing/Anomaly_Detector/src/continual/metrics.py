"""
Evaluation Metrics for Continual Learning

Implements metrics from Section 5 of the paper:
- Average Precision (AP) for anomaly detection
- Backward Transfer (BWT) - measures catastrophic forgetting
- Forward Transfer (FWT) - measures positive knowledge transfer
- Average Accuracy across all experiences

Based on:
- Lopez-Paz & Ranzato (2017) "Gradient Episodic Memory for Continual Learning"
- The paper's experimental evaluation (Section 5)
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn


class ContinualLearningMetrics:
    """
    Compute and track continual learning metrics across experiences.

    Maintains a performance matrix R where R[i,j] is the performance
    on experience j after training on experience i.
    """

    def __init__(self, n_experiences: int):
        """
        Initialize metrics tracker.

        Args:
            n_experiences: Total number of experiences
        """
        self.n_experiences = n_experiences

        # Performance matrix: R[i,j] = performance on exp j after training on exp i
        # Using Average Precision (AP) as the performance metric
        self.performance_matrix = np.zeros((n_experiences, n_experiences))
        self.performance_matrix[:] = np.nan  # Initially unknown

    def update(
        self,
        current_experience: int,
        test_experience: int,
        performance: float
    ):
        """
        Update performance matrix.

        Args:
            current_experience: Experience number we just trained on (0-indexed)
            test_experience: Experience we're evaluating on (0-indexed)
            performance: Performance score (e.g., Average Precision)
        """
        self.performance_matrix[current_experience, test_experience] = performance

    def compute_average_accuracy(self, up_to_experience: int = None) -> float:
        """
        Compute average accuracy up to a given experience.

        A_t = (1/t) * Σ_{j=1}^{t} R[t,j]

        Args:
            up_to_experience: Compute up to this experience (None = all)

        Returns:
            Average accuracy
        """
        if up_to_experience is None:
            up_to_experience = self.n_experiences - 1

        # Get diagonal and lower triangular values up to current experience
        scores = []
        for i in range(up_to_experience + 1):
            for j in range(i + 1):
                score = self.performance_matrix[i, j]
                if not np.isnan(score):
                    scores.append(score)

        if len(scores) == 0:
            return 0.0

        return np.mean(scores)

    def compute_backward_transfer(self) -> float:
        """
        Compute Backward Transfer (BWT).

        Measures how much learning new experiences affects performance on old ones.
        Negative BWT indicates catastrophic forgetting.

        BWT = (1/(T-1)) * Σ_{i=1}^{T-1} (R[T,i] - R[i,i])

        where:
        - T is the total number of experiences
        - R[i,i] is performance on exp i right after training on it
        - R[T,i] is performance on exp i after training on all experiences

        Returns:
            BWT score (negative = forgetting, positive = backward transfer)
        """
        T = self.n_experiences
        bwt_sum = 0.0
        n_valid = 0

        for i in range(T - 1):
            # Performance on exp i right after training on it
            R_ii = self.performance_matrix[i, i]

            # Performance on exp i after training on all experiences
            R_Ti = self.performance_matrix[T - 1, i]

            if not np.isnan(R_ii) and not np.isnan(R_Ti):
                bwt_sum += (R_Ti - R_ii)
                n_valid += 1

        if n_valid == 0:
            return 0.0

        return bwt_sum / n_valid

    def compute_forward_transfer(self) -> float:
        """
        Compute Forward Transfer (FWT).

        Measures how much learning previous experiences helps with new ones.
        Positive FWT indicates beneficial knowledge transfer.

        FWT = (1/(T-1)) * Σ_{i=2}^{T} (R[i-1,i] - R_random[i])

        where:
        - R[i-1,i] is performance on exp i before training on it (after training on i-1)
        - R_random[i] is random baseline performance (we use R[0,i] as proxy)

        Returns:
            FWT score (positive = positive transfer, negative = negative transfer)
        """
        T = self.n_experiences
        fwt_sum = 0.0
        n_valid = 0

        for i in range(1, T):
            # Performance on exp i before training on it
            R_prev_i = self.performance_matrix[i - 1, i]

            # Use performance on exp i at very start as baseline
            # (or 0.5 as random baseline for binary classification)
            R_random_i = 0.5  # Random baseline for AP

            if not np.isnan(R_prev_i):
                fwt_sum += (R_prev_i - R_random_i)
                n_valid += 1

        if n_valid == 0:
            return 0.0

        return fwt_sum / n_valid

    def compute_forgetting_measure(self) -> float:
        r"""
        Compute average forgetting across all experiences.

        Forgetting_i = max_{j \in {i,...,T-1}} R[j,i] - R[T,i]

        Measures the maximum performance achieved on each experience minus
        the final performance.

        Returns:
            Average forgetting (0 = no forgetting, positive = forgetting occurred)
        """
        T = self.n_experiences
        forgetting_sum = 0.0
        n_valid = 0

        for i in range(T - 1):
            # Maximum performance achieved on exp i
            max_perf = np.nanmax(self.performance_matrix[i:, i])

            # Final performance on exp i
            final_perf = self.performance_matrix[T - 1, i]

            if not np.isnan(max_perf) and not np.isnan(final_perf):
                forgetting = max_perf - final_perf
                forgetting_sum += forgetting
                n_valid += 1

        if n_valid == 0:
            return 0.0

        return forgetting_sum / n_valid

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of all metrics.

        Returns:
            Dictionary with all computed metrics
        """
        return {
            'average_accuracy': self.compute_average_accuracy(),
            'backward_transfer': self.compute_backward_transfer(),
            'forward_transfer': self.compute_forward_transfer(),
            'forgetting': self.compute_forgetting_measure(),
        }

    def get_performance_matrix(self) -> np.ndarray:
        """Get the full performance matrix."""
        return self.performance_matrix.copy()

    def print_summary(self):
        """Print a formatted summary of metrics."""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("Continual Learning Metrics Summary")
        print("=" * 80)
        print(f"Average Accuracy:     {summary['average_accuracy']:.4f}")
        print(f"Backward Transfer:    {summary['backward_transfer']:.4f} {'(forgetting!)' if summary['backward_transfer'] < 0 else '(positive transfer!)'}")
        print(f"Forward Transfer:     {summary['forward_transfer']:.4f} {'(positive transfer!)' if summary['forward_transfer'] > 0 else '(negative transfer)'}")
        print(f"Average Forgetting:   {summary['forgetting']:.4f}")
        print("=" * 80)

        # Print performance matrix
        print("\nPerformance Matrix (rows=trained up to, cols=tested on):")
        print("  ", end="")
        for j in range(self.n_experiences):
            print(f"Exp{j:2d}", end="  ")
        print()

        for i in range(self.n_experiences):
            print(f"Exp{i:2d} ", end="")
            for j in range(self.n_experiences):
                val = self.performance_matrix[i, j]
                if np.isnan(val):
                    print("  -  ", end="  ")
                else:
                    print(f"{val:.3f}", end="  ")
            print()


def evaluate_anomaly_detection(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    labels: np.ndarray,
    categorical_indices: List[int],
    numerical_indices: List[int],
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate anomaly detection performance.

    Args:
        model: Trained autoencoder model
        dataloader: Test data loader
        labels: Ground truth labels (0=normal, 1=anomaly)
        categorical_indices: Indices of categorical features
        numerical_indices: Indices of numerical features
        device: Device to run on

    Returns:
        Dictionary with metrics: AP, precision, recall, F1
    """
    from src.models.autoencoder import AnomalyDetector

    model.eval()

    # Compute reconstruction errors
    all_errors = []

    with torch.no_grad():
        for batch_x, in dataloader:
            batch_x = batch_x.to(device)
            reconstructed = model(batch_x)

            # Compute error per sample
            from src.models.autoencoder import combined_loss
            for i in range(batch_x.size(0)):
                x_sample = batch_x[i:i+1]
                x_recon = reconstructed[i:i+1]

                error = combined_loss(
                    x_sample,
                    x_recon,
                    categorical_indices,
                    numerical_indices
                ).item()

                all_errors.append(error)

    errors = np.array(all_errors)

    # Compute metrics
    if len(np.unique(labels)) < 2:
        # No anomalies or all anomalies
        return {
            'average_precision': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

    # Average Precision (primary metric in paper)
    ap = average_precision_score(labels, errors)

    # Compute threshold (95th percentile)
    threshold = np.percentile(errors, 95)
    predictions = (errors > threshold).astype(int)

    # Precision, Recall, F1
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    return {
        'average_precision': ap,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold
    }


def compute_stability_plasticity_tradeoff(
    metrics: ContinualLearningMetrics
) -> Tuple[float, float]:
    """
    Compute stability-plasticity tradeoff.

    Stability: Ability to retain old knowledge (1 - forgetting)
    Plasticity: Ability to learn new knowledge (average accuracy on new tasks)

    Args:
        metrics: ContinualLearningMetrics instance

    Returns:
        Tuple of (stability, plasticity)
    """
    forgetting = metrics.compute_forgetting_measure()
    stability = max(0.0, 1.0 - forgetting)

    # Plasticity: average performance on diagonal (learning new tasks)
    diagonal_scores = []
    for i in range(metrics.n_experiences):
        score = metrics.performance_matrix[i, i]
        if not np.isnan(score):
            diagonal_scores.append(score)

    plasticity = np.mean(diagonal_scores) if len(diagonal_scores) > 0 else 0.0

    return stability, plasticity
