"""
Autoencoder Network for GL Account Anomaly Detection
Based on research paper: "Federated Continual Learning to Detect Accounting Anomalies in Financial Auditing"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Import SPOT + ADWIN for adaptive thresholding
try:
    from .threshold_manager import ThresholdManager
except ImportError:
    # Handle case where threshold_manager not yet available
    ThresholdManager = None


class GLAutoencoder(nn.Module):
    """
    Autoencoder for detecting anomalies in GL account patterns.

    Implements the architecture from the paper with:
    - Symmetric encoder-decoder structure
    - LeakyReLU activations (α=0.4)
    - Tanh in bottleneck and output
    - Combined BCE + MSE loss for categorical and numerical features

    Args:
        input_dim: Dimension of input features (one-hot GL codes + numerical balances)
        latent_dim: Dimension of latent representation (default: 32)
        architecture: 'shallow' for global anomalies, 'deep' for local anomalies
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        architecture: str = 'deep'
    ):
        super(GLAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.architecture = architecture

        if architecture == 'shallow':
            # Shallow architecture for global anomalies (Table 2 in paper)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.LeakyReLU(0.4),
                nn.Linear(128, 64),
                nn.LeakyReLU(0.4),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.4),
                nn.Linear(32, 16),
                nn.LeakyReLU(0.4),
                nn.Linear(16, 8),
                nn.LeakyReLU(0.4),
                nn.Linear(8, 4),
                nn.LeakyReLU(0.4),
                nn.Linear(4, latent_dim),
                nn.Tanh()
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 4),
                nn.LeakyReLU(0.4),
                nn.Linear(4, 8),
                nn.LeakyReLU(0.4),
                nn.Linear(8, 16),
                nn.LeakyReLU(0.4),
                nn.Linear(16, 32),
                nn.LeakyReLU(0.4),
                nn.Linear(32, 64),
                nn.LeakyReLU(0.4),
                nn.Linear(64, 128),
                nn.LeakyReLU(0.4),
                nn.Linear(128, input_dim),
                nn.Sigmoid()  # Sigmoid for [0,1] normalized data (paper uses Tanh for [-1,1])
            )

        elif architecture == 'deep':
            # Deep architecture for local anomalies (Table 3 in paper)
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 2048),
                nn.LeakyReLU(0.4),
                nn.Linear(2048, 1024),
                nn.LeakyReLU(0.4),
                nn.Linear(1024, 512),
                nn.LeakyReLU(0.4),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.4),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.4),
                nn.Linear(128, 64),
                nn.LeakyReLU(0.4),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.4),
                nn.Linear(32, 16),
                nn.LeakyReLU(0.4),
                nn.Linear(16, 8),
                nn.LeakyReLU(0.4),
                nn.Linear(8, 4),
                nn.LeakyReLU(0.4),
                nn.Linear(4, latent_dim),
                nn.Tanh()
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 4),
                nn.LeakyReLU(0.4),
                nn.Linear(4, 8),
                nn.LeakyReLU(0.4),
                nn.Linear(8, 16),
                nn.LeakyReLU(0.4),
                nn.Linear(16, 32),
                nn.LeakyReLU(0.4),
                nn.Linear(32, 64),
                nn.LeakyReLU(0.4),
                nn.Linear(64, 128),
                nn.LeakyReLU(0.4),
                nn.Linear(128, 256),
                nn.LeakyReLU(0.4),
                nn.Linear(256, 512),
                nn.LeakyReLU(0.4),
                nn.Linear(512, 1024),
                nn.LeakyReLU(0.4),
                nn.Linear(1024, 2048),
                nn.LeakyReLU(0.4),
                nn.Linear(2048, input_dim),
                nn.Sigmoid()  # Sigmoid for [0,1] normalized data (paper uses Tanh for [-1,1])
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}. Choose 'shallow' or 'deep'.")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder."""
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed

    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation for input."""
        with torch.no_grad():
            return self.encode(x)


def combined_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    categorical_indices: List[int],
    numerical_indices: List[int],
    theta: float = 2/3
) -> torch.Tensor:
    """
    Combined Binary Cross Entropy (categorical) + MSE (numerical) loss.

    From paper Equation 4:
    L_Rec = θ * Σ L_BCE(categorical) + (1-θ) * Σ L_MSE(numerical)

    Args:
        x: Original input tensor [batch_size, input_dim]
        x_recon: Reconstructed input tensor [batch_size, input_dim]
        categorical_indices: Indices of one-hot encoded categorical features
        numerical_indices: Indices of numerical features
        theta: Balance parameter (default: 2/3 as per paper)

    Returns:
        Combined loss value
    """
    # Get device from input tensor to ensure consistency
    device = x.device
    
    # Validate input shapes match
    if x.shape != x_recon.shape:
        raise ValueError(f"Shape mismatch: input {x.shape} vs reconstruction {x_recon.shape}")
    
    # Validate indices are within bounds
    max_idx = x.shape[1] - 1
    if len(categorical_indices) > 0 and max(categorical_indices) > max_idx:
        raise ValueError(f"Categorical index {max(categorical_indices)} out of bounds for input dim {x.shape[1]}")
    if len(numerical_indices) > 0 and max(numerical_indices) > max_idx:
        raise ValueError(f"Numerical index {max(numerical_indices)} out of bounds for input dim {x.shape[1]}")
    
    # BCE for categorical GL codes, cost centers, etc.
    if len(categorical_indices) > 0:
        x_cat = x[:, categorical_indices]
        x_recon_cat = x_recon[:, categorical_indices]
        # Clamp to prevent log(0) which causes NaN/Inf in BCE
        x_recon_cat = torch.clamp(x_recon_cat, min=1e-7, max=1-1e-7)
        bce_loss = F.binary_cross_entropy(x_recon_cat, x_cat, reduction='mean')
    else:
        bce_loss = torch.tensor(0.0, device=device, dtype=x.dtype)

    # MSE for numerical GL balances
    if len(numerical_indices) > 0:
        x_num = x[:, numerical_indices]
        x_recon_num = x_recon[:, numerical_indices]
        mse_loss = F.mse_loss(x_recon_num, x_num, reduction='mean')
    else:
        mse_loss = torch.tensor(0.0, device=device, dtype=x.dtype)

    # Combined loss
    total_loss = theta * bce_loss + (1 - theta) * mse_loss

    return total_loss


class AnomalyDetector:
    """
    Anomaly detection using reconstruction error.

    Implements the anomaly detection approach from the paper where
    journal entries with high reconstruction error are flagged.
    """

    def __init__(
        self,
        model: GLAutoencoder,
        categorical_indices: List[int],
        numerical_indices: List[int],
        threshold_percentile: float = 95.0,
        theta: float = 2/3
    ):
        """
        Initialize anomaly detector.

        Args:
            model: Trained autoencoder model
            categorical_indices: Indices of categorical features
            numerical_indices: Indices of numerical features
            threshold_percentile: Percentile for anomaly threshold (default: 95)
            theta: Loss balance parameter
        """
        self.model = model
        self.categorical_indices = categorical_indices
        self.numerical_indices = numerical_indices
        self.threshold_percentile = threshold_percentile
        self.theta = theta
        self.threshold = None
        self.threshold_manager: Optional['ThresholdManager'] = None  # For SPOT-based thresholding

        self.model.eval()

    def compute_reconstruction_errors(
        self,
        data: torch.Tensor,
        batch_size: int = 1024
    ) -> np.ndarray:
        """
        Compute reconstruction error for each sample (optimized for GPU).

        Args:
            data: Input tensor [n_samples, input_dim]
            batch_size: Batch size for processing (default 1024 for GPU efficiency)

        Returns:
            Array of reconstruction errors [n_samples]
        """
        # Handle edge case: empty dataset
        if len(data) == 0:
            return np.array([])
        
        # Ensure model is in eval mode (no dropout, batchnorm in eval mode)
        self.model.eval()
        
        errors = []
        
        with torch.no_grad():
            # Verify no gradients are being tracked (safety check)
            assert not torch.is_grad_enabled(), "Gradients should be disabled during inference!"
            
            # Process in batches for GPU efficiency
            n_samples = len(data)
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                batch = data[i:batch_end]
                batch_recon = self.model(batch)
                
                # Compute loss for each sample in batch
                for j in range(len(batch)):
                    error = combined_loss(
                        batch[j:j+1],
                        batch_recon[j:j+1],
                        self.categorical_indices,
                        self.numerical_indices,
                        self.theta
                    )
                    # Extract value immediately and don't keep tensor reference
                    errors.append(float(error.item()))
                
                # Clear batch from GPU memory
                del batch, batch_recon
        
        # Force CUDA to release memory after all batches if on GPU
        if data.is_cuda:
            torch.cuda.empty_cache()

        return np.array(errors, dtype=np.float32)

    def fit_threshold(self, normal_data: torch.Tensor):
        """
        Compute anomaly threshold on normal data.

        Args:
            normal_data: Tensor of normal (non-anomalous) samples
        """
        if len(normal_data) == 0:
            raise ValueError("Cannot fit threshold on empty dataset")
        
        errors = self.compute_reconstruction_errors(normal_data)
        
        if len(errors) == 0 or np.all(np.isnan(errors)):
            raise ValueError("All reconstruction errors are NaN - check model and data")
        
        # Validate percentile range
        if not (0 <= self.threshold_percentile <= 100):
            raise ValueError(f"threshold_percentile must be in [0, 100], got {self.threshold_percentile}")
        
        self.threshold = np.percentile(errors, self.threshold_percentile)
        print(f"Anomaly threshold set to: {self.threshold:.4f}")

    def detect_anomalies(
        self,
        test_data: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in test data.

        Args:
            test_data: Input tensor to check for anomalies

        Returns:
            Tuple of (reconstruction_errors, anomaly_predictions as 0/1 integers)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call fit_threshold() first.")

        errors = self.compute_reconstruction_errors(test_data)
        anomalies = (errors > self.threshold).astype(np.int32)  # Convert bool to int

        return errors, anomalies

    def compute_average_precision(
        self,
        y_true: np.ndarray,
        scores: np.ndarray
    ) -> float:
        """
        Compute Average Precision metric as per paper (Equation 7).

        AP = Σ(R_i - R_{i-1}) * P_i

        Args:
            y_true: Binary labels (1 for anomaly, 0 for normal)
            scores: Anomaly scores (reconstruction errors)

        Returns:
            Average Precision score
        """
        from sklearn.metrics import average_precision_score
        return average_precision_score(y_true, scores)
    
    def fit_threshold_spot(self, data: torch.Tensor, entity_ids: np.ndarray) -> None:
        """
        Fit per-entity SPOT adaptive thresholds instead of global percentile.
        
        Uses Extreme Value Theory (EVT) to set statistically optimal thresholds
        that adapt per entity based on their error distribution tails.
        
        Args:
            data: Feature tensor [n_samples, input_dim]
            entity_ids: Array of entity identifiers (e.g., AGENCY codes) [n_samples]
        
        Raises:
            ImportError: If ThresholdManager not available
            ValueError: If data and entity_ids have mismatched lengths
        """
        if ThresholdManager is None:
            raise ImportError(
                "ThresholdManager not available. Ensure threshold_manager.py is in models/"
            )
        
        if len(data) != len(entity_ids):
            raise ValueError(
                f"Data and entity_ids length mismatch: {len(data)} != {len(entity_ids)}"
            )
        
        if len(data) == 0:
            raise ValueError("Cannot fit threshold on empty dataset")
        
        print("\n" + "="*80)
        print("FITTING SPOT ADAPTIVE THRESHOLDS (Per-Entity)")
        print("="*80)
        
        # Compute reconstruction errors
        errors = self.compute_reconstruction_errors(data)
        
        if len(errors) == 0 or np.all(np.isnan(errors)):
            raise ValueError("All reconstruction errors are NaN - check model and data")
        
        # Initialize threshold manager
        self.threshold_manager = ThresholdManager()
        
        # Group errors by entity and calibrate SPOT
        unique_entities = np.unique(entity_ids)
        n_entities = len(unique_entities)
        
        print(f"\nCalibrating SPOT for {n_entities} entities...")
        
        calibrated_count = 0
        fallback_count = 0
        
        for entity in unique_entities:
            entity_mask = (entity_ids == entity)
            entity_errors = errors[entity_mask]
            
            if len(entity_errors) >= 100:
                # Sufficient samples for SPOT calibration
                self.threshold_manager.calibrate_entity(str(entity), entity_errors)
                calibrated_count += 1
            else:
                # Too few samples - will use global fallback
                fallback_count += 1
        
        # Set global fallback threshold for low-sample entities
        if fallback_count > 0:
            global_threshold = np.percentile(errors, self.threshold_percentile)
            self.threshold_manager.set_global_threshold(global_threshold)
            print(f"\n[INFO] {fallback_count} entities have <100 samples")
            print(f"       Using global fallback threshold: {global_threshold:.4f}")
        
        print(f"\n[OK] SPOT calibrated for {calibrated_count} entities")
        print(f"     Fallback threshold for {fallback_count} low-sample entities")
        
        # Print sample thresholds
        summary = self.threshold_manager.get_entity_summary()
        sample_entities = list(summary['thresholds'].keys())[:5]
        if sample_entities:
            print(f"\nSample thresholds:")
            for entity in sample_entities:
                threshold = summary['thresholds'][entity]
                print(f"  Entity {entity}: {threshold:.4f}")
        
        print("="*80 + "\n")
    
    def predict_with_entity(
        self,
        data: torch.Tensor,
        entity_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies using per-entity SPOT adaptive thresholds.
        
        Includes ADWIN drift detection for defense against poisoning attacks.
        
        Args:
            data: Feature tensor [n_samples, input_dim]
            entity_ids: Array of entity identifiers [n_samples]
        
        Returns:
            Tuple of:
                - predictions: Anomaly flags (0/1) [n_samples]
                - drift_flags: Drift detection flags (bool) [n_samples]
        
        Raises:
            ValueError: If threshold_manager not fitted
        """
        if self.threshold_manager is None:
            raise ValueError(
                "Threshold manager not fitted. Call fit_threshold_spot() first."
            )
        
        if len(data) != len(entity_ids):
            raise ValueError(
                f"Data and entity_ids length mismatch: {len(data)} != {len(entity_ids)}"
            )
        
        # Compute reconstruction errors
        errors = self.compute_reconstruction_errors(data)
        
        # Initialize output arrays
        predictions = np.zeros(len(data), dtype=np.int32)
        drift_flags = np.zeros(len(data), dtype=bool)
        
        # OPTIMIZED: Process by entity using batch method
        unique_entities = np.unique(entity_ids)
        n_entities = len(unique_entities)
        
        print(f"[INFO] Predicting anomalies for {len(data):,} samples across {n_entities} entities...")
        
        processed = 0
        for entity_idx, entity in enumerate(unique_entities):
            # Get all samples for this entity
            entity_mask = (entity_ids == entity)
            entity_errors = errors[entity_mask]
            entity_indices = np.where(entity_mask)[0]
            
            # Batch process (5-10x faster, identical results)
            result = self.threshold_manager.check_anomaly_batch(str(entity), entity_errors)
            
            # Assign to output arrays
            predictions[entity_indices] = result['predictions']
            drift_flags[entity_indices] = result['drift_flags']
            
            processed += len(entity_errors)
            
            # Progress indicator every 10 entities
            if (entity_idx + 1) % 10 == 0 or entity_idx == n_entities - 1:
                pct = (processed / len(data)) * 100
                print(f"  Progress: {entity_idx + 1}/{n_entities} entities ({processed:,}/{len(data):,} samples, {pct:.1f}%)", end='\r')
        
        print()  # New line after progress
        
        return predictions, drift_flags
    
    def detect_anomalies_detailed(
        self,
        data: torch.Tensor,
        entity_ids: np.ndarray,
        original_df: 'pd.DataFrame'
    ) -> List[Dict]:
        """
        Enhanced detection with full threshold provenance for audit trails.
        
        Returns detailed records for each anomaly including:
        - Anomaly ID, entity, GL account, amount, error
        - Threshold snapshot: {q_alpha, delta, xi, beta, drift_flag, surge_mode}
        - SPOT parameters for reproducibility
        
        This enables:
        1. Audit trail - why was this flagged?
        2. Threshold debugging - what parameters were used?
        3. Attack forensics - was surge mode active?
        
        Args:
            data: Feature tensor [n_samples, input_dim]
            entity_ids: Array of entity identifiers [n_samples]
            original_df: Original dataframe with GL details (must have same row count)
        
        Returns:
            List of detailed anomaly records:
                {
                    'anomaly_id': str,
                    'entity_id': str,
                    'entity_name': str,
                    'gl_account': str,
                    'amount': float,
                    'reconstruction_error': float,
                    'threshold_snapshot': {
                        'q_alpha': float (SPOT threshold),
                        'initial_t': float (initial threshold),
                        'xi': float (GPD shape parameter),
                        'beta': float (GPD scale parameter),
                        'n_excesses': int (calibration data points),
                        'drift_flag': bool (ADWIN detected drift),
                        'surge_mode': bool (attack defense active),
                        'adt_delta': float (ADT adjustment if enabled)
                    },
                    'timestamp': str (ISO format)
                }
        
        Raises:
            ValueError: If data/entity_ids/original_df lengths don't match
        """
        if len(data) != len(entity_ids) or len(data) != len(original_df):
            raise ValueError(
                f"Length mismatch: data={len(data)}, entity_ids={len(entity_ids)}, "
                f"original_df={len(original_df)}"
            )
        
        # Run detection
        predictions, drift_flags = self.predict_with_entity(data, entity_ids)
        errors = self.compute_reconstruction_errors(data)
        
        # Collect detailed records for anomalies only
        detailed_records = []
        anomaly_indices = np.where(predictions == 1)[0]
        
        print(f"\n[INFO] Exporting {len(anomaly_indices)} anomalies with threshold provenance...")
        
        for idx in anomaly_indices:
            entity_id = str(entity_ids[idx])
            
            # Build threshold snapshot
            if entity_id in self.threshold_manager.entity_spots:
                spot = self.threshold_manager.entity_spots[entity_id]
                
                # Get ADT delta if enabled
                adt_delta = 0.0
                if self.threshold_manager.enable_adt and entity_id in self.threshold_manager.adt_controllers:
                    adt_delta = self.threshold_manager.adt_controllers[entity_id].current_delta
                
                threshold_snapshot = {
                    'q_alpha': float(spot.threshold),
                    'initial_t': float(spot.initial_threshold),
                    'xi': float(spot.xi) if spot.xi is not None else None,
                    'beta': float(spot.beta) if spot.beta is not None else None,
                    'n_excesses': int(len(spot.excesses)),
                    'drift_flag': bool(drift_flags[idx]),
                    'surge_mode': self.threshold_manager.surge_mode.get(entity_id, False),
                    'adt_delta': float(adt_delta),
                    'adt_enabled': self.threshold_manager.enable_adt
                }
            else:
                # Fallback to global threshold (handle None case)
                global_thresh = self.threshold_manager.global_threshold
                if global_thresh is None:
                    global_thresh = 0.1  # Default fallback if not set
                
                threshold_snapshot = {
                    'q_alpha': float(global_thresh),
                    'type': 'global_fallback',
                    'drift_flag': bool(drift_flags[idx]),
                    'surge_mode': False,
                    'adt_delta': 0.0,
                    'adt_enabled': False
                }
            
            # Extract GL details from original dataframe
            row = original_df.iloc[idx]
            
            record = {
                'anomaly_id': f"ANO_{datetime.now().strftime('%Y%m%d')}_{idx:06d}",
                'entity_id': entity_id,
                'entity_name': str(row.get('entity_name', 'UNKNOWN')),
                'gl_account': str(row.get('gl_account', 'UNKNOWN')),
                'amount': float(row.get('net_balance', 0.0)),
                'reconstruction_error': float(errors[idx]),
                'threshold_snapshot': threshold_snapshot,
                'timestamp': datetime.now().isoformat()
            }
            
            detailed_records.append(record)
        
        return detailed_records


def initialize_weights(model: nn.Module):
    """
    Initialize model weights using Xavier initialization.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


if __name__ == "__main__":
    # Test the autoencoder
    print("Testing GLAutoencoder...")

    # Create a sample autoencoder
    input_dim = 100  # Example: 80 categorical features + 20 numerical features
    model = GLAutoencoder(input_dim=input_dim, latent_dim=32, architecture='deep')
    initialize_weights(model)

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, input_dim)
    x_recon = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {x_recon.shape}")

    # Test loss computation
    categorical_indices = list(range(0, 80))
    numerical_indices = list(range(80, 100))
    loss = combined_loss(x, x_recon, categorical_indices, numerical_indices)

    print(f"Reconstruction loss: {loss.item():.4f}")

    # Test anomaly detector
    print("\nTesting AnomalyDetector...")
    detector = AnomalyDetector(
        model,
        categorical_indices,
        numerical_indices,
        threshold_percentile=95.0
    )

    # Fit threshold on normal data
    normal_data = torch.randn(1000, input_dim)
    detector.fit_threshold(normal_data)

    # Detect anomalies in test data
    test_data = torch.randn(100, input_dim)
    errors, anomalies = detector.detect_anomalies(test_data)

    print(f"Detected {anomalies.sum()} anomalies out of {len(test_data)} samples")
    print(f"Anomaly rate: {anomalies.mean() * 100:.2f}%")

    print("\n✓ Autoencoder tests passed!")
