"""
Data Preprocessing for GL Account Data
Handles transformation of trial balance data into model-ready format
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from typing import Dict, List, Tuple, Optional
import pickle


class GLDataPreprocessor:
    """
    Preprocess GL account data for autoencoder model.

    Handles:
    - Categorical encoding (GL codes, cost centers, etc.)
    - Numerical scaling (balances, amounts)
    - Feature extraction
    - Train/test split
    """

    def __init__(
        self,
        categorical_columns: List[str],
        numerical_columns: List[str],
        scaler_type: str = 'minmax'
    ):
        """
        Initialize preprocessor.

        Args:
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            scaler_type: Type of scaler ('minmax' or 'standard')
        """
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.scaler_type = scaler_type

        # Encoders and scalers (fitted during fit())
        self.encoders = {}  # One encoder per categorical column
        self.scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()

        # Feature indices (set during fit())
        self.categorical_indices = []
        self.numerical_indices = []
        self.feature_names = []

        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> 'GLDataPreprocessor':
        """
        Fit encoders and scalers on training data.

        Args:
            df: Training data DataFrame

        Returns:
            self
        """
        print("Fitting preprocessor...")

        # Fit one-hot encoders for categorical columns
        current_idx = 0
        for col in self.categorical_columns:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoder.fit(df[[col]])
            self.encoders[col] = encoder

            # Track indices for this categorical feature
            n_categories = len(encoder.categories_[0])
            self.categorical_indices.extend(range(current_idx, current_idx + n_categories))
            self.feature_names.extend([f"{col}_{cat}" for cat in encoder.categories_[0]])
            current_idx += n_categories

        # Fit scaler for numerical columns
        self.scaler.fit(df[self.numerical_columns])
        n_numerical = len(self.numerical_columns)
        self.numerical_indices = list(range(current_idx, current_idx + n_numerical))
        self.feature_names.extend(self.numerical_columns)
        
        # Check for constant features (would cause NaN with MinMaxScaler)
        if hasattr(self.scaler, 'data_range_'):
            constant_features = (self.scaler.data_range_ == 0)
            if constant_features.any():
                const_cols = [self.numerical_columns[i] for i, is_const in enumerate(constant_features) if is_const]
                print(f"  [WARNING] Constant numerical features detected: {const_cols}")
                print(f"            These will be scaled to 0 (may affect model performance)")

        self.is_fitted = True
        print(f"Preprocessor fitted:")
        print(f"  - {len(self.categorical_columns)} categorical columns -> {len(self.categorical_indices)} features")
        print(f"  - {len(self.numerical_columns)} numerical columns -> {len(self.numerical_indices)} features")
        print(f"  - Total features: {len(self.feature_names)}")

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted encoders and scalers.

        Args:
            df: Data to transform

        Returns:
            Transformed numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        features = []

        # Encode categorical columns
        for col in self.categorical_columns:
            encoded = self.encoders[col].transform(df[[col]])
            features.append(encoded)

        # Scale numerical columns
        scaled = self.scaler.transform(df[self.numerical_columns])
        features.append(scaled)

        # Concatenate all features
        X = np.hstack(features)
        
        # Ensure float32 for PyTorch compatibility (default hstack might give float64)
        X = X.astype(np.float32)

        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            df: Training data DataFrame

        Returns:
            Transformed numpy array
        """
        self.fit(df)
        return self.transform(df)

    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        """
        Inverse transform to original feature space.

        Args:
            X: Transformed data

        Returns:
            DataFrame with original columns
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted.")

        df_reconstructed = pd.DataFrame()

        # Inverse transform categorical columns
        current_idx = 0
        for col in self.categorical_columns:
            n_categories = len(self.encoders[col].categories_[0])
            encoded = X[:, current_idx:current_idx + n_categories]
            decoded = self.encoders[col].inverse_transform(encoded)
            df_reconstructed[col] = decoded.flatten()
            current_idx += n_categories

        # Inverse transform numerical columns
        n_numerical = len(self.numerical_columns)
        scaled = X[:, current_idx:current_idx + n_numerical]
        unscaled = self.scaler.inverse_transform(scaled)
        for i, col in enumerate(self.numerical_columns):
            df_reconstructed[col] = unscaled[:, i]

        return df_reconstructed

    def save(self, filepath: str):
        """Save fitted preprocessor to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'GLDataPreprocessor':
        """Load fitted preprocessor from file."""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor


def create_gl_dataset(
    df: pd.DataFrame,
    entity_id: str,
    period: str
) -> pd.DataFrame:
    """
    Create a GL dataset from trial balance data.

    Args:
        df: Raw trial balance data
        entity_id: Entity identifier
        period: Reporting period (e.g., '2024-01')

    Returns:
        Preprocessed DataFrame ready for model training
    """
    # Filter for specific entity and period
    df_filtered = df[
        (df['entity_id'] == entity_id) &
        (df['period'] == period)
    ].copy()

    # Feature engineering
    df_filtered['debit_credit_ratio'] = (
        df_filtered['debit_amount'] / (df_filtered['credit_amount'] + 1e-6)
    )
    df_filtered['net_balance'] = df_filtered['debit_amount'] - df_filtered['credit_amount']
    df_filtered['abs_balance'] = np.abs(df_filtered['net_balance'])

    return df_filtered


def inject_global_anomalies(
    df: pd.DataFrame,
    n_anomalies: int = 20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Inject global anomalies (unusual individual attribute values).

    Examples:
    - Extremely high/low GL balances
    - Rare GL account usage
    - Unusual posting times

    Args:
        df: Clean data
        n_anomalies: Number of anomalies to inject
        random_state: Random seed

    Returns:
        Tuple of (df_with_anomalies, anomaly_labels)
    """
    np.random.seed(random_state)
    df_anomalous = df.copy()
    labels = np.zeros(len(df))

    # Sample rows to modify
    anomaly_indices = np.random.choice(len(df), size=n_anomalies, replace=False)

    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['high_balance', 'low_balance', 'rare_gl', 'unusual_time'])

        if anomaly_type == 'high_balance':
            # Make balance 10-100x higher
            multiplier = np.random.uniform(10, 100)
            df_anomalous.loc[idx, 'net_balance'] *= multiplier
            df_anomalous.loc[idx, 'abs_balance'] *= multiplier

        elif anomaly_type == 'low_balance':
            # Make balance very small
            df_anomalous.loc[idx, 'net_balance'] *= 0.01

        elif anomaly_type == 'rare_gl':
            # Use a rare GL account code (create synthetic code)
            df_anomalous.loc[idx, 'gl_account'] = f"99999{np.random.randint(100)}"

        elif anomaly_type == 'unusual_time':
            # Post on unusual date (e.g., weekend, holiday)
            df_anomalous.loc[idx, 'posting_date'] = '2024-12-25'  # Christmas

        labels[idx] = 1  # Mark as anomaly

    print(f"Injected {n_anomalies} global anomalies")
    return df_anomalous, labels


def inject_local_anomalies(
    df: pd.DataFrame,
    n_anomalies: int = 20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Inject local anomalies (unusual attribute correlations).

    Examples:
    - Unusual GL account + cost center combinations
    - Mismatched debit/credit pairs
    - Inconsistent amount correlations

    Args:
        df: Clean data
        n_anomalies: Number of anomalies to inject
        random_state: Random seed

    Returns:
        Tuple of (df_with_anomalies, anomaly_labels)
    """
    np.random.seed(random_state)
    df_anomalous = df.copy()
    labels = np.zeros(len(df))

    anomaly_indices = np.random.choice(len(df), size=n_anomalies, replace=False)

    for idx in anomaly_indices:
        anomaly_type = np.random.choice(['swap_gl_cc', 'mismatch_debit_credit', 'unusual_correlation'])

        if anomaly_type == 'swap_gl_cc':
            # Swap GL account and cost center (creating unusual combination)
            if idx + 1 < len(df):
                df_anomalous.loc[idx, 'cost_center'] = df.loc[idx + 1, 'cost_center']

        elif anomaly_type == 'mismatch_debit_credit':
            # Create unusual debit/credit relationship
            df_anomalous.loc[idx, 'debit_amount'] = df_anomalous.loc[idx, 'credit_amount'] * 2

        elif anomaly_type == 'unusual_correlation':
            # Create unusual amount vs GL account correlation
            # (e.g., expense GL with revenue-like amount)
            df_anomalous.loc[idx, 'net_balance'] *= -1

        labels[idx] = 1

    print(f"Injected {n_anomalies} local anomalies")
    return df_anomalous, labels


if __name__ == "__main__":
    # Test preprocessing
    print("Testing GL Data Preprocessor...")

    # Create sample trial balance data
    sample_data = pd.DataFrame({
        'entity_id': ['E001'] * 100,
        'period': ['2024-01'] * 100,
        'gl_account': np.random.choice(['100000', '200000', '300000', '400000'], 100),
        'gl_name': np.random.choice(['Cash', 'Accounts Receivable', 'Revenue', 'Expenses'], 100),
        'cost_center': np.random.choice(['CC001', 'CC002', 'CC003'], 100),
        'profit_center': np.random.choice(['PC001', 'PC002'], 100),
        'debit_amount': np.random.uniform(0, 100000, 100),
        'credit_amount': np.random.uniform(0, 100000, 100),
        'posting_date': pd.date_range('2024-01-01', periods=100, freq='D')
    })

    # Feature engineering
    sample_data = create_gl_dataset(sample_data, 'E001', '2024-01')

    # Define columns
    categorical_cols = ['gl_account', 'cost_center', 'profit_center']
    numerical_cols = ['debit_amount', 'credit_amount', 'debit_credit_ratio', 'net_balance', 'abs_balance']

    # Test preprocessor
    preprocessor = GLDataPreprocessor(categorical_cols, numerical_cols)

    # Fit and transform
    X = preprocessor.fit_transform(sample_data)

    print(f"\nTransformed data shape: {X.shape}")
    print(f"Feature breakdown:")
    print(f"  - Categorical indices: {len(preprocessor.categorical_indices)}")
    print(f"  - Numerical indices: {len(preprocessor.numerical_indices)}")

    # Test anomaly injection
    print("\nTesting anomaly injection...")
    df_with_global, global_labels = inject_global_anomalies(sample_data, n_anomalies=10)
    df_with_local, local_labels = inject_local_anomalies(sample_data, n_anomalies=10)

    print(f"Global anomalies: {global_labels.sum()}")
    print(f"Local anomalies: {local_labels.sum()}")

    # Test save/load
    preprocessor.save('preprocessor_test.pkl')
    loaded_preprocessor = GLDataPreprocessor.load('preprocessor_test.pkl')

    print("\n✓ Data preprocessing tests passed!")
