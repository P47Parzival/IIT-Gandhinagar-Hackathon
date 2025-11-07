"""
Anomaly Exporter - Export detected anomalies for validation.

Converts Paper 1's anomaly detection output into structured records
that can be consumed by Paper 2's validation framework.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path


class AnomalyExporter:
    """
    Exports anomalies detected by the FCL system with all necessary
    context for the validation pipeline.
    """
    
    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize exporter.
        
        Args:
            output_dir: Directory to save exported anomalies
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Anomaly Exporter initialized: {self.output_dir}")
    
    def export_anomalies(
        self,
        df: pd.DataFrame,
        reconstruction_errors: np.ndarray,
        threshold: float,
        entity_name: Optional[str] = None,
        period: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Export detected anomalies with full context.
        
        Args:
            df: DataFrame with GL transactions
            reconstruction_errors: Reconstruction errors from autoencoder
            threshold: Anomaly threshold
            entity_name: Entity name
            period: Accounting period
            metadata: Additional metadata
            
        Returns:
            List of anomaly records
        """
        # Find anomalies
        is_anomaly = reconstruction_errors > threshold
        anomaly_indices = np.where(is_anomaly)[0]
        
        print(f"Found {len(anomaly_indices)} anomalies above threshold {threshold:.4f}")
        
        anomaly_records = []
        
        for idx in anomaly_indices:
            record = self._create_anomaly_record(
                df=df,
                idx=idx,
                reconstruction_error=reconstruction_errors[idx],
                threshold=threshold,
                entity_name=entity_name,
                period=period,
                metadata=metadata
            )
            anomaly_records.append(record)
        
        return anomaly_records
    
    def _create_anomaly_record(
        self,
        df: pd.DataFrame,
        idx: int,
        reconstruction_error: float,
        threshold: float,
        entity_name: Optional[str],
        period: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create structured anomaly record."""
        row = df.iloc[idx]
        
        # Generate anomaly ID
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        anomaly_id = f"ANO_{timestamp}_{idx}"
        
        # Extract GL information
        gl_account = str(row.get('Konto', row.get('gl_account', 'UNKNOWN')))
        gl_name = str(row.get('Kontobezeichnung', row.get('gl_name', 'Unknown Account')))
        
        # Extract amounts
        amount = float(row.get('Betrag', row.get('amount', 0)))
        
        # Build record
        record = {
            # Identification
            'anomaly_id': anomaly_id,
            'detection_timestamp': datetime.now().isoformat(),
            'source': 'fcl_detector',
            
            # Entity & Period
            'entity_id': str(row.get('Buchungskreis', row.get('entity_id', 'UNKNOWN'))),
            'entity_name': entity_name or 'Unknown Entity',
            'period': period or str(row.get('Periode', row.get('period', 'UNKNOWN'))),
            'fiscal_year': int(row.get('Geschäftsjahr', row.get('fiscal_year', datetime.now().year))),
            
            # GL Account
            'gl_account': gl_account,
            'gl_name': gl_name,
            'gl_category': self._categorize_gl_account(gl_account),
            
            # Transaction Details
            'amount': amount,
            'currency': str(row.get('Währung', row.get('currency', 'USD'))),
            'document_number': str(row.get('Belegnummer', row.get('doc_number', ''))),
            'document_type': str(row.get('Belegart', row.get('doc_type', ''))),
            'posting_date': str(row.get('Buchungsdatum', row.get('posting_date', ''))),
            'entry_date': str(row.get('Erfassungsdatum', row.get('entry_date', ''))),
            
            # Anomaly Metrics
            'anomaly_score': float(reconstruction_error / threshold),
            'reconstruction_error': float(reconstruction_error),
            'threshold': float(threshold),
            'severity': self._calculate_severity(reconstruction_error, threshold, amount),
            
            # Expected Range (estimated from historical data)
            'expected_min': self._estimate_expected_min(df, gl_account),
            'expected_max': self._estimate_expected_max(df, gl_account),
            
            # Additional Features (for context)
            'features': self._extract_features(row),
            
            # Metadata
            'metadata': metadata or {},
            
            # Document path (for validator to find supporting documents)
            'document_path': f"documents/{entity_name or 'unknown'}/{period or 'unknown'}/{gl_account}"
        }
        
        return record
    
    def _categorize_gl_account(self, gl_account: str) -> str:
        """Categorize GL account by type."""
        try:
            account_num = int(gl_account)
            
            if 1000 <= account_num < 2000:
                return 'ASSETS'
            elif 2000 <= account_num < 3000:
                return 'LIABILITIES'
            elif 3000 <= account_num < 4000:
                return 'EQUITY'
            elif 4000 <= account_num < 5000:
                return 'REVENUE'
            elif 5000 <= account_num < 6000:
                return 'COST_OF_GOODS'
            elif 6000 <= account_num < 8000:
                return 'EXPENSES'
            else:
                return 'OTHER'
        except:
            return 'UNKNOWN'
    
    def _calculate_severity(
        self,
        reconstruction_error: float,
        threshold: float,
        amount: float
    ) -> str:
        """Calculate anomaly severity."""
        score = reconstruction_error / threshold
        abs_amount = abs(amount)
        
        # High severity: large deviation + large amount
        if score > 5.0 and abs_amount > 1_000_000:
            return 'CRITICAL'
        elif score > 3.0 or abs_amount > 500_000:
            return 'HIGH'
        elif score > 2.0 or abs_amount > 100_000:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _estimate_expected_min(self, df: pd.DataFrame, gl_account: str) -> float:
        """Estimate expected minimum value for GL account."""
        try:
            gl_data = df[df.get('Konto', df.get('gl_account')) == gl_account]
            amount_col = 'Betrag' if 'Betrag' in df.columns else 'amount'
            
            if len(gl_data) > 0:
                return float(gl_data[amount_col].quantile(0.05))
            return 0.0
        except:
            return 0.0
    
    def _estimate_expected_max(self, df: pd.DataFrame, gl_account: str) -> float:
        """Estimate expected maximum value for GL account."""
        try:
            gl_data = df[df.get('Konto', df.get('gl_account')) == gl_account]
            amount_col = 'Betrag' if 'Betrag' in df.columns else 'amount'
            
            if len(gl_data) > 0:
                return float(gl_data[amount_col].quantile(0.95))
            return 0.0
        except:
            return 0.0
    
    def _extract_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract all available features for context."""
        features = {}
        
        # Convert all row values to features
        for col, val in row.items():
            try:
                # Convert to JSON-serializable types
                if pd.isna(val):
                    features[str(col)] = None
                elif isinstance(val, (np.integer, np.floating)):
                    features[str(col)] = float(val)
                else:
                    features[str(col)] = str(val)
            except:
                features[str(col)] = str(val)
        
        return features
    
    def save_anomalies(
        self,
        anomalies: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """
        Save anomalies to JSON file.
        
        Args:
            anomalies: List of anomaly records
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"anomalies_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(anomalies, f, indent=2, default=str)
        
        print(f"✓ Saved {len(anomalies)} anomalies to: {filepath}")
        return str(filepath)
    
    def save_anomalies_csv(
        self,
        anomalies: List[Dict[str, Any]],
        filename: Optional[str] = None
    ) -> str:
        """
        Save anomalies to CSV file (flattened).
        
        Args:
            anomalies: List of anomaly records
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"anomalies_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Flatten records (exclude nested dicts)
        flattened = []
        for record in anomalies:
            flat = {k: v for k, v in record.items() if not isinstance(v, dict)}
            flattened.append(flat)
        
        df = pd.DataFrame(flattened)
        df.to_csv(filepath, index=False)
        
        print(f"✓ Saved {len(anomalies)} anomalies to: {filepath}")
        return str(filepath)


if __name__ == "__main__":
    print("Anomaly Exporter")
    print("✓ Ready to export detected anomalies")

