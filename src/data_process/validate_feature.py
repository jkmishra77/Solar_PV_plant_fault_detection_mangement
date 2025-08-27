import pandas as pd
import numpy as np
from pathlib import Path
import logging

class DataValidator:
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger('DataValidator')
        
    def load_interim_data(self) -> pd.DataFrame:
        """Load data from interim folder"""
        interim_path = self.base_path / "data" / "interim" / "merged_data.csv"
        if not interim_path.exists():
            raise FileNotFoundError(f"Interim data not found at {interim_path}")
        
        df = pd.read_csv(interim_path)
        print(f"Loaded interim data: {df.shape}")
        return df
        
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the merged dataset"""
        # Required columns check
        required_cols = {'V', 'I', 'G', 'P', 'fault_type'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            print(f"Missing columns: {missing}")
            return False
            
        # Data quality checks
        checks = [
            (not df.empty, "Data is empty"),
            (len(df) > 0, "No rows in data"),
            ((df['V'] > 0).all(), "Non-positive voltage values"),
            ((df['I'] >= 0).all(), "Negative current values"),
            ((df['G'] >= 0).all(), "Negative irradiance values"),
            ((df['P'] >= 0).all(), "Negative power values"),
            (df['fault_type'].nunique() >= 2, "Insufficient fault types")
        ]
        
        for condition, error_msg in checks:
            if not condition:
                print(f"Validation failed: {error_msg}")
                return False
                
        print("Data validation passed")
        return True
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features"""
        df = df.copy()
        
        # Power features
        df['P_actual'] = df['V'] * df['I']
        df['I_G_ratio'] = np.where(df['G'] > 0, df['I'] / df['G'], 0)
        df['V_G_ratio'] = np.where(df['G'] > 0, df['V'] / df['G'], 0)
        df['efficiency'] = np.where(df['G'] > 0, df['P'] / df['G'], 0)
        
        print("Feature engineering completed")
        return df
        
    def save_processed_data(self, df: pd.DataFrame):
        """Save processed data to processed folder"""
        processed_dir = self.base_path / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_path = processed_dir / "processed_data.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"Saved processed data to {csv_path}")
        return csv_path

def main():
    """Run validation and feature engineering"""
    validator = DataValidator()
    
    try:
        # Load from interim
        df = validator.load_interim_data()
        
        # Validate
        if not validator.validate_data(df):
            raise ValueError("Data validation failed")
            
        # Engineer features
        df_processed = validator.engineer_features(df)
        
        # Save to processed
        csv_path = validator.save_processed_data(df_processed)
        print(f"Success! Processed data saved to: {csv_path}")
        print(f"Final columns: {df_processed.columns.tolist()}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()