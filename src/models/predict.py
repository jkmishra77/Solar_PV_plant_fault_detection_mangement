import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

class PVFaultPredictor:
    def __init__(self):
        self.models_dir = Path("models")
        self.config = {
            "features": ["V", "I", "G", "P", "P_actual", "I_G_ratio", "V_G_ratio", "efficiency"]
        }
        
    def load_artifacts(self):
        """Load model artifacts"""
        self.model = joblib.load(self.models_dir / "xgboost_model.joblib")
        self.scaler = joblib.load(self.models_dir / "scaler.joblib")
        self.label_encoder = joblib.load(self.models_dir / "label_encoder.joblib")
        
    def engineer_features(self, df):
        """Add engineered features"""
        df = df.copy()
        df['P_actual'] = df['V'] * df['I']
        df['I_G_ratio'] = np.where(df['G'] > 0, df['I'] / df['G'], 0)
        df['V_G_ratio'] = np.where(df['G'] > 0, df['V'] / df['G'], 0)
        df['efficiency'] = np.where(df['G'] > 0, df['P'] / df['G'], 0)
        return df
        
    def predict_csv(self, csv_path):
        """Predict from CSV file with optional verification"""
        df = pd.read_csv(csv_path)
        
        # Check if fault_type exists for verification
        has_ground_truth = 'fault_type' in df.columns
        
        # Engineer features
        df_engineered = self.engineer_features(df)
        X = df_engineered[self.config['features']]
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Prepare results
        results = df.copy()
        results['predicted_fault'] = predictions
        
        # Add verification if ground truth exists
        if has_ground_truth:
            results['correct'] = results['fault_type'] == results['predicted_fault']
            accuracy = results['correct'].mean()
            print(f"Verification accuracy: {accuracy:.3f}")
        
        return results

def main():
    """Predict from CSV file"""
    if len(sys.argv) != 2:
        print("Usage: python predict.py <input_csv_file>")
        print("Example: python predict.py predict.csv")
        return
    
    csv_file = sys.argv[1]
    
    predictor = PVFaultPredictor()
    predictor.load_artifacts()
    
    results = predictor.predict_csv(csv_file)
    
    # Display results
    print("\nPrediction Results:")
    print("=" * 80)
    
    for i, row in results.iterrows():
        line = f"Sample {i+1}: V={row['V']:.2f}V, I={row['I']:.2f}A, G={row['G']:.2f}W/m²"
        line += f" → Predicted: {row['predicted_fault']}"
        
        if 'fault_type' in row:
            line += f" | Actual: {row['fault_type']}"
            if 'correct' in row:
                status = "✓" if row['correct'] else "✗"
                line += f" {status}"
                
        print(line)
    
    # Save results
    output_file = "prediction_results.csv"
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()