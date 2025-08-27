import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

class CombinedAnalyzer:
    def __init__(self):
        self.config = {
            "target": "fault_type",
            "features": ["V", "I", "G", "P", "P_actual", "I_G_ratio", "V_G_ratio", "efficiency"],
            "random_state": 42
        }
        
    def load_artifacts(self):
        """Load model, scaler, and label encoder"""
        models_dir = Path("models")
        
        model = joblib.load(models_dir / "xgboost_model.joblib")
        scaler = joblib.load(models_dir / "scaler.joblib")
        label_encoder = joblib.load(models_dir / "label_encoder.joblib")
        
        print("Loaded model artifacts")
        return model, scaler, label_encoder
        
    def load_data(self):
        """Load processed data"""
        processed_path = Path("data") / "processed" / "processed_data.csv"
        df = pd.read_csv(processed_path)
        print(f"Loaded data: {df.shape}")
        return df
        
    def combined_analysis(self, df, model, label_encoder):
        """
        Perform both supervised fault prediction and unsupervised anomaly detection
        Returns DataFrame with both predictions and anomaly scores
        """
        # ===== 1. Supervised Fault Prediction =====
        X = df[self.config['features']]
        y_true = label_encoder.transform(df[self.config['target']])
        
        # Scale features (same as training)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Make predictions
        y_pred = model.predict(X_scaled)
        df_result = df.copy()
        df_result['predicted_fault'] = label_encoder.inverse_transform(y_pred)
        df_result['predicted_fault_code'] = y_pred
        
        # ===== 2. Unsupervised Anomaly Detection ===== 
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.05,  # Expect ~5% anomalies
            random_state=self.config['random_state']
        )
        iso_forest.fit(X_scaled)
        
        # Get anomaly scores
        df_result['anomaly_score'] = iso_forest.decision_function(X_scaled)
        df_result['is_anomaly'] = iso_forest.predict(X_scaled)
        df_result['is_anomaly'] = df_result['is_anomaly'].map({1: False, -1: True})
        
        # ===== 3. Analysis & Visualization =====
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Confusion Matrix (Supervised)
        plt.subplot(2, 2, 1)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                    display_labels=label_encoder.classes_)
        disp.plot(ax=plt.gca())
        plt.title('Supervised: Fault Prediction Accuracy')
        
        # Plot 2: Anomaly Score Distribution
        plt.subplot(2, 2, 2)
        sns.histplot(df_result['anomaly_score'], bins=50, kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Unsupervised: Anomaly Score Distribution')
        
        # Plot 3: Feature vs Anomaly Score
        plt.subplot(2, 2, 3)
        sample_df = df_result.sample(min(1000, len(df_result)), random_state=42)
        sns.scatterplot(data=sample_df, x='P_actual', y='anomaly_score', 
                       hue='is_anomaly', alpha=0.6)
        plt.title('Power vs Anomaly Score')
        
        # Plot 4: Agreement Analysis
        plt.subplot(2, 2, 4)
        agreement = df_result.groupby(['predicted_fault', 'is_anomaly']).size().unstack()
        sns.heatmap(agreement, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Fault Prediction vs Anomaly Detection')
        
        plt.tight_layout()
        plt.savefig(Path("models") / "combined_analysis.png")
        plt.show()
        
        return df_result
        
    def save_results(self, df_result):
        """Save combined results"""
        output_path = Path("data") / "processed" / "combined_analysis_results.csv"
        df_result.to_csv(output_path, index=False)
        print(f"Saved combined results to: {output_path}")
        
    def print_insights(self, df_result):
        """Print key insights"""
        print("\n=== Key Insights ===")
        print("1. Supervised Model detects known fault types:")
        print(df_result['predicted_fault'].value_counts())
        
        print(f"\n2. Anomaly Detection finds suspicious samples:")
        print(f"{df_result['is_anomaly'].sum()} anomalous samples detected")
        
        print("\n3. Agreement Analysis:")
        print(df_result.groupby(['predicted_fault', 'is_anomaly']).size())

def main():
    """Main combined analysis pipeline"""
    analyzer = CombinedAnalyzer()
    
    try:
        # Load artifacts and data
        model, scaler, label_encoder = analyzer.load_artifacts()
        df = analyzer.load_data()
        
        # Perform combined analysis
        print("Running combined fault prediction and anomaly detection...")
        results_df = analyzer.combined_analysis(df, model, label_encoder)
        
        # Save results
        analyzer.save_results(results_df)
        
        # Print insights
        analyzer.print_insights(results_df)
        
        print("\nCombined analysis completed successfully!")
        
    except Exception as e:
        print(f"Combined analysis failed: {e}")

if __name__ == "__main__":
    main()