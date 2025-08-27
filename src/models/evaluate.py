import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import joblib
from pathlib import Path
import logging

class ModelEvaluator:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger('ModelEvaluator')
        self.logger.setLevel(logging.INFO)
        
    def load_artifacts(self):
        """Load model, scaler, and label encoder"""
        models_dir = Path("models")
        
        model = joblib.load(models_dir / "xgboost_model.joblib")
        scaler = joblib.load(models_dir / "scaler.joblib")
        label_encoder = joblib.load(models_dir / "label_encoder.joblib")
        
        self.logger.info("Loaded model artifacts")
        return model, scaler, label_encoder
        
    def load_test_data(self):
        """Load and prepare test data"""
        processed_path = Path("data") / "processed" / "processed_data.csv"
        df = pd.read_csv(processed_path)
        
        X = df[self.config['features']]
        y = df[self.config['target']]
        
        return X, y
        
    def evaluate(self, model, scaler, label_encoder, X, y):
        """Comprehensive model evaluation"""
        # Preprocess
        y_encoded = label_encoder.transform(y)
        X_scaled = scaler.transform(X)
        
        # Generate predictions
        y_pred = model.predict(X_scaled)
        
        # Get actual unique classes
        present_classes = np.union1d(np.unique(y_encoded), np.unique(y_pred))
        class_names = [str(cls) for cls in label_encoder.classes_]
        present_class_names = [class_names[i] for i in present_classes if i < len(class_names)]
        
        # 1. Classification Report
        print("=== Detailed Classification Report ===")
        print(classification_report(y_encoded, y_pred, 
                                  labels=present_classes,
                                  target_names=present_class_names,
                                  zero_division=0))
        
        # 2. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_encoded, y_pred, labels=present_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=present_class_names,
                    yticklabels=present_class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(Path("models") / "confusion_matrix.png")
        plt.show()
        
        # 3. Feature Importance
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            feat_importances = pd.Series(model.feature_importances_, 
                                       index=self.config['features'])
            feat_importances.nlargest(10).plot(kind='barh')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig(Path("models") / "feature_importance.png")
            plt.show()
        
        # 4. Metrics
        metrics = {
            'accuracy': accuracy_score(y_encoded, y_pred),
            'f1_macro': f1_score(y_encoded, y_pred, average='macro'),
            'f1_weighted': f1_score(y_encoded, y_pred, average='weighted')
        }
        
        print(f"\n=== Model Metrics ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        return metrics

def main():
    """Main evaluation pipeline"""
    CONFIG = {
        "target": "fault_type",
        "features": ["V", "I", "G", "P", "P_actual", "I_G_ratio", "V_G_ratio", "efficiency"]
    }
    
    try:
        evaluator = ModelEvaluator(CONFIG)
        
        # Load artifacts
        model, scaler, label_encoder = evaluator.load_artifacts()
        
        # Load data
        X, y = evaluator.load_test_data()
        
        # Evaluate
        metrics = evaluator.evaluate(model, scaler, label_encoder, X, y)
        
        print("Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()