import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
import logging

class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.logger = logging.getLogger('ModelTrainer')
        self.logger.setLevel(logging.INFO)
        
    def load_data(self) -> pd.DataFrame:
        """Load processed data from processed folder"""
        processed_path = Path("data") / "processed" / "processed_data.csv"
        if not processed_path.exists():
            raise FileNotFoundError(f"Processed data not found at {processed_path}")
        
        df = pd.read_csv(processed_path)
        self.logger.info(f"Loaded processed data: {df.shape}")
        return df
        
    def preprocess(self, df: pd.DataFrame) -> tuple:
        """Prepare features and labels"""
        try:
            X = df[self.config['features']]
            y = self.label_encoder.fit_transform(df[self.config['target']])
            
            # Handle class imbalance
            if self.config.get('use_smote', False):
                X, y = SMOTE(random_state=self.config['random_state']).fit_resample(X, y)
                
            return X, y
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {str(e)}")
            raise

    def train(self, X: pd.DataFrame, y: np.ndarray):
        """Optimized training workflow with MLflow tracking"""
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # MLflow setup
        mlflow.set_experiment("SolarFaultDetection")
        
        with mlflow.start_run():
            # Log parameters and data info
            mlflow.log_params(self.config['model_params'])
            mlflow.log_param("dataset_size", len(X))
            mlflow.log_param("feature_count", X.shape[1])
            mlflow.log_param("class_count", len(np.unique(y)))
            
            # Initialize and train model
            model = XGBClassifier(
                **self.config['model_params'],
                random_state=self.config['random_state'],
                early_stopping_rounds=20
            )
            
            model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=10
            )
            
            # Evaluation
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = {
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_weighted': f1_score(y_test, y_pred, average='weighted')
            }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model and artifacts
            mlflow.sklearn.log_model(model, "model")
            mlflow.sklearn.log_model(self.scaler, "scaler")
            mlflow.sklearn.log_model(self.label_encoder, "label_encoder")
            
            # Log classification report
            report = classification_report(y_test, y_pred)
            mlflow.log_text(report, "classification_report.txt")
            
            self.logger.info("\nClassification Report:\n" + report)
            
            return model

    def save_artifacts(self, model):
        """Save model and preprocessing artifacts locally"""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        joblib.dump(model, models_dir / "xgboost_model.joblib")
        joblib.dump(self.scaler, models_dir / "scaler.joblib")
        joblib.dump(self.label_encoder, models_dir / "label_encoder.joblib")
        
        self.logger.info(f"Artifacts saved to {models_dir}")

def main():
    """Main training pipeline"""
    CONFIG = {
        "target": "fault_type",
        "features": ["V", "I", "G", "P", "P_actual", "I_G_ratio", "V_G_ratio", "efficiency"],
        "test_size": 0.2,
        "random_state": 42,
        "use_smote": True,
        "model_params": {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "objective": "multi:softmax",
        }
    }
    
    try:
        trainer = ModelTrainer(CONFIG)
        
        # Load and preprocess data
        df = trainer.load_data()
        X, y = trainer.preprocess(df)
        
        # Update num_classes based on actual data
        CONFIG['model_params']['num_class'] = len(np.unique(y))
        
        # Train model
        model = trainer.train(X, y)
        
        # Save artifacts locally
        trainer.save_artifacts(model)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")

if __name__ == "__main__":
    main()