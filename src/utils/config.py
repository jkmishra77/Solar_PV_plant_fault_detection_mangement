# src/utils/config.py
import yaml
from pathlib import Path

class Config:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance
    
    def _load(self):
        config_path = Path("config/config.yaml")
        default_config = {
            "data": {
                "raw_path": "data/raw",
                "processed_path": "data/processed",
                "interim_path": "data/interim"
            },
            "model": {
                "features": ["V", "I", "G", "P_actual", "I_G_ratio"],
                "target": "fault_type",
                "test_size": 0.2,
                "random_state": 42
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/solar_fault_detection.log"
            }
        }
        
        if config_path.exists():
            with open(config_path) as f:
                self.data = yaml.safe_load(f)
        else:
            self.data = default_config
            config_path.parent.mkdir(exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f)
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.data
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default

config = Config()