# src/utils/logger.py
import logging
from pathlib import Path
from .config import config

def get_logger(name="solar"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    # Set level
    level = config.get("logging.level", "INFO")
    logger.setLevel(getattr(logging, level))
    
    # Formatter
    fmt = config.get("logging.format")
    formatter = logging.Formatter(fmt)
    
    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File handler
    log_file = config.get("logging.file", "logs/app.log")
    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True)
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = get_logger()