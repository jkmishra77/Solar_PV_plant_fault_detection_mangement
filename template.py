import os
from pathlib import Path

# Folder structure
folders = [
    "data/raw",
    "data/processed", 
    "data/interim",
    "src/data_process",
    "src/models",
    "src/utils",
    "logs",
    "models",
    "notebooks",
    "tests",
    "config"
]

# Files to create
files = [
    "src/data_process/load_merge.py",
    "src/data_process/validate_feature.py",
    "src/models/train.py",
    "src/models/evaluate.py",
    "src/models/predict.py",
    "src/models/combined_analysis.py",
    "src/utils/config.py",
    "src/utils/logger.py",
    "config/config.yaml",
    "requirements.txt",
    "README.md"
]

def create_project_structure():
    """Create the complete project structure"""
    print("Creating project structure...")
    
    # Create all folders
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created folder: {folder}/")
    
    # Create all files
    for file in files:
        # Create parent directories if they don't exist
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        Path(file).touch()
        print(f"✓ Created file: {file}")
    
    # Create __init__.py files for Python packages
    init_files = [
        "src/__init__.py",
        "src/data_process/__init__.py", 
        "src/models/__init__.py",
        "src/utils/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✓ Created package: {init_file}")
    
    print("\n✅ Project structure created successfully!")
    print("\nNext steps:")
    print("1. Add your raw data files to data/raw/")
    print("2. Configure settings in config/config.yaml")
    print("3. Install dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    create_project_structure()