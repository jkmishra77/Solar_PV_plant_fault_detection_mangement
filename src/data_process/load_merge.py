import pandas as pd
import os
from typing import Dict
from pathlib import Path

class DataLoader:
    """Load and merge solar PV fault detection datasets."""
    
    def load_and_merge_data(self, data_dir: str, interim_dir: str = 'data/interim') -> pd.DataFrame:
        """
        Load and merge CSV datasets with integer encoding.
        Returns DataFrame with fault_type (0-3) and cleaned features.
        """
        # Create interim directory
        Path(interim_dir).mkdir(parents=True, exist_ok=True)
        
        # Check if interim file exists
        interim_file = Path(interim_dir) / 'merged_data.csv'
        if interim_file.exists():
            return pd.read_csv(interim_file)
        
        # Load all CSV files
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        data = {}
        
        for file in csv_files:
            name = file.replace('.csv', '')
            file_path = os.path.join(data_dir, file)
            data[name] = pd.read_csv(file_path)
        
        # Merge with fault encoding
        fault_mapping = {'normal': 0, 'LL1': 1, 'LL2': 2, 'Partial_shading': 3}
        merged_data = []
        
        for name, df in data.items():
            if name not in fault_mapping:
                continue
                
            temp = df.copy()
            temp['fault_type'] = fault_mapping[name]
            
            # Drop fault columns
            columns_to_drop = ['no_module_fault', 'fault', 'partial_shading']
            for col in columns_to_drop:
                if col in temp.columns:
                    temp.drop(columns=[col], inplace=True)
            
            merged_data.append(temp)
        
        # Combine and save
        final_df = pd.concat(merged_data, ignore_index=True)
        final_df.to_csv(interim_file, index=False)
        
        return final_df

    def get_fault_labels(self) -> Dict[int, str]:
        """Return fault code to label mapping"""
        return {
            0: 'normal (no fault)',
            1: 'line-line fault type 1', 
            2: 'line-line fault type 2',
            3: 'partial shading'
        }


def main():
    """Example usage."""
    loader = DataLoader()
    merged_df = loader.load_and_merge_data('data/raw')
    
    print("Shape:", merged_df.shape)
    print("Columns:", merged_df.columns.tolist())
    print("\nFault distribution:")
    print(merged_df['fault_type'].value_counts().sort_index())
    
    return merged_df


if __name__ == "__main__":
    main()