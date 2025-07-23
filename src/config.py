import os
from pathlib import Path


# Configuration settings for GenomeML
DEFAULT_SETTINGS = {
    'FILE_NUM': 0,
    'FILE_NUM2': 0,
    'DO_DOUBLE_COLUMNS': False,
    'DO_ENSEMBLE': False,
    'DO_PCA': False,
    'DO_SINGLE_COLUMN': True,
    'KFOLD': False,
    'TRANSPOSE': False,
    'MAX_SEQS': 999999,
    'PCA_COMPONENTS': 4,
    'TRAIN_PERCENTAGE': 50,
    'DATA_SPLITS': 10,
    'RANGE_START': 0,
    'RANGE_END': 50,
    'SHOW_BOUNDS': True,
    'STD_MULTIPLIER': 2,
    'FRAC': 0.3,  # 0-1. Affects smoothness and sensitivity of curve
    'MODE': 'both',  # Log outliers. 'simple', 'complex', 'both', 'off'
    'RANDOM_SEED': 42,
}


# Parse string value to appropriate Python type
def _parse_value(value_str):
    value_str = value_str.strip()
    
    # Boolean values
    if value_str.lower() in ('true', 'false'):
        return value_str.lower() == 'true'
    
    # Try integer
    try:
        return int(value_str)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Return as string
    return value_str


# Load local configuration overrides from config_local.txt
def _load_config_from_file():
    config_file = Path(__file__).parent / 'config_local.txt'
    
    if not config_file.exists():
        return {}
    
    overrides = {}
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = _parse_value(value)
                    
                    if key in DEFAULT_SETTINGS:
                        overrides[key] = value
                    else:
                        print(f"Warning: Unknown config key '{key}'")
    
    except Exception as e:
        print(f"Error reading config_local.txt: {e}")
        return {}
    
    return overrides

# Load settings with overrides
settings = {**DEFAULT_SETTINGS, **_load_config_from_file()}

# Export as module-level variables
for key, value in settings.items():
    globals()[key] = value