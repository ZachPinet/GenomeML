from pathlib import Path


# Parse the string value to Python type given by default_settings.
def _parse_value(default_settings, key, value):
    
    # Get the expected type from default_settings.
    if key not in default_settings:
        raise ValueError(f"Invalid config key {key}: {value}")
    
    default_value = default_settings[key]
    expected_type = type(default_value)

    try:
        if expected_type == bool:
            if value.capitalize() in ('True', 'False'):
                return value.capitalize() == 'True'
        
        elif expected_type == int:
            if key == 'TRAIN_PERCENTAGE':
                if int(value) < 1 or int(value) > 99:
                    raise ValueError(
                        f"Invalid Train Percentage: '{value}'. "
                        "Percentage must be between 1 and 99."
                    )
            if key == 'VERBOSE':
                if int(value) < 0 or int(value) > 2:
                    raise ValueError(
                        f"Invalid Verbose: '{value}'. "
                        "Verbose must be 0, 1, or 2."
                    )
            return int(value)
        
        elif expected_type == float:
            if key == 'FRAC':
                if float(value) < 0 or float(value) > 1:
                    raise ValueError(
                        f"Invalid Frac: '{value}'. "
                        "Frac must be a float between 0 and 1."
                    )
            return float(value)
        
        elif expected_type == str:
            # Make sure outlier mode is valid.
            if key == 'OUTLIER_MODE':
                if value.lower() not in ('simple', 'complex', 'both', 'off'):
                    raise ValueError(
                        f"Invalid Outlier Mode: '{value}'. "
                        "Mode must be 'simple', 'complex', 'both', or 'off'."
                    )
            # Make sure filenames are valid.
            elif key in (
                'COMPLEX_OUTLIER_FILE', 'PERCENT_OUTLIER_FILE', 'RUN_DATA_FILE'
            ):
                if len(value) < 1 or len(value) > 255:
                    raise ValueError(
                        f"Invalid File Name: '{value}'. "
                        "File names must be between 1-255 characters."
                    )
                elif value[0] in '._-' or value[-1] in '._-':
                    raise ValueError(
                        f"Invalid File Name: '{value}'. "
                        "Names cannot start or end with '.', '_', or '-'."
                    )
                for x in value:
                    if x.isalnum() == False and x not in '._-':
                        raise ValueError(
                            f"Invalid File Name: '{value}'. "
                            "All chars must be alphanumeric, '.', '_', or '-'."
                        )
                if value[-4:] == '.txt':
                    value = value[:-4]
                    
            return value

    # If there is an issue with the local config, use the default value.
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not parse '{value}' as {expected_type.__name__}"
              f" for key '{key}', using default: {e}.")
        return default_value


# Load local configuration overrides from config_local.txt.
def _load_config_from_file(default_settings):
    config_file = Path(__file__).parent / 'config_local.txt'
    
    if not config_file.exists():
        return {}
    
    overrides = {}
    try:
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments.
                if not line or line.startswith('#'):
                    continue
                
                # Parse key=value pairs.
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    value = _parse_value(default_settings, key, value)

                    if key in default_settings:
                        overrides[key] = value
                    else:
                        print(f"Warning: Unknown config key '{key}'")
    
    except Exception as e:
        print(f"Error reading config_local.txt: {e}")
        return {}
    
    return overrides


# This reloads config values and updates module-level variables.
def reload_config():
    # These are the default configuration settings for GenomeML.
    default_settings = {
        'SINGLE_FILE_NUM': 1,
        'RANGE_START_FILE_NUM': 1,
        'RANGE_END_FILE_NUM': 1,
        'DO_RANGE': False,
        'DO_DOUBLE_COLUMNS': False,
        'DO_ENSEMBLE': False,
        'DO_PCA': False,
        'DO_SINGLE_COLUMN': True,
        'KFOLD': False,
        'PCA_COMPONENTS': 4,
        'DATA_SPLITS': 10,
        'DOUBLE_TRAIN_FILE': 0,
        'MAX_SEQS': 999999,
        'TRAIN_PERCENTAGE': 50,
        'ENABLE_FILTER_RANGE': True,
        'FILTER_RANGE_MIN': -12.0,
        'FILTER_RANGE_MAX': 12.0,
        'VERBOSE': 1,
        'MAKE_PLOT': True,
        'SHOW_BOUNDS': True,
        'STD_MULTIPLIER': 2.0,
        'FRAC': 0.3,
        'OUTLIER_MODE': 'both',
        'MODEL': 1,
        'RANDOM_SEED': 42,
        'WINDOW': True,
        'COMPLEX_OUTLIER_FILE': 'outliers',
        'PERCENT_OUTLIER_FILE': 'outlier_percents',
        'RUN_DATA_FILE': 'run_data',
    }
    
    # Load settings with overrides.
    settings = {**default_settings, **_load_config_from_file(default_settings)}
    
    # Export as module-level variables for other files' ease of access.
    for key, value in settings.items():
        globals()[key] = value