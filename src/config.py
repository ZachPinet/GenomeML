# These default settings can be overriden by a local config file
DEFAULT_SETTINGS = {
    'FILE_NUM': 0,
    'FILE_NUM2': 0,
    'DO_SINGLE_COLUMN': True,
    'USE_PCA': False,
    'DO_ENSEMBLE': False,
    'DO_DOUBLE_COLUMNS': False,
    'DO_KFOLD': False,
    'TRANSPOSE': False,
    'MAX_SEQS': 999999,
    'PCA_COMPONENTS': 4,
    'TRAIN_PERCENTAGE': 50,
    'DATA_SPLITS': 10,
    'SHOW_BOUNDS': True,
    'STD_MULTIPLIER': 2,
    'FRAC': 0.3,  # 0-1. Affects smoothness and sensitivity of curve
    'MODE': 'both',  # Log outliers. 'simple', 'complex', 'both', 'off'
    'RANDOM_SEED': 42,
}

# Try to load user overrides from config_local.py (not committed)
try:
    from .config_local import USER_SETTINGS
    settings = {**DEFAULT_SETTINGS, **USER_SETTINGS}
except ImportError:
    settings = DEFAULT_SETTINGS

# Export as module-level variables
for key, value in settings.items():
    globals()[key] = value