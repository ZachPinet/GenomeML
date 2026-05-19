import importlib
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

from src import config


# Log the run's BIC score to a .txt file
def log_bic_score(y_true, y_pred, model, scenario_name, run_details):
    # Calculate BIC
    n_params = model.count_params()
    n_samples = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    log_likelihood = -n_samples/2 * np.log(2 * np.pi * rss/n_samples) - rss/(2 * rss/n_samples)
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    
    # Calculate other metrics
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    smse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Log to file (append mode like outlier_percents.txt)
    run_data_path = Path.cwd() / 'outputs' / 'run_data'
    run_data_file = run_data_path / f'{config.RUN_DATA_FILE}.txt'
    with open(run_data_file, 'a') as f:
        f.write(
            f"{scenario_name}\t{bic:.4f}\t{correlation:.4f}\t"
            f"{smse:.4f}\t{n_params}\t{run_details}\n"
        )
    
    print(f"BIC logged: {scenario_name} = {bic:.4f}")


# This one-hot encodes any sequence and returns it.
def one_hot_encode(sequence):
    mapping = {
        'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'T': [0, 0, 1, 0], 
        'G': [0, 0, 0, 1], 'N': [0, 0, 0, 0]
    }
    return np.array([mapping[nuc] for nuc in sequence], dtype=np.float32)


# This one-hot-decodes any sequence back into letters and returns it.
def one_hot_decode(encoded_sequence):
    reverse_mapping = {
        (1, 0, 0, 0): 'A', (0, 1, 0, 0): 'C', (0, 0, 1, 0): 'T', 
        (0, 0, 0, 1): 'G', (0, 0, 0, 0): 'N'
    }
    decoded_sequence = ''.join(
        reverse_mapping.get(tuple(vec), 'N') for vec in encoded_sequence
    )
    return decoded_sequence


# This selects the model to be used from the models/ folder.
def select_model(input_shape, output_dim=1, num_batches=3):
    # Access the files in the 'models' folder.
    cwd = Path.cwd()
    models_dir = cwd / 'src' / 'models'
    model_file_prefix = f"model{config.MODEL}_"
    model_name = None

    # Select the correct model based off the config value.
    for file in models_dir.iterdir():
        if file.is_file() and file.name.startswith(model_file_prefix):
            model_name = file.stem
            break

    if model_name is None:
        model_name = "model1_basic"

    # Print feedback to terminal.
    model_nickname = model_name.removeprefix(model_file_prefix)
    print(f"Running Model {config.MODEL} - {model_nickname.title()}")

    # Pass the inputs to the build_model function of the selected model.
    model_file = importlib.import_module(f"src.models.{model_name}")
    build_model = model_file.build_model
    return build_model(input_shape, output_dim, num_batches)