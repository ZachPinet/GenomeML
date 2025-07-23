from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error


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
    bic_file = Path.cwd() / 'outputs' / 'bic_scores.txt'
    with open(bic_file, 'a') as f:
        f.write(
            f"{scenario_name}\t{bic:.4f}\t{correlation:.4f}\t"
            f"{smse:.4f}\t{n_params}\t{run_details}\n"
        )
    
    print(f"BIC logged: {scenario_name} = {bic:.4f}")