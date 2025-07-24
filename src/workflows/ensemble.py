from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

from src import config
from src.graphs.dot_plot import plot_graph
from src.model import build_model
from src.outliers import detect_outliers


# This ensures the input numbers for ensemble() are valid.
def is_compatible(train_pct, data_splits):
    # The train_pct must be a multiple of (100 / data_splits) or bin_pct
    bin_pct = 100 / data_splits
    
    # Check for any conditions that would make the numbers invalid
    if train_pct % bin_pct != 0:
        print(f"train_pct ({train_pct}) must be a multiple of {bin_pct:.1f}.")
        return False
    elif train_pct <= 0 or train_pct > 100:
        print(f"train_pct ({train_pct}) must be between 1 and 100.")
        return False
    elif data_splits <= 0:
        print(f"data_splits ({data_splits}) must be positive.")
        return False
    
    print(f"{train_pct}% training is compatible with {data_splits} splits.")
    return True


# This ensemble approach averages predictions from multiple models.
def ensemble(
        x, y, train_percentage, data_splits, col_name, show_bounds, 
        std_multiplier, frac, output_file, pct_file, mode
):
    # Train on overlapping continuous segments, test on full dataset

    # Compatibility check
    if not is_compatible(train_percentage, data_splits):
        print(f"Incompatible parameters: {train_percentage}, {data_splits}")
        return
    
    # Save outputs to 'ensemble' folder
    ensemble_dir = Path.cwd() / 'outputs' / 'ensemble'
    if not ensemble_dir.exists():
        print(f"'outliers' directory not found at {ensemble_dir}")
        return
    
    # Split the data into bins to iterate through
    total_samples = len(x)
    bin_size = total_samples // data_splits
    bins_to_use = int(data_splits * train_percentage / 100)
    n_models = data_splits

    print(f"Total samples: {total_samples:,}")
    print(f"Bin size: {bin_size:,}")
    print(f"Bins to use per model: {bins_to_use}")
    print(f"Number of models: {n_models}")

    # Calculate training range percentages for each model
    train_ranges = []
    step_size = 100 / data_splits  # Step size as percentage

    for model_idx in range(n_models):
        start_pct = (model_idx * step_size) % 100
        end_pct = (start_pct + train_percentage)
        start_idx = int(total_samples * start_pct / 100)
        
        # Doesn't wrap around to the start
        if end_pct <= 100:
            end_idx = int(total_samples * end_pct / 100)
            train_indices = list(range(start_idx, end_idx))
            range_desc = f"{start_pct:.1f}-{end_pct:.1f}"
    
        # Wraps around to the start
        else:
            end_pct_wrapped = end_pct - 100
            end_idx = int(total_samples * end_pct_wrapped / 100)
            train_indices = (
                list(range(start_idx, total_samples)) + 
                list(range(0, end_idx))
            )
            range_desc = f"{start_pct:.1f}-100, 0-{end_pct_wrapped:.1f}"
        
        train_ranges.append(train_indices)

        # Print readable range description
        print(
            f"Model {model_idx + 1}: {range_desc}% "
            f"({len(train_indices):,} samples)"
        )
        
    # Train models and collect predictions on full dataset
    all_predictions = []
    for model_idx, train_indices in enumerate(train_ranges):
        print(f"Training Model {model_idx + 1}/{n_models} ...")
        
        x_train = x[train_indices]
        y_train = y[train_indices]
        
        model = build_model((500, 4), 1)
        model.fit(x_train, y_train, 
                  epochs=10, batch_size=32, 
                  verbose=config.VERBOSE
        )
        
        full_predictions = model.predict(x, verbose=0).flatten()
        all_predictions.append(full_predictions)

    print(f"Averaging predictions from {n_models} models...")
    
    # Convert to array of shape (n_models, n_samples)
    predictions_array = np.array(all_predictions)
    averaged_predictions = np.mean(predictions_array, axis=0)

    # Calculate metrics
    final_corr = np.corrcoef(y, averaged_predictions)[0, 1]
    final_smse = np.sqrt(mean_squared_error(y, averaged_predictions))
    print(f"Final correlation: {final_corr:.4f}")
    print(f"Final SMSE: {final_smse:.4f}")
    print(f"Prediction std: {np.std(averaged_predictions):.4f}")
    
    # Save results
    method_name = f"ensemble{train_percentage}p{data_splits}s"
    pred_file = ensemble_dir / f'{method_name}_{col_name}.txt'
    np.savetxt(pred_file, averaged_predictions, fmt='%.6f')
    print(f"Saved predictions: {pred_file}")
    
    print(f"Creating graph and outlier files...")
    final_col_name = f"ensemble{train_percentage}p{data_splits}s-{col_name}"
    plot_graph(
        y, averaged_predictions, final_col_name, final_smse, 
        show_bounds, std_multiplier, frac
    )
    detect_outliers(
        x, y, averaged_predictions, output_file, pct_file, 
        final_col_name, std_multiplier, frac, mode
    )