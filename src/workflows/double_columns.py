import numpy as np
from sklearn.metrics import mean_squared_error

from src import config
from src.graphs.dot_plot import plot_graph
from src.outliers import detect_outliers
from src.utils import log_bic_score, select_model


# This trains on half of one column and tests on half of another.
def double_columns(
        x, y, y2, batch, batch_test, train_file, test_file, train_pctg
):
    # Validate train_percentage
    if not isinstance(train_pctg, int) or train_pctg < 1 or train_pctg > 99:
        print(f"Error: train_percentage ({train_pctg}) must be an integer between 1 and 99")
        return
    
    # Split the already-sorted pairs into train and test by train_pctg.
    num_train = int(len(x) * train_pctg / 100)
    x_train, x_test = x[:num_train], x[num_train:]
    col1A, col1B = y[:num_train], y[num_train:]  # A = train, B = test
    col2A, col2B = y2[:num_train], y2[num_train:]
    batch_train = batch[:num_train]  # Batch IDs from training file
    batch_test_split = batch_test[num_train:]  # Batch IDs from test file
    
    print(f"Train size: {len(x_train):,} ({len(x_train)/len(x)*100:.1f}%)")
    print(f"Test size: {len(x_test):,} ({len(x_test)/len(x)*100:.1f}%)")

    # Define the four training scenarios
    scenarios = [
        (f"col{train_file}A{train_pctg}_on_col{test_file}B{100-train_pctg}", 
         x_train, col1A, x_test, col2B
        ),
        #(f"col{train_file}B_on_col{test_file}A", x_odd, col1B, x_even, col2A),
    ]
    '''scenarios2 = [
        (f"col{test_file}A_on_col{train_file}B", x_even, col2A, x_odd, col1B),
        (f"col{test_file}B_on_col{train_file}A", x_odd, col2B, x_even, col1A)
    ]

    if file_num != file_num2:
        scenarios.extend(scenarios2)'''

    # Train and evaluate each scenario
    for scenario_name, x_train, y_train, x_test, y_test in scenarios:
        print(f"Training {scenario_name}")
        
        # Build, train, and test model
        model = select_model((500, 4), output_dim=1)
        
        if config.MODEL in (2, 3, 4, 5):
            model.fit([x_train, batch_train], y_train, 
                      epochs=10, batch_size=32, 
                      verbose=config.VERBOSE
            )
            predictions = model.predict([x_test, batch_test_split], verbose=0).flatten()
        else:
            model.fit(x_train, y_train, 
                      epochs=10, batch_size=32, 
                      verbose=config.VERBOSE
            )
            predictions = model.predict(x_test, verbose=0).flatten()
        
        # Calculate metrics
        smse = np.sqrt(mean_squared_error(y_test, predictions))
        correlation = np.corrcoef(y_test, predictions)[0, 1]
        print(f"SMSE: {smse:.4f}, Correlation: {correlation:.4f}")

        # Log BIC score (new line)
        run_details = f"train_file={train_file},test_file={test_file}"
        log_bic_score(y_test, predictions, model, scenario_name, run_details)
        
        # Save results
        print(f"Creating graph and outlier files...")
        plot_graph(y_test, predictions, scenario_name, smse)
        detect_outliers(x_test, y_test, predictions, scenario_name)