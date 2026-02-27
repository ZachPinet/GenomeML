import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from src import config
from src.graphs.dot_plot import plot_graph
from src.model import build_model
from src.outliers import detect_outliers


# This trains on and predicts values for one specific column, no PCA.
def single_column(x, y, batch, kfold, col_name, mode):
    if kfold:
        print(f"PCA disabled - using K-fold cross-validation on one file.")
    else:
        print(f"PCA disabled - single run on column {col_name}.")

    # Save all values for final combined graph
    all_y_true = []
    all_y_pred = []
    all_seqs = []

    # Begin fold loop
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    for train_idx, test_idx in kf.split(x):
        if kfold:
            print(f"Fold {fold} begin: ")

        # Prepare data for the model
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        batch_train, batch_test = batch[train_idx], batch[test_idx]

        model = build_model((500, 4))

        # Train the model
        if config.USE_BATCH_CORRECTION:
            model.fit(
                [x_train, batch_train], y_train, 
                epochs=10, batch_size=32, 
                validation_data=([x_test, batch_test], y_test), 
                verbose=config.VERBOSE
            )
            predictions = model.predict([x_test, batch_test], verbose=0).flatten()
        else:
            model.fit(
                x_train, y_train, 
                epochs=10, batch_size=32, 
                validation_data=(x_test, y_test), 
                verbose=config.VERBOSE
            )
            predictions = model.predict(x_test, verbose=0).flatten()

        # Calculate and print SMSE
        smse = np.sqrt(mean_squared_error(y_test, predictions))
        if kfold:
            print(f"Fold {fold} SMSE: {smse}")
        else:
            print(f"Single run SMSE: {smse}")

        if kfold:
            # Plot graph for individual fold
            plot_graph(
                y_test, predictions, f"{col_name} - Fold {fold}", smse
            )

            # Detect and log outliers
            detect_outliers(
                x_test, y_test, predictions, f"{col_name} - Fold {fold}", mode
            )

            # Store for the combined graph
            all_y_true.extend(y_test)
            all_y_pred.extend(predictions)
            all_seqs.extend(x_test)

            fold += 1
        
        else:
            # Graph and outliers for single run and end loop
            plot_graph(y_test, predictions, col_name, smse)
            detect_outliers(x_test, y_test, predictions, col_name, mode)
            return

    # Final combined graph after all folds
    total_smse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    plot_graph(
        np.array(all_y_true), np.array(all_y_pred), 
        f"{col_name} - Combined 5-Fold", total_smse
    )

    # Final outlier after all folds
    detect_outliers(
        all_seqs, all_y_true, all_y_pred, f"{col_name} - Combined", mode
    )