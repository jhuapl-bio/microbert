# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC

import itertools
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from analysis.experiment.utils.data_processor import DataProcessor
from analysis.experiment.utils.config import CONFIG_DIR, Config
from analysis.experiment.utils.embeddings_utils import read_group_in_batches
from analysis.experiment.utils.train_logger import add_file_handler, logger
from analysis.experiment.utils.train_utils import multiclass_metrics
from hiclass import LocalClassifierPerNode
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def main(config: Config):
    filename = "hiclass"
    add_file_handler(logger, config.save_dir, filename)
    logger.info(
        f"Logger {filename} initialized. Logs will be saved to {config.save_dir}"
    )

    # initializing data processor
    data_processor = DataProcessor(
        sequence_column=config.sequence_column,
        labels=config.labels,
        save_file=config.data_processor_filename,
    )
    # Must have a data processor already fit ina valid directory specified in config
    data_processor.load_processor(config.data_processor_path)

    # load generated embeddings and labels from train/val/test datasets

    embeddings_save_path = os.path.join(config.save_dir, "checkpoints.h5")

    logger.info(f"Fetching embeddings from {embeddings_save_path}")
    train_embeddings, train_labels = read_group_in_batches(
        h5_file_path=embeddings_save_path, group_name="train"
    )
    # val_embeddings, val_labels = read_group_in_batches(
    #     h5_file_path=embeddings_save_path, group_name="val"
    # )
    test_embeddings, test_labels = read_group_in_batches(
        h5_file_path=embeddings_save_path, group_name="test"
    )

    # find columns for labels
    cols = data_processor.labels

    # define model save path
    model_save_dir = os.path.join(config.save_dir, "hiclass_transfer")
    os.makedirs(model_save_dir, exist_ok=True)

    # Define models and hyperparameter grids
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(
                random_state=42, max_iter=10000, class_weight="balanced", penalty="l2"
            ),
            "param_grid": {"C": [0.01, 0.1, 1, 10, 100]},
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, class_weight="balanced"),
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
            },
        },
        # "LightGBM": {
        #     "model": LGBMClassifier(objective="multiclass",
        #                             class_weight='balanced',
        #                             random_state=42),
        #     "param_grid": {
        #         "num_leaves": [31, 50, 100],
        #         "learning_rate": [0.01, 0.1, 0.3],
        #         "n_estimators": [100, 200, 500]
        #     }
        # },
        "HistGradientBoosting": {
            "model": HistGradientBoostingClassifier(
                random_state=42, class_weight="balanced"
            ),
            "param_grid": {
                "learning_rate": [0.01, 0.1, 0.3],
                "max_iter": [100, 200, 500],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 5, 10],
            },
        },
    }

    # Dictionary to store the best model
    best_models = {}
    best_overall_model = None
    best_overall_score = -np.inf
    best_overall_model_name = None

    # Number of CV folds
    n_folds = 3
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Perform hyperparameter search for each model
    for model_name, model_dict in models.items():
        model_filename = os.path.join(model_save_dir, f"{model_name}_best.pkl")

        if os.path.exists(model_filename):
            logger.info(f"Already trained model {model_name} and saved...skipping")
            continue

        logger.info(f"Tuning hyperparameters for {model_name}")

        best_score = -float("inf")
        best_model = None
        best_params = None

        # Get hyperparameter grid
        param_grid = model_dict["param_grid"]
        param_names = list(param_grid.keys())
        param_combinations = [
            dict(zip(param_names, values))
            for values in itertools.product(*param_grid.values())
        ]

        for params in param_combinations:
            logger.info(f"Evaluating model: {model_name}")
            logger.info(f"Hyperparameters: {params}")

            fold_scores = []
            for train_index, val_index in kf.split(
                train_embeddings, train_labels[:, 0]
            ):  # Use first column for stratification
                X_train, X_val = (
                    train_embeddings[train_index],
                    train_embeddings[val_index],
                )
                y_train, y_val = train_labels[train_index], train_labels[val_index]

                # Set parameters for base classifier
                base_classifier = model_dict["model"]
                base_classifier_params = base_classifier.set_params(**params)

                # Create hierarchical classifier
                hierarchical_model_params = LocalClassifierPerNode(
                    local_classifier=base_classifier_params, n_jobs=os.cpu_count()
                )

                # Train on training fold
                hierarchical_model_params.fit(X_train, y_train)

                # Predict on validation fold
                val_predictions = hierarchical_model_params.predict(X_val)

                # Evaluate using balanced accuracy for the relevant label column
                for idx, col in enumerate(cols):

                    y_val_col = np.array(y_val)[:, idx].astype(str)
                    val_predictions_col = np.array(val_predictions)[:, idx].astype(str)

                    metrics_per_col = multiclass_metrics(
                        y_val_col, val_predictions_col, suffix=None
                    )
                    fold_scores.append(metrics_per_col["metric_balanced_accuracy"])

            # Compute mean score across folds
            avg_score = np.mean(fold_scores)
            logger.info(f"Average CV Score: {avg_score:.4f}")

            # Keep track of best model
            if avg_score > best_score:
                best_score = avg_score
                best_model = hierarchical_model_params
                best_params = params

        logger.info(f"Best score for {model_name}: {best_score:.4f}")
        logger.info(f"Best Parameters for {model_name}: {best_params}")

        # Save the best model for this type
        best_models[model_name] = best_model
        with open(model_filename, "wb") as f:
            pickle.dump(best_model, f)
        logger.info(f"Saved best {model_name} model to {model_filename}")

        # Check if this model is the best overall
        if best_score > best_overall_score:
            best_overall_model = best_model
            best_overall_score = best_score
            best_overall_model_name = model_name

    logger.info(
        f"Best Overall Model: {best_overall_model_name} with score: {best_overall_score:.4f}"
    )
    # Evaluate the best overall model on the test set
    logger.info(f"Evaluating {best_overall_model_name} on test set")
    predictions = best_overall_model.predict(test_embeddings)

    # store metrics for all columns/ranks
    metrics = {}

    # Iterate through each level's logits and encoders by index to ensure correct order
    for idx, col in enumerate(cols):
        logger.info(f"Generating predictions for {col}...")
        # Convert to human-readable text predictions
        actual_labels_text = data_processor.encoders[col].inverse_transform(
            np.array(test_labels)[:, idx].astype(int)
        )
        predictions_text = data_processor.encoders[col].inverse_transform(
            predictions[:, idx].astype(int)
        )

        metrics_per_col = multiclass_metrics(
            actual_labels_text, predictions_text, suffix=None
        )

        results_df = pd.DataFrame(
            {"Predicted_Label": predictions_text, "Actual_Label": actual_labels_text}
        )

        results_save_file = os.path.join(
            config.test_metrics_dir, f"hiclass_{col}_predictions.csv"
        )
        logger.info(f"Saving predictions for column {col} to {results_save_file}")
        results_df.to_csv(results_save_file)

        # update all metrics
        metrics[col] = metrics_per_col

    # Transform and clean metrics for each dictionary in the nested structure
    cleaned_metrics = {
        col: {
            metric.replace("metric", "")
            .replace("_", " ")
            .title()
            .replace("Macro", "(Macro)")
            .replace("Micro", "(Micro)")
            .replace("Weighted", "(Weighted)")
            .replace("Mrr", "MRR"): f"{value:.4f}"
            for metric, value in nested_metric.items()
        }
        for col, nested_metric in metrics.items()
    }

    # Convert to DataFrame and save
    df_metrics = pd.DataFrame.from_dict(cleaned_metrics, orient="index")

    metric_save_file = os.path.join(
        config.test_metrics_dir, "hiclass_test_metrics_summary.csv"
    )
    logger.info(f"Saving summary metrics to {metric_save_file}")
    df_metrics.to_csv(metric_save_file)

    for col, metrics_dict in cleaned_metrics.items():
        logger.info(f"Summary Metrics on test set for column {col}:")
        for metric, value in metrics_dict.items():
            logger.info(f"{metric}: {value}")

    logger.info("All done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    # Add a positional argument for the config path
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML configuration file.",
        default="bertax/sample/train_embeddings.yaml",
        nargs="?",  # Makes it optional with a default value
    )
    args = parser.parse_args()

    # Access the config path
    config_file = args.config_file
    config_path = os.path.join(CONFIG_DIR, config_file)
    logger.info(f"Using configuration file: {config_path}")
    config = Config(config_path)

    # Get all attributes as a dictionary
    attributes = vars(config)
    for key in attributes:
        logger.info(f"{key}: {attributes[key]}")

    main(config)
