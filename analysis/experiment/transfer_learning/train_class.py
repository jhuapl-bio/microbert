# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


def main(config: Config):
    filename = "transfer"
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
        h5_file_path=embeddings_save_path,
        group_name="train",
    )
    # val_embeddings, val_labels = read_group_in_batches(
    #     h5_file_path=embeddings_save_path, group_name="val"
    # )
    test_embeddings, test_labels = read_group_in_batches(
        h5_file_path=embeddings_save_path,
        group_name="test",
    )
    # find columns for labels
    cols = data_processor.labels

    # parse out only desired label from saved hierarchical embeddings
    for idx, col in enumerate(cols):
        if col == "genus":
            train_labels = np.array(train_labels)[:, idx]
            test_labels = np.array(test_labels)[:, idx]

    # define model save path
    model_save_dir = os.path.join(config.save_dir, "transfer")
    os.makedirs(model_save_dir, exist_ok=True)

    # Define classifiers with hyperparameter grids
    models = {
        "LogisticRegression": {
            "model": LogisticRegression(
                random_state=42,
                max_iter=10000,
                class_weight="balanced",
                penalty="l2",
                n_jobs=-1,
            ),
            "param_grid": {"model__C": [0.01, 0.1, 1, 10, 100]},
        },
        # "RandomForest": {
        #     "model": RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=100),
        #     "param_grid": {
        #         "model__n_estimators": [100, 200],
        #         "model__max_depth": [None, 10, 20],
        #         "model__min_samples_split": [2, 5]
        #     }
        # },
        # "LightGBM": {
        #     "model": LGBMClassifier(objective="multiclass",
        #                             class_weight='balanced', # only valid for multiclass problems, not binary classification
        #                             random_state=42,
        #                             n_jobs=-1),
        #     "param_grid": {
        #         "model__num_leaves": [31, 50, 100],
        #         "model__learning_rate": [0.01, 0.1, 0.3],
        #         "model__n_estimators": [100, 200, 500]
        #     }
        # },
        "HistGradientBoosting": {
            "model": HistGradientBoostingClassifier(
                random_state=42,
                class_weight="balanced",
            ),  # no n_jobs parameter for HistGradientBoostingClassifiers
            "param_grid": {
                "model__learning_rate": [
                    0.01,
                    0.1,
                ],
                "model__max_iter": [100, 200],
                "model__max_depth": [None, 10, 20],
                # "model__min_samples_leaf": [1, 5, 10]
            },
        },
    }

    # Dictionary to store the best model from each model type
    best_models = {}
    best_overall_model = None
    best_overall_score = -np.inf
    best_overall_model_name = ""

    # Define stratified k-fold cross-validation
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Perform hyperparameter optimization for each model
    for model_name, model_dict in models.items():
        model_filename = os.path.join(model_save_dir, f"{model_name}_best.pkl")

        if os.path.exists(model_filename):
            logger.info(f"Already trained model {model_name} and saved...skipping")
            continue

        logger.info(f"Tuning hyperparameters for {model_name}")

        pipeline = Pipeline([("model", model_dict["model"])])

        search = GridSearchCV(
            pipeline,
            model_dict["param_grid"],
            scoring="balanced_accuracy",
            cv=cv_strategy,  # Ensure class distribution in folds
            n_jobs=-1,  # If set to -1, all CPUs are used
        )

        search.fit(
            train_embeddings, train_labels
        )  # Train on train set, validate on val set

        best_pipeline = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_

        logger.info(f"Best score for {model_name}: {best_score:.4f}")
        logger.info(f"Best Parameters for {model_name}: {best_params}")

        # Save the best model for this type
        best_models[model_name] = best_pipeline
        with open(model_filename, "wb") as f:
            pickle.dump(best_pipeline, f)
        logger.info(f"Saved best {model_name} model to {model_filename}")

        # Check if this model is the best overall
        if best_score > best_overall_score:
            best_overall_model = best_pipeline
            best_overall_score = best_score
            best_overall_model_name = model_name

    all_metrics = {}

    # Evaluate each saved model
    for model_name in models.keys():
        model_filename = os.path.join(model_save_dir, f"{model_name}_best.pkl")
        if not os.path.exists(model_filename):
            logger.warning(f"Model file for {model_name} not found...skipping.")
            continue

        logger.info(f"Loading model {model_name} from {model_filename}")
        with open(model_filename, "rb") as f:
            model = pickle.load(f)

        logger.info(f"Evaluating model {model_name} on test set...")
        predictions = model.predict(test_embeddings)

        for idx, col in enumerate(cols):
            if col != "genus":
                continue

            actual_labels_text = data_processor.encoders[col].inverse_transform(
                np.array(test_labels).astype(int)
            )
            predictions_text = data_processor.encoders[col].inverse_transform(
                np.array(predictions).astype(int)
            )

            metrics_per_col = multiclass_metrics(
                actual_labels_text, predictions_text, suffix=None
            )

            results_df = pd.DataFrame(
                {
                    "Predicted_Label": predictions_text,
                    "Actual_Label": actual_labels_text,
                }
            )

            results_save_file = os.path.join(
                config.test_metrics_dir, f"{model_name}_{col}_predictions.csv"
            )
            logger.info(
                f"Saving predictions for {model_name} ({col}) to {results_save_file}"
            )
            results_df.to_csv(results_save_file)

            if model_name not in all_metrics:
                all_metrics[model_name] = {}

            all_metrics[model_name][col] = metrics_per_col

    # store metrics for all columns/ranks
    metrics = {}

    # Iterate through each level's logits and encoders by index to ensure correct order
    for idx, col in enumerate(cols):
        # Convert to human-readable text predictions
        if col == "genus":
            actual_labels_text = data_processor.encoders[col].inverse_transform(
                np.array(test_labels).astype(int)
            )
            predictions_text = data_processor.encoders[col].inverse_transform(
                np.array(predictions).astype(int)
            )
            # else:
            #     if col != config.label_column:
            #         logger.info(f"Ignoring labels for {col}...")
            #         continue
            #     logger.info(f"Generating predictions for {col}...")
            #     # Convert to human-readable text predictions
            #     actual_labels_text = data_processor.encoders[col].inverse_transform(
            #         np.array(test_labels)[:, idx].astype(int)
            #     )
            #     predictions_text = data_processor.encoders[col].inverse_transform(
            #         predictions[:, idx].astype(int)
            #     )

            metrics_per_col = multiclass_metrics(
                actual_labels_text, predictions_text, suffix=None
            )

            results_df = pd.DataFrame(
                {
                    "Predicted_Label": predictions_text,
                    "Actual_Label": actual_labels_text,
                }
            )

            results_save_file = os.path.join(
                config.test_metrics_dir, f"{col}_predictions.csv"
            )
            logger.info(f"Saving predictions for column {col} to {results_save_file}")
            results_df.to_csv(results_save_file)

            # update all metrics
            metrics[col] = metrics_per_col

    # Clean and save metrics for each model
    for model_name, metrics_by_col in all_metrics.items():
        cleaned_metrics = {
            col: {
                metric.replace("metric", "")
                .replace("_", " ")
                .title()
                .replace("Macro", "(Macro)")
                .replace("Micro", "(Micro)")
                .replace("Weighted", "(Weighted)")
                .replace("Mrr", "MRR"): f"{value:.4f}"
                for metric, value in metric_dict.items()
            }
            for col, metric_dict in metrics_by_col.items()
        }

        df_metrics = pd.DataFrame.from_dict(cleaned_metrics, orient="index")
        metric_save_file = os.path.join(
            config.test_metrics_dir, f"{model_name}_test_metrics_summary.csv"
        )
        logger.info(f"Saving summary metrics for {model_name} to {metric_save_file}")
        df_metrics.to_csv(metric_save_file)

        for col, metrics_dict in cleaned_metrics.items():
            logger.info(
                f"Summary Metrics on test set for model {model_name}, column {col}:"
            )
            for metric, value in metrics_dict.items():
                logger.info(f"{metric}: {value}")

    # Convert to DataFrame and save
    df_metrics = pd.DataFrame.from_dict(cleaned_metrics, orient="index")

    metric_save_file = os.path.join(config.test_metrics_dir, "test_metrics_summary.csv")
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
