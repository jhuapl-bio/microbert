# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC


import os
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn.functional as F
from transformers import Trainer
from datasets import load_from_disk
from datasets import Dataset
from safetensors.torch import load_file, load_model, save_model

from analysis.experiment.utils.train_logger import logger

# suppress scikit learn warning about recall
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.metrics._ranking"
)


def is_directory_non_empty(directory_path):
    """Check if a given directory is non-empty."""
    return (
        os.path.exists(directory_path)
        and os.path.isdir(directory_path)
        and bool(os.listdir(directory_path))
    )


def load_dataset_if_available(path, dataset_name):
    """Load dataset from disk if available; otherwise, return None."""
    if is_directory_non_empty(path):
        logger.info(f"Loading tokenized {dataset_name} dataset from: {path}")
        dataset = load_from_disk(path)
        logger.info(f"{dataset_name.capitalize()} dataset size: {len(dataset)}")
        return dataset
    return None


# Save datasets to disk
def save_dataset(dataset, path, dataset_name):
    """Save dataset to disk if it exists."""
    if dataset:
        dataset.save_to_disk(path)
        logger.info(f"{dataset_name.capitalize()} dataset saved at: {path}")
        logger.info(f"{dataset_name.capitalize()} dataset size: {len(dataset)}")


def custom_label_binarize(y, classes):
    binarized = label_binarize(y, classes=classes)
    return (
        binarized if binarized.shape[1] > 1 else np.hstack([1 - binarized, binarized])
    )
    

def dataset_split(dataset, test_size, config, random_seed=42):
    if config.stratify:
        logger.info(f"Stratifying by column {config.stratify}")
        try:
            strat_col = f"label_level_{config.stratify}" # using data processor convention
            # make label column a HF class label column
            dataset = dataset.class_encode_column(strat_col)
            train_test_split = dataset.train_test_split(
                        test_size=test_size, seed=random_seed, stratify_by_column=strat_col
                    )
        except Exception as e:
            logger.info(f"Error {e} encountered stratifying by column {config.stratify}...splitting set by random sampling instead")
            train_test_split = dataset.train_test_split(
                    test_size=test_size, seed=random_seed, stratify_by_column=None
                )
    else:
        logger.info(f"Splitting set by randomly sampling")
        train_test_split = dataset.train_test_split(
            test_size=test_size, seed=random_seed, stratify_by_column=None
        )

    return train_test_split

class SafetensorsTrainer(Trainer):
    """
    Need to use custom Trainer class SafetensorsTrainer for training HyenaDNA model due to shared memory across NN Layers
    """

    def save_model(self, output_dir=None, _internal_call=True, safe_serialization=True):
        """
        Override the save_model method to save using safetensors.
        """
        output_dir = output_dir or self.args.output_dir
        if self.model is not None:
            # Ensure the directory exists
            os.makedirs(output_dir, exist_ok=True)
            # Save the model using safetensors
            model_path = os.path.join(output_dir, "model.safetensors")
            save_model(self.model, model_path)
        else:
            raise RuntimeError(
                "No model found to save. Ensure that the model is initialized and passed to the trainer."
            )

    def _load_best_model(self):
        """
        Override the loading of the best model to use safetensors.
        """
        if self.state.best_model_checkpoint is not None:
            model_path = os.path.join(
                self.state.best_model_checkpoint, "model.safetensors"
            )
            load_model(self.model, model_path)
        else:
            raise RuntimeError(
                "No best model checkpoint found. Ensure that training has completed and checkpoints exist."
            )


def load_model_weights(model, model_dir, model_weights_file = "model.safetensors"):
    """
    Load model weights from a directory containing a safetensors file or checkpoint subdirectories.
    If no valid weights are found, returns the base model and checkpoint number as 0.

    Args:
        model: The model instance to which weights will be loaded.
        model_dir (str): Path to a directory containing a safetensors file or checkpoint subdirectories.

    Returns:
        tuple: (model, checkpoint_number)
            - model: The model with loaded weights, or the base model if no checkpoint exists.
            - checkpoint_number: The checkpoint number if loaded, otherwise 0.
    """
    # Ensure model_dir is a valid directory
    if not os.path.isdir(model_dir):
        logger.warning(
            f"{model_dir} is not a valid directory. Returning base model with checkpoint 0."
        )
        return model, 0  # Return base model with checkpoint 0

    logger.info(f"Searching for weights in directory: {model_dir}")

    # First, try loading from safetensors file directly in model_dir
    safetensors_file = os.path.join(model_dir, model_weights_file)
    if os.path.exists(safetensors_file):
        logger.info(
            f"Loading final model weights from safetensors file: {safetensors_file}"
        )
        try:
            state_dict = load_file(safetensors_file)
            model.load_state_dict(state_dict, strict=False)
            logger.info("Weights successfully loaded into the model.")
            return model, -1  # No explicit checkpoints since all training completed
        except Exception as e:
            logger.error(f"Failed to load safetensors file: {e}")
            return model, 0  # Return base model

    # Search for checkpoint directories if no safetensors file is found
    logger.info("No safetensors file found. Searching for checkpoint directories...")

    # Find all 'checkpoint-NUMBER' directories
    checkpoints_list = [
        f for f in os.listdir(model_dir) if re.match(r"checkpoint-\d+$", f)
    ]
    checkpoints_dict = {int(k.split("-")[-1]): k for k in checkpoints_list}
    num_epochs_trained = len(checkpoints_list)

    if not checkpoints_dict:
        logger.warning("No checkpoints found. Returning base model with checkpoint 0.")
        return model, 0  # Return base model

    # Load from the latest checkpoint directory
    max_checkpoint = max(checkpoints_dict.keys())
    checkpoint_dir = os.path.join(model_dir, f"checkpoint-{max_checkpoint}")
    checkpoint_file = os.path.join(checkpoint_dir, model_weights_file)

    if not os.path.exists(checkpoint_file):
        logger.warning(
            f"No model.safetensors file found in latest checkpoint: {checkpoint_dir}. Returning base model with checkpoint 0."
        )
        return model, 0  # Return base model

    logger.info(f"Loading weights from the latest checkpoint: {checkpoint_file}")
    try:
        state_dict = load_file(checkpoint_file)
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Weights successfully loaded from checkpoint {max_checkpoint}.")
        return model, num_epochs_trained  # Return model with num epochs trained already
    except Exception as e:
        logger.error(f"Failed to load checkpoint weights: {e}")

    return model, 0  # Return base model with checkpoint 0 if loading fails


def multiclass_metrics(labels, predictions, suffix=None):
    """get multiclass metrics with labels and predictions"""

    # Initialize metrics dictionary
    metrics = {}

    # Compute accuracy metrics
    metrics["metric_accuracy"] = accuracy_score(labels, predictions)
    metrics["metric_balanced_accuracy"] = balanced_accuracy_score(labels, predictions)

    # Compute precision, recall, f1, support (macro)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    metrics["metric_precision_macro"] = macro_precision
    metrics["metric_recall_macro"] = macro_recall
    metrics["metric_f1_macro"] = macro_f1

    # Compute precision, recall, f1, support (micro)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="micro", zero_division=0
    )
    metrics["metric_precision_micro"] = micro_precision
    metrics["metric_recall_micro"] = micro_recall
    metrics["metric_f1_micro"] = micro_f1

    # Compute precision, recall, f1, support (weighted)
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            labels, predictions, average="weighted", zero_division=0
        )
    )
    metrics["metric_precision_weighted"] = weighted_precision
    metrics["metric_recall_weighted"] = weighted_recall
    metrics["metric_f1_weighted"] = weighted_f1

    # Add suffix to keys if provided
    if suffix:
        metrics = {f"{key}_{suffix}": value for key, value in metrics.items()}

    return metrics


def compute_metrics_hierarchical(eval_pred):
    """
    Compute metrics for a hierarchical classification problem.

    Args:
        eval_pred: A tuple containing logits (list of arrays) and labels (array).
            - logits: List of numpy arrays, each corresponding to a level.
            - labels: Numpy array of shape (num_samples, num_levels).

    Returns:
        dict: A dictionary containing metrics for each level.
    """
    # Unpack logits and labels
    logits, labels = eval_pred

    # Initialize dictionary to store metrics for each level
    metrics = {}

    # Iterate over each level's logits and labels
    for idx in range(len(logits)):
        # Get predictions for the current level
        predictions = np.argmax(logits[idx], axis=1)
        actual_labels = labels[:, idx]

        # specify suffix for each rank
        metrics_rank = multiclass_metrics(
            actual_labels, predictions, suffix=f"level_{idx+1}"
        )
        # Combine metrics with metrics_rank
        metrics.update(metrics_rank)

    return metrics


def compute_metrics_classification(eval_pred):
    """
    Compute metrics (accuracy, precision, recall, F1) for a classification problem.

    Args:
        eval_pred: A tuple containing logits and labels.
            - logits: Numpy array of shape (num_samples, num_classes).
            - labels: Numpy array of shape (num_samples,).

    Returns:
        dict: A dictionary containing metrics .
    """
    # Unpack logits and labels
    logits, labels = eval_pred

    # Convert logits to predicted classes
    predictions = np.argmax(logits, axis=1)
    metrics = multiclass_metrics(labels, predictions)

    return metrics


def extract_and_plot_training_metrics(trainer, save_directory):
    """
    Extracts and saves metrics from a trainer's log history and plots validation results.

    Args:
        trainer: The Hugging Face Trainer object.
        save_directory: The directory where results and plots will be saved.

    Returns:
        None
    """

    eval_log = [entry for entry in trainer.state.log_history if "eval_loss" in entry]
    train_log = [entry for entry in trainer.state.log_history if "loss" in entry]
    eval_log_df = pd.DataFrame(eval_log)
    train_log_df = pd.DataFrame(train_log)
    history_log = train_log_df.merge(eval_log_df, on=["epoch", "step"])
    # clean up column names
    history_log_clean = history_log.rename(
        columns=lambda col: col.replace("eval_metric", "Metric")
        .replace("_", " ")
        .title()
    )
    history_log_clean = history_log_clean.rename(
        columns={"Loss": "Training Loss", "Eval Loss": "Validation Loss"}
    )

    history_log_clean.to_csv(
        os.path.join(save_directory, "model_training_history.csv"), index=False
    )

    plt.figure()
    plt.plot(
        history_log_clean["Epoch"],
        history_log_clean["Training Loss"],
        label="Training Loss",
    )
    plt.plot(
        history_log_clean["Epoch"],
        history_log_clean["Validation Loss"],
        label="Validation Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # Customize x-axis to show only integer values
    plt.xticks(
        ticks=range(
            int(min(history_log_clean["Epoch"])),
            int(max(history_log_clean["Epoch"])) + 1,
        )
    )
    plt.legend()
    plt.savefig(os.path.join(save_directory, "Learning Curve.png"))
    # do not show for simplicity
    # plt.show()
    plt.close()  # Close the plot to prevent overlap in successive iterations

    plots_dir = os.path.join(save_directory, "Validation Curves")
    os.makedirs(plots_dir, exist_ok=True)

    # hugging face prepends "eval" to metric name
    for col in history_log_clean.columns:
        if "Metric" in col:
            name_col = col[7:]  # don't have Metric in Title
            plt.figure()
            plt.title(name_col)
            plt.plot(history_log_clean["Epoch"], history_log_clean[col])
            plt.xlabel("Epoch")
            plt.ylabel(name_col)
            # Customize x-axis to show only integer values
            plt.xticks(
                ticks=range(
                    int(min(history_log_clean["Epoch"])),
                    int(max(history_log_clean["Epoch"])) + 1,
                )
            )
            plt.savefig(os.path.join(plots_dir, f"Validation {name_col}.png"))
            # do not show for simplicity
            # plt.show()
            plt.close()  # Close the plot to prevent overlap in successive iterations


def evaluate_and_save_model_metrics(trainer, dataset, data_processor, config):
    """
    Evaluates the model on a given dataset, computes performance metrics, and saves the results
    """
    data_filename = config.testing_data
    save_directory = config.test_metrics_dir
    classification = config.classification
    model_type = config.model_type
    top_k = config.top_k

    # Run predictions on the test dataset and get metrics
    if data_filename is None:
        data_filename = "training_subset"
    else:
        data_filename = Path(data_filename).parts[-1]
        data_filename = re.sub(".csv|.tsv", "", data_filename)

    # get model predictions on test set, assumes "labels" is a column
    start_time = time.time()
    if model_type == "HYENA":
        # no attention mask column for Hyena
        test_results = trainer.predict(dataset.select_columns(["labels", "input_ids"]))
    else:
        test_results = trainer.predict(
            dataset.select_columns(["labels", "input_ids", "attention_mask"])
        )
    eval_time = time.time() - start_time
    logger.info(f"Test set evaluation completed in {eval_time:.2f} seconds.")

    # get labels
    labels = dataset["labels"]
    data_index = dataset.to_pandas().index

    # store metrics for all columns/ranks
    metrics = {}

    cols = []
    if classification:
        cols = [data_processor.label_column]
    else:
        cols = data_processor.taxonomic_ranks

    # Iterate through each level's logits and encoders by index to ensure correct order
    for idx, col in enumerate(cols):
        # Apply softmax to get probabilities for the current level
        if classification:
            probs = F.softmax(torch.tensor(test_results.predictions), dim=-1).cpu()
        else:
            probs = F.softmax(torch.tensor(test_results.predictions[idx]), dim=-1).cpu()

        # Create a DataFrame for probabilities across all classes
        probs_df = pd.DataFrame(
            probs,
            index=data_index,
            columns=data_processor.encoders[col].classes_,
        )
        # Get the predicted class (highest probability) for the current level
        predictions_raw = torch.argsort(-1 * probs, dim=-1)[:, 0:top_k].cpu()
        # Remove the extra dimension and convert to pandas series
        predictions = pd.Series(predictions_raw.squeeze(dim=1))
        # Convert to human-readable text predictions
        predictions_text = data_processor.encoders[col].inverse_transform(predictions)

        if classification:
            actual_labels = labels
        else:
            actual_labels = [sublist[idx] for sublist in labels]

        actual_labels_text = data_processor.encoders[col].inverse_transform(
            actual_labels
        )

        metrics_per_col = multiclass_metrics(actual_labels, predictions, suffix=None)

        # threshold-based average precision
        y_true_binarized = custom_label_binarize(
            actual_labels_text, classes=data_processor.encoders[col].classes_
        )

        metrics_per_col["metric_average_precision_macro"] = average_precision_score(
            y_true_binarized, probs, average="macro"
        )
        metrics_per_col["metric_average_precision_micro"] = average_precision_score(
            y_true_binarized, probs, average="micro"
        )
        metrics_per_col["metric_average_precision_weighted"] = average_precision_score(
            y_true_binarized, probs, average="weighted"
        )

        # List to store reciprocal ranks for each sample at this taxonomic rank
        reciprocal_ranks = []

        # Iterate over each sample
        for x in range(len(actual_labels)):
            y_true = actual_labels_text[x]  # Ground truth label for the sample
            y_pred_probs = probs_df.iloc[
                x
            ].to_dict()  # Predicted probabilities {class: probability}

            # Sort classes by predicted probabilities in descending order
            sorted_classes = [
                cls
                for cls, prob in sorted(
                    y_pred_probs.items(), key=lambda item: item[1], reverse=True
                )
            ]

            # Find the rank of the true class
            try:
                rank = (
                    sorted_classes.index(y_true) + 1
                )  # Adding 1 because ranks start from 1
                reciprocal_rank = 1 / rank
            except ValueError:
                # If true class is not among the predicted classes
                reciprocal_rank = 0.0

            reciprocal_ranks.append(reciprocal_rank)

        # Calculate the MRR for this taxonomic level
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

        metrics_per_col["metric_MRR"] = mrr

        results_df = pd.DataFrame(
            {"Predicted_Label": predictions_text, "Actual_Label": actual_labels_text},
            index=data_index,
        )

        # save results
        probs_save_file = os.path.join(save_directory, f"{col}_probabilities.csv")
        logger.info(f"Saving probabilities for column {col} to {probs_save_file}")
        probs_df.to_csv(probs_save_file)

        results_save_file = os.path.join(save_directory, f"{col}_predictions.csv")
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

    metric_save_file = os.path.join(save_directory, "test_metrics_summary.csv")
    logger.info(f"Saving summary metrics to {metric_save_file}")
    df_metrics.to_csv(metric_save_file)

    return cleaned_metrics
