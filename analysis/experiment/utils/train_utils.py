# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC

import os
import re
import glob
import time
import math
import psutil
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
)

from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback
from datasets import load_from_disk
from safetensors.torch import load_file, load_model, save_model

from analysis.experiment.utils.train_logger import logger

# suppress scikit learn warning about recall
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.metrics._ranking"
)

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


class EpochTimingCallback(TrainerCallback):
    def __init__(self, epochs_trained_file: str):
        self.epoch_times = []
        self.epochs_trained_file = epochs_trained_file
        self.epochs_trained = self._load_or_initialize_epochs_trained()
        logger.info(f"Reading in total epochs persisted: {self.epochs_trained}")

    def _load_or_initialize_epochs_trained(self) -> int:
        if os.path.exists(self.epochs_trained_file):
            try:
                with open(self.epochs_trained_file, "r") as f:
                    value = f.read().strip()
                    if not value:
                        logger.warning("Epochs file is empty. Writing 0.")
                        self._write_epochs_trained(0)
                        return 0
                    return int(value)
            except Exception as e:
                logger.warning(f"Failed to read epochs_trained from file: {e}")
                self._write_epochs_trained(0)
                return 0
        else:
            logger.info("Epochs file not found. Initializing with 0.")        
            self._write_epochs_trained(0)
            return 0

    def _write_epochs_trained(self, value: int):
        try:
            with open(self.epochs_trained_file, "w") as f:
                f.write(f"{value}")
        except Exception as e:
            logger.error(f"Failed to write epochs_trained to file: {e}")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        end_time = time.time()
        epoch_time = end_time - self.start_time
        self.epoch_times.append(epoch_time)
        
        self.epochs_trained += 1
        self._write_epochs_trained(self.epochs_trained)
        logger.info(f"Updated epochs trained to {self.epochs_trained}")
        logger.info(f"Epoch {state.epoch:.0f} completed in {epoch_time:.2f} seconds.")        

class LogMetricsCallback(TrainerCallback):
    """
    A custom callback to log metrics during the training loop.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:  # Check if there are logs to process
            for key, value in logs.items():
                logger.info(f"{key}: {value}")

class MemoryDebugCallback(TrainerCallback):
    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps
        self.last_mem = None  # Track memory from the last log
        self.last_time = None  # Track time of the last log

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every_n_steps == 0:
            # Time tracking
            current_time = time.time()
            if self.last_time is not None:
                elapsed = current_time - self.last_time
                logger.info(f"[Step {state.global_step}] Time for last {self.log_every_n_steps} steps: {elapsed:.2f} seconds")
            self.last_time = current_time

            # Memory tracking
            process = psutil.Process()
            mem = process.memory_info().rss // (1024 ** 2)  # in MB
            if self.last_mem is None or mem != self.last_mem:
                logger.info(f"[Step {state.global_step}] Memory used: {mem} MB")
                self.last_mem = mem

            gc.collect()


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
    """Save tokenized HF dataset to disk if it exists."""
    if dataset:
        dataset.save_to_disk(path)
        logger.info(f"{dataset_name.capitalize()} dataset saved at: {path}")
        logger.info(f"{dataset_name.capitalize()} dataset size: {len(dataset)}")


def custom_label_binarize(y, classes):
    binarized = label_binarize(y, classes=classes)
    return (
        binarized if binarized.shape[1] > 1 else np.hstack([1 - binarized, binarized])
    )

def sort_file_paths_by_suffix_number(file_paths):
    return sorted(
        file_paths,
        key=lambda x: int(re.search(r'_(\d+)\.csv$', os.path.basename(x)).group(1))
    )

def dataset_split(df, test_size, config, random_seed=42):
    """
    Splits a DataFrame into train and test sets, optionally using stratification.

    Args:
        df (pd.DataFrame): Input dataset.
        test_size (float or int): Proportion (or count) of test samples.
        config: Config object with an optional `stratify` attribute specifying the stratification column.
        random_seed (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    if config.stratify:
        strat_col = f"label_level_{config.stratify}"
        logger.info(f"Stratifying by column '{strat_col}'")
        try:
            stratify_values = df[strat_col]
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_seed,
                stratify=stratify_values,
            )
        except Exception as e:
            logger.warning(
                f"Error '{e}' encountered during stratification by '{strat_col}'. Falling back to random split."
            )
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_seed,
                stratify=None,
            )
    else:
        logger.info(
            "No stratification specified. Splitting dataset by random sampling."
        )
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_seed,
            stratify=None,
        )

    return train_df, test_df


def count_csv_rows_fast(file_path, has_header=True):
    with open(file_path, "r", encoding="utf-8") as f:
        count = sum(1 for _ in f)  # reads one line at a time, never loads whole file
    return count - 1 if has_header else count


def calculate_steps_per_epoch(
    dataset_size, per_device_batch_size, num_devices=1, accumulation_steps=1
):
    effective_batch_size = per_device_batch_size * num_devices * accumulation_steps
    steps_per_epoch = math.ceil(dataset_size / effective_batch_size)
    return steps_per_epoch


def calculate_max_steps(
    dataset_size,
    per_device_batch_size,
    num_devices=1,
    accumulation_steps=1,
    num_epochs=1,
):
    steps_per_epoch = calculate_steps_per_epoch(
        dataset_size, per_device_batch_size, num_devices, accumulation_steps
    )
    max_steps = steps_per_epoch * num_epochs
    return max_steps


def load_model_weights(model, model_dir, model_weights_file="model.safetensors"):
    """
    Load model weights from a directory containing a safetensors file or checkpoint subdirectories.
    If no valid weights are found, returns the base model and checkpoint number as 0.

    Args:
        model: The model instance to which weights will be loaded.
        model_dir (str): Path to a directory containing a safetensors file or checkpoint subdirectories.

    Returns:
        tuple: (model, checkpoint_number)
            - model: The model with loaded weights, or the base model if no checkpoint exists.
            - num_checkpoints_saved: The nunber of checkpoints saved and, thus latest loaded, otherwise 0.
    """
    # Ensure model_dir is a valid directory
    if not os.path.isdir(model_dir):
        logger.warning(
            f"{model_dir} is not a valid directory. Returning base model with checkpoint 0."
        )
        return model, 0  # Return base model with checkpoint 0

    logger.info(f"Searching for weights in directory: {model_dir}")

    # First, try loading from safetensors file directly in model_dir
    # This only exists if training has fully completed
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
    logger.info("No fully trained safetensors file found. Searching for checkpoint directories instead...")

    # Find all 'checkpoint-NUMBER' directories
    checkpoints_list = [
        f for f in os.listdir(model_dir) if re.match(r"checkpoint-\d+$", f)
    ]
    checkpoints_dict = {int(k.split("-")[-1]): k for k in checkpoints_list}
    num_checkpoints_saved = len(checkpoints_list)
    logger.info(f"Found {num_checkpoints_saved} existing model cheeckpoints")
    
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
        return model, num_checkpoints_saved  # Return model with num_checkpoints_saved
    except Exception as e:
        logger.error(f"Failed to load checkpoint weights: {e}")

    return model, 0  # Return base model with checkpoint 0 if loading fails

def multiclass_metrics(labels, predictions, suffix=None):
    """get multiclass metrics with labels and predictions"""

    # Initialize metrics dictionary
    metrics = {}

    # Compute accuracy metrics
    metrics["metric_accuracy"] = accuracy_score(labels, predictions)
    metrics["metric_balanced_accuracy"] = balanced_accuracy_score(labels, predictions) # automatically restricst to classes present in truth

    
    # Restrict to classes present in ground truth only
    present_classes = np.unique(labels)

    # Macro-averaged precision, recall, f1
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=present_classes,
        average="macro",
        zero_division=0
    )
    metrics["metric_precision_macro"] = macro_precision
    metrics["metric_recall_macro"] = macro_recall
    metrics["metric_f1_macro"] = macro_f1

    # Micro-averaged precision, recall, f1
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=present_classes,
        average="micro",
        zero_division=0
    )
    metrics["metric_precision_micro"] = micro_precision
    metrics["metric_recall_micro"] = micro_recall
    metrics["metric_f1_micro"] = micro_f1

    # Weighted-averaged precision, recall, f1
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=present_classes,
        average="weighted",
        zero_division=0
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
            plt.close()


def evaluate_model(trainer, dataset, data_processor, config):
    """
    Evaluates the model on a given dataset saves the results, either in total or in batches
    """
    model_type = config.model_type
    cols = config.labels  # list of labels to use
    save_directory = config.test_metrics_dir
    top_k = 1 # always use top_k = 1
    
    # unset any defined prediction_loss_only set for validation loop
    trainer.args.prediction_loss_only = False

    cols_select = ["labels", "input_ids"]
    if model_type != "HYENA":
        # no attention mask column for Hyena
        cols_select.append("attention_mask")

    start_time = time.time()
    
    # Run predictions on the test dataset    
    if config.predictions_batch is not None:
        chunk_size = int(config.predictions_batch)
        # Chunked prediction
        num_chunks = len(dataset) // chunk_size + int(len(dataset) % chunk_size != 0)
        logger.info(f"Running evaluations for {num_chunks} chunks...")
        for i in tqdm(range(num_chunks), desc="Running chunked prediction"):
            # get chunk
            logger.info(f"Getting model evaluations for chunk {i}...")
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(dataset))
            dataset_chunk = dataset.select(range(start_idx, end_idx))
            
            # get labels for chunk
            data_index = dataset_chunk.to_pandas().index
            labels = dataset_chunk["labels"]
            
            # predict for chunk
            chunk_results = trainer.predict(dataset_chunk.select_columns(cols_select))
            # Iterate through each level's logits and encoders by index to ensure correct order
            for idx, col in enumerate(cols):
                # List to store reciprocal ranks for each sample at this label
                reciprocal_ranks = []
                # Apply softmax to get probabilities for the current level
                probs = F.softmax(torch.tensor(chunk_results.predictions[idx]), dim=-1).cpu()
            
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
                
                # get actual labels
                actual_labels = [sublist[idx] for sublist in labels]
                # get actual labels text
                actual_labels_text = data_processor.encoders[col].inverse_transform(
                    actual_labels
                )
            
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
                results_df = pd.DataFrame(
                    {"Predicted_Label": predictions_text, "Actual_Label": actual_labels_text, "Reciprocal_Rank": reciprocal_ranks},
                    index=data_index,
                )
                # save predictions results
                results_save_file = os.path.join(save_directory, f"{col}_predictions_{i}.csv")
                logger.info(f"Saving predictions for column {col} to {results_save_file}")
                results_df.to_csv(results_save_file)

                # check if want to save probabilities results
                if bool(config.save_probabilities):
                    probs_save_file = os.path.join(save_directory, f"{col}_probabilities_{i}.csv")
                    logger.info(f"Saving probabilities for column {col} to {probs_save_file}")
                    probs_df.to_csv(probs_save_file)
            
    else:
        # run in full memory without batching test set
        test_results = trainer.predict(
                dataset.select_columns(cols_select)
            )
        # get labels
        labels = dataset["labels"]
        data_index = dataset.to_pandas().index
        # Iterate through each level's logits and encoders by index to ensure correct order
        for idx, col in enumerate(cols):
            reciprocal_ranks = []
            # Apply softmax to get probabilities for the current level
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
            
            # get actual labels
            actual_labels = [sublist[idx] for sublist in labels]
            # get actual labels text
            actual_labels_text = data_processor.encoders[col].inverse_transform(
                actual_labels
            )
        
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
                
            results_df = pd.DataFrame(
                {"Predicted_Label": predictions_text, "Actual_Label": actual_labels_text, "Reciprocal_Rank": reciprocal_ranks},
                index=data_index,
            )
            # save predictions results
            results_save_file = os.path.join(save_directory, f"{col}_predictions.csv")
            logger.info(f"Saving predictions for column {col} to {results_save_file}")
            results_df.to_csv(results_save_file)

            # check if want to save probabilities results
            if bool(config.save_probabilities):
                logger.info(f"Saving probabilities for column {col}")
                probs_save_file = os.path.join(save_directory, f"{col}_probabilities.csv")
                logger.info(f"Saving probabilities for column {col} to {probs_save_file}")
                probs_df.to_csv(probs_save_file)
    
    eval_time = time.time() - start_time
    logger.info(f"Test set evaluation completed in {eval_time:.2f} seconds.")


def save_model_metrics(data_processor, config):
    # Collects and saves evaluation metrics for each classification col
    
    cols = config.labels  # list of labels
    save_directory = config.test_metrics_dir

    # initialize full metrics_dic
    metrics = {}
    
    for idx, col in enumerate(cols):
        if config.predictions_batch is not None:
            # if batching testing, read in individual chunk results files and concat
            pattern = os.path.join(save_directory, f"{col}_predictions_*.csv")
            list_results_files = glob.glob(pattern)
            num_result_files = len(list_results_files)
            list_results_files = sort_file_paths_by_suffix_number(list_results_files)
            
            logger.info(f"Found {num_result_files} chunked predictions files...")
            df_results_full = pd.DataFrame()
            for results_save_file in list_results_files:
                logger.info(f"Concating chunked test results found in {results_save_file}")
                df_results = pd.read_csv(results_save_file)
                df_results_full = pd.concat([df_results_full, df_results])
        else:
            results_save_file = os.path.join(save_directory, f"{col}_predictions.csv")
            df_results_full = pd.read_csv(results_save_file)

        # Extract columns
        actual_labels_text = df_results_full['Actual_Label']
        predicted_label = df_results_full['Predicted_Label']
        reciprocal_ranks = df_results_full['Reciprocal_Rank'].astype(float)

        # Calculate classification metrics
        metrics_per_col = multiclass_metrics(actual_labels_text, predicted_label, suffix=None)

        # Calculate MRR
        if len(reciprocal_ranks) > 0:
            mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        else:
            mrr = 0.0
            
        metrics_per_col["metric_MRR"] = mrr

        # Optionally compute average precision metrics
        # NOTE this can be memory intensive if dealing with large number of classes
        if config.predictions_batch is None and bool(config.save_probabilities):
            # load in probabilities df
            probs_save_file = os.path.join(save_directory, f"{col}_probabilities.csv")
        
            # Load probabilities from CSV
            probs_df = pd.read_csv(probs_save_file, index_col=0)
            
            # Convert to NumPy array
            probs = probs_df.values
        
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
        
        # Save metrics for this column
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
