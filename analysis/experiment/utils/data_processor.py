# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC


import time
import os
import pickle
import re
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
from analysis.experiment.utils.train_logger import logger

def get_class_weights(df: pd.DataFrame, col: str) -> list[float]:
    labels = df[col].values
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return weights.tolist()

class DataProcessor:
    def __init__(
        self,
        sequence_column="sequence",
        labels=None,
        save_file="data_processor.pkl",
    ):
        """
        Initializes the DataProcessor for handling sequence data and labels.

        Args:
            sequence_column (str): The name of the column containing sequences.
            labels (List[str]): A list of label column names to encode.
            save_file (str): File path to save the processor's state.
        """
        self.sequence_column = sequence_column
        self.labels = labels
        self.save_file = save_file

        self.num_labels = []
        self.encoders = {}
        self.class_weights = []
        self.label_columns = [f"label_level_{label}" for label in self.labels]
        self.cols_keep = [self.sequence_column] + self.label_columns

    def fit_encoder_from_df(
        self, df: pd.DataFrame, save_directory: str = None
    ) -> pd.DataFrame:
        """
        Fits label encoders to a DataFrame and adds encoded label columns.

        Args:
            df (pd.DataFrame): Input DataFrame.
            save_directory (str, optional): Directory to save the processor state.

        Returns:
            pd.DataFrame: DataFrame with encoded label columns.
        """
        start_time = time.time()
        logger.info(f"Fitting encoders on dataframe with {df.shape[0]} rows.")
        df_labels = df.copy()

        for label in self.labels:
            if label not in df_labels.columns:
                logger.warning(f"Label '{label}' not found in DataFrame. Skipping.")
                continue

            encoder = LabelEncoder()
            labels_encoded = encoder.fit_transform(df_labels[label])
            self.encoders[label] = encoder
            df_labels[f"label_level_{label}"] = labels_encoded

            unique_labels = len(encoder.classes_)
            self.num_labels.append(unique_labels)
            logger.info(f"Fitted label '{label}' with {unique_labels} unique classes.")

            weights = get_class_weights(df_labels, label)
            self.class_weights.append(weights)
            logger.info(f"Computed class weights for label '{label}'...")
            
        if save_directory:
            self.save_processor(save_directory)

        logger.info(f"Fitting completed in {time.time() - start_time:.2f} seconds.")
        return df_labels[self.cols_keep]

    def fit_encoder_from_file(
        self, file_name: str, save_directory: str
    ) -> pd.DataFrame:
        """
        Loads a CSV/TSV file, fits label encoders, and saves processor.

        Args:
            file_name (str): Path to the CSV/TSV file.
            save_directory (str): Directory to save processor state.

        Returns:
            pd.DataFrame: Encoded DataFrame with selected columns.
        """
        start_time = time.time()
        logger.info(f"Loading file: {file_name}")

        if file_name.endswith(".tsv") or re.search(r"\.tsv$", file_name):
            df = pd.read_csv(file_name, sep="\t")
        else:
            df = pd.read_csv(file_name)

        logger.info(f"Data loaded in {time.time() - start_time:.2f} seconds.")
        return self.fit_encoder_from_df(df, save_directory=save_directory)

    def apply_encoder_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies pre-fitted encoders to a DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with encoded label columns.
        """
        if not self.encoders:
            raise ValueError(
                "Encoders not fitted. Please run fit_encoder_from_df or load processor."
            )

        start_time = time.time()
        logger.info(f"Applying encoders to dataframe with {df.shape[0]} rows.")

        df_labels = df.copy()

        for label in self.labels:
            if label not in df_labels.columns:
                logger.warning(f"Label '{label}' not found in DataFrame. Skipping.")
                continue

            encoder = self.encoders.get(label)
            if encoder is None:
                raise ValueError(f"Encoder for label '{label}' not found in processor.")

            df_labels[f"label_level_{label}"] = encoder.transform(df_labels[label])
            logger.info(f"Encoded label '{label}' applied successfully.")
        
       
        out = df_labels[self.cols_keep]
        for col in self.label_columns:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="raise").astype(int)
        logger.info(f"Encoding completed in {time.time() - start_time:.2f} seconds.")
        return out

    def apply_encoder_to_file(self, file_name: str) -> pd.DataFrame:
        """
        Applies pre-fitted label encoders to a dataset file.

        Args:
            file_name (str): Path to a CSV or TSV file containing labels.

        Returns:
            pd.DataFrame: DataFrame with encoded label columns.
        """
        if not self.encoders:
            raise ValueError(
                "Encoders not fitted. Please fit or load encoders before applying them."
            )

        start_time = time.time()
        logger.info(f"Loading file: {file_name}")

        if file_name.endswith(".tsv") or re.search(r"\.tsv$", file_name):
            df = pd.read_csv(file_name, sep="\t")
        else:
            df = pd.read_csv(file_name)

        logger.info(f"Data loaded in {time.time() - start_time:.2f} seconds.")
        return self.apply_encoder_to_df(df)

    def save_processor(self, save_directory):
        """
        Saves the processor state to a specified directory.
        
        Args:
            save_directory (str): Path to the directory where the processor state will be saved.
            weights (optional): Class weights (e.g., torch.Tensor, np.ndarray, or list).
        """
        # Ensure the save_directory exists
        logger.info(f"Ensuring directory '{save_directory}' exists...")
        os.makedirs(save_directory, exist_ok=True)
        logger.info(f"Directory '{save_directory}' is ready.")
    
        # Prepare encoders for saving
        encoders = {cl: self.encoders[cl].classes_ for cl in self.encoders}
        logger.info(f"Encoders prepared: {list(encoders.keys())}")
        
        # Save state
        save_path = os.path.join(save_directory, self.save_file)
        logger.info(f"Saving processor state to '{save_path}'...")
    
        processor_state = {
            "encoders": encoders,
            "weights": self.class_weights,
            "labels": self.labels,
            "num_labels": self.num_labels,
            "cols_keep": self.cols_keep,
            "label_columns": self.label_columns,
        }
    
        with open(save_path, "wb") as f:
            pickle.dump(processor_state, f)
    
        logger.info(f"Processor state saved successfully to '{save_path}'.")

    
    def load_processor(self, save_directory):
        """Loads the processor state from a specified directory."""
        logger.info(f"Checking if directory '{save_directory}' exists...")
        if not os.path.exists(save_directory):
            raise FileNotFoundError(f"The directory '{save_directory}' does not exist.")
        logger.info(f"Directory '{save_directory}' found.")
    
        file_path = os.path.join(save_directory, self.save_file)
        logger.info(f"Checking if file '{file_path}' exists...")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"The file '{self.save_file}' does not exist in the directory '{save_directory}'."
            )
        logger.info(f"File '{file_path}' found.")
    
        logger.info(f"Loading processor state from '{file_path}'...")
        with open(file_path, "rb") as f:
            state = pickle.load(f)
        logger.info("Processor state loaded successfully.")
    
        # Restore components
        logger.info("Restoring processor attributes...")
        self.labels = state["labels"]
        self.num_labels = state["num_labels"]
        self.cols_keep = state["cols_keep"]
        self.label_columns = state["label_columns"]
        self.class_weights = state.get("weights", None)  # Default to None if missing
        logger.info(f"Label columns: {self.label_columns}")
        logger.info(f"Number of labels: {self.num_labels}")
    
        # Restore encoders
        logger.info("Restoring label encoders...")
        self.encoders = {}
        for cl, class_list in state["encoders"].items():
            enc = LabelEncoder()
            enc.classes_ = class_list
            self.encoders[cl] = enc
            logger.info(f"Restored encoder for label '{cl}'.")
    
        logger.info("All processor attributes restored successfully.")

# TODO refactor to make possible with inference mode only (no label columns provided)
class DataTokenizer:
    def __init__(
        self,
        tokenizer,
        sequence_column,
        label_columns,
        tokenizer_kwargs=None,
    ):
        """
        Args:
            tokenizer (PreTrainedTokenizer): A Hugging Face tokenizer instance used to tokenize input sequences.
            sequence_column (str): Column containing input sequences.
            label_columns (List[str]): Columns containing labels.
            tokenizer_kwargs (Optional[Dict[str, Any]]): Optional tokenizer settings to override defaults.
        """
        self.tokenizer = tokenizer
        self.sequence_column = sequence_column
        self.label_columns = label_columns
        self.cols_keep = [self.sequence_column] + self.label_columns

        # Default tokenizer arguments
        default_tokenizer_kwargs = {
            "return_tensors": "pt",  # Return results as PyTorch tensors
            "padding": True,  # Pad sequences to the longest in the batch
            "truncation": True,  # Truncate sequences longer than max length
        }

        # Merge default and user-provided kwargs
        self.tokenizer_kwargs = {**default_tokenizer_kwargs, **(tokenizer_kwargs or {})}

    def tokenize_function(self, examples):
        """Tokenizes input sequences."""
        return self.tokenizer(
            examples[self.sequence_column],
            **self.tokenizer_kwargs,
        )
    
    def safe_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value.clone().detach()
        elif isinstance(value, list) and len(value) == 1 and isinstance(value[0], torch.Tensor):
            return value[0].clone().detach()
        return torch.tensor(value, dtype=torch.long)

    def create_labels(self, example):
        """Creates a combined label list."""
        example["labels"] = torch.stack([
            self.safe_tensor(example[label]) for label in self.label_columns
        ])
        return example

    def create_labels_subset(self, example, subset_columns):
        """Creates a combined label list from a subset."""
        example["labels"] = torch.stack([
            self.safe_tensor(example[label]) for label in subset_columns
        ])
        return example
        
    def tokenize_iterable_dataset_from_file(self, file_name, subset_columns):
        """Loads a CSV/TSV and tokenizes it in memory using streaming and batching."""
        start_time = time.time()
        logger.info(f"Loading file: {file_name}")

        # Load CSV as streaming IterableDataset
        iterable_dataset = load_dataset(
            "csv", data_files=file_name, split="train", streaming=True
        )

        logger.info(
            f"Iterable Dataset loaded in {time.time() - start_time:.2f} seconds."
        )
        
        iterable_dataset = iterable_dataset.map(self.create_labels_subset, fn_kwargs={"subset_columns": subset_columns})

        # Apply tokenization
        iterable_dataset = iterable_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.cols_keep,
        )

        return iterable_dataset

    def tokenize_dataset_from_df(self, df_labels, batch_size=100000):
        """Tokenizes a pandas DataFrame in memory in batches."""
        logger.info(f"Columns in input DataFrame: {list(df_labels.columns)}")
        num_rows = len(df_labels)
        logger.info(f"Total rows in input DataFrame: {num_rows}")
        num_batches = int(np.ceil(num_rows / batch_size))
        logger.info(f"Processing in {num_batches} batches of size {batch_size}")

        tokenized_batches = []

        for i in range(0, num_rows, batch_size):
            df_batch = df_labels.iloc[i : i + batch_size]
            logger.info(
                f"Processing batch {i // batch_size + 1}/{num_batches} with {len(df_batch)} rows"
            )

            hf_dataset = Dataset.from_pandas(df_batch)
            hf_dataset = hf_dataset.select_columns(self.cols_keep)
            hf_dataset = hf_dataset.map(self.create_labels)

            start_time = time.time()
            tokenized = hf_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=self.cols_keep,
            )
            logger.info(f"Batch tokenized in {time.time() - start_time:.2f} seconds.")
            tokenized_batches.append(tokenized)

        logger.info("Combining all tokenized batches into final dataset...")
        final_dataset = concatenate_datasets(tokenized_batches)
        return final_dataset

    def tokenize_dataset_from_file(self, file_name, batch_size=100000):
        """Loads a CSV/TSV and tokenizes in memory in batches."""
        start_time = time.time()
        logger.info(f"Loading file: {file_name}")

        if file_name.endswith(".tsv") or re.search(r"\.tsv$", file_name):
            df = pd.read_csv(file_name, sep="\t")
        else:
            df = pd.read_csv(file_name)

        logger.info(f"Data loaded in {time.time() - start_time:.2f} seconds.")
        return self.tokenize_dataset_from_df(df, batch_size=batch_size)
