# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC


import time
import os
import pickle
import re

import pandas as pd
from datasets import Dataset

from sklearn.preprocessing import LabelEncoder
from analysis.experiment.utils.train_logger import logger


class HierarchicalDataProcessor:
    def __init__(
        self,
        tokenizer,
        tokenizer_kwargs=None,
        sequence_column="sequence",
        taxonomic_ranks=[
            "superkingdom",
            "phylum",
            "genus",
        ],
        save_file="data_processor.pkl",
    ):
        self.tokenizer = tokenizer
        # Default tokenizer arguments
        default_tokenizer_kwargs = {
            "return_tensors": "pt",  # return results as pytorch tensor
            "padding": True,  # pads sequences to longest in batch
            "truncation": True,  # truncates if sequence exceeds max_length
        }
        # no max length is set, defaulting to maximum supported by the model tokenizer.model_max_length

        # Merge default and user-provided kwargs
        self.tokenizer_kwargs = {**default_tokenizer_kwargs, **(tokenizer_kwargs or {})}

        self.sequence_column = sequence_column
        self.taxonomic_ranks = taxonomic_ranks
        self.save_file = save_file

        self.num_labels = []
        self.encoders = {}
        self.cols_keep = []
        self.cols_keep.append(self.sequence_column)
        self.label_columns = [f"label_level_{i}" for i in taxonomic_ranks]

    def fit_encoder(self, file_name, save_directory):
        """Fits and initializes the label encoders for each taxonomic rank."""
        # Load the dataset from file_name
        start_time = time.time()
        logger.info(f"Loading file: {file_name}")
        if re.search(".tsv", file_name):
            df = pd.read_csv(file_name, sep="\t")
        else:
            df = pd.read_csv(file_name)
        load_time = time.time() - start_time
        logger.info("Data Loaded in %.2f seconds.", load_time)

        df_labels = df.copy()
        logger.info(f"Number of sequences: {df_labels.shape[0]}")
        logger.info(f"Dataset columns: {list(df_labels.columns)}")

        # Initialize label encoders for each rank
        logger.info(
            f"Starting to fit label encoders for taxonomic ranks {self.taxonomic_ranks}..."
        )
        for idx, rank in enumerate(self.taxonomic_ranks):
            if rank not in df_labels.columns:
                logger.info(f"Skipping rank '{rank}': not found in dataset columns.")
                continue

            labels = df_labels[rank]
            unique_labels = labels.nunique()
            logger.info(
                f"Processing rank '{rank}': {unique_labels} unique labels found."
            )

            # Store number of labels
            self.num_labels.append(unique_labels)

            # Initialize and fit the LabelEncoder
            encoder = LabelEncoder()
            labels_encoded = encoder.fit_transform(labels)
            self.encoders[rank] = encoder

            # Add encoded labels to the dataset
            col = f"label_level_{rank}"
            df_labels[col] = labels_encoded
            self.cols_keep.append(col)
            logger.info(
                f"Rank '{rank}' successfully encoded and added as column '{col}'."
            )

        logger.info(f"Encoders initialized for ranks: {list(self.encoders.keys())}")
        logger.info(f"Number of labels per rank: {self.num_labels}")

        # Create the labels dataset for further processing
        logger.info("Creating label for Hugging Face dataset...")
        hf_dataset_labels = self.create_labels_cols(df_labels)

        # Save the processor state
        self.save_processor(save_directory)
        return hf_dataset_labels

    def apply_encoder(self, file_name):
        """Applies the pre-fit label encoders to a new dataset. Requires encoders to already be populated."""
        if not self.encoders:
            raise ValueError(
                "Label encoders are not initialized. Please ensure that encoders are fitted before applying them."
            )

        # Load the dataset
        start_time = time.time()
        logger.info(f"Loading file: {file_name}")
        if re.search(".tsv", file_name):
            df = pd.read_csv(file_name, sep="\t")
        else:
            df = pd.read_csv(file_name)

        load_time = time.time() - start_time
        logger.info("Data Loaded in %.2f seconds.", load_time)
        df_labels = df.copy()
        logger.info(f"Number of sequences: {df_labels.shape[0]}")
        logger.info(f"Dataset columns: {list(df_labels.columns)}")

        # Check if taxonomic ranks are present in the dataset
        missing_ranks = [
            rank for rank in self.taxonomic_ranks if rank not in df_labels.columns
        ]
        if missing_ranks:
            logger.info(f"Missing taxonomic ranks in the dataset: {missing_ranks}")

        if all([rank not in df_labels.columns for rank in self.taxonomic_ranks]):
            logger.info(
                "No taxonomic ranks found in the dataset. Creating inference columns..."
            )
            hf_dataset_labels = self.create_inference_cols(df_labels)
        else:
            logger.info("Applying pre-trained label encoders to the dataset...")
            for idx, rank in enumerate(self.taxonomic_ranks):
                if rank not in df_labels.columns:
                    logger.info(
                        f"Skipping rank '{rank}': not found in dataset columns."
                    )
                    continue

                labels = df_labels[rank]
                logger.info(f"Encoding rank '{rank}' with {len(labels)} entries.")

                # Encode the labels
                if rank not in self.encoders:
                    raise ValueError(
                        "Label encoders are not initialized. Please ensure that encoders are fitted before applying them."
                    )
                labels_encoded = self.encoders[rank].transform(labels)
                col = f"label_level_{rank}"
                df_labels[col] = labels_encoded
                logger.info(f"Encoded rank '{rank}' added as column '{col}'.")

            logger.info("All applicable ranks encoded successfully.")
            hf_dataset_labels = self.create_labels_cols(df_labels)

        return hf_dataset_labels

    def create_labels_cols(self, df_labels):
        # Retain only necessary columns
        df_labels = df_labels[self.cols_keep]

        # Convert the DataFrame to a Hugging Face Dataset
        logger.info("Converting DataFrame to Hugging Face Dataset...")
        hf_dataset = Dataset.from_pandas(df_labels)

        # Select required columns
        hf_dataset = hf_dataset.select_columns(self.cols_keep)

        # Apply label creation logic
        hf_dataset_labels = hf_dataset.map(self.create_labels)

        # Tokenize the dataset
        start_time = time.time()
        logger.info("Tokenizing the dataset...")
        tokenized_dataset = hf_dataset_labels.map(
            self.tokenize_function,
            batched=True,
            remove_columns=[self.sequence_column],
        )
        tokenize_time = time.time() - start_time
        logger.info("Data Tokenized in %.2f seconds.", tokenize_time)

        return tokenized_dataset

    def create_inference_cols(self, df_labels):
        # Retain only the sequence column
        df_labels = df_labels[[self.sequence_column]]

        # Convert the DataFrame to a Hugging Face Dataset
        logger.info("Converting DataFrame to Hugging Face Dataset...")
        hf_dataset = Dataset.from_pandas(df_labels)

        # Select the sequence column
        hf_dataset = hf_dataset.select_columns([self.sequence_column])

        # Tokenize the dataset
        start_time = time.time()
        logger.info("Tokenizing the dataset...")
        hf_dataset_labels = hf_dataset
        tokenized_dataset = hf_dataset_labels.map(
            self.tokenize_function,
            batched=True,
            remove_columns=[self.sequence_column],
        )
        tokenize_time = time.time() - start_time
        logger.info("Data Tokenized in %.2f seconds.", tokenize_time)

        return tokenized_dataset

    def create_labels(self, example):
        example["labels"] = [example[label] for label in self.label_columns]
        return example

    def save_processor(self, save_directory):
        """Saves the processor state to a specified directory."""
        # Ensure the save_directory exists
        logger.info(f"Ensuring directory '{save_directory}' exists...")
        os.makedirs(save_directory, exist_ok=True)
        logger.info(f"Directory '{save_directory}' is ready.")

        # Prepare encoders for saving
        encoders = {cl: self.encoders[cl].classes_ for cl in self.encoders}
        logger.info(f"Encoders prepared: {list(encoders.keys())}")

        # Save the processor state
        save_path = os.path.join(save_directory, self.save_file)
        logger.info(f"Saving processor state to '{save_path}'...")
        with open(save_path, "wb") as f:
            pickle.dump(
                [
                    encoders,
                    self.taxonomic_ranks,
                    self.num_labels,
                    self.cols_keep,
                    self.label_columns,
                ],
                f,
            )
        logger.info(f"Processor state saved successfully to '{save_path}'.")

    def load_processor(self, save_directory):
        """Loads the processor state from a specified directory."""
        logger.info(f"Checking if directory '{save_directory}' exists...")
        # Check if the save_directory exists
        if not os.path.exists(save_directory):
            raise FileNotFoundError(f"The directory '{save_directory}' does not exist.")
        logger.info(f"Directory '{save_directory}' found.")

        # Construct the file path
        file_path = os.path.join(save_directory, self.save_file)
        logger.info(f"Checking if file '{file_path}' exists...")
        # Check if the file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"The file '{self.save_file}' does not exist in the directory '{save_directory}'."
            )
        logger.info(f"File '{file_path}' found.")

        # Load the processor state
        logger.info(f"Loading processor state from '{file_path}'...")
        with open(file_path, "rb") as f:
            (
                encoders,
                self.taxonomic_ranks,
                self.num_labels,
                self.cols_keep,
                self.label_columns,
            ) = pickle.load(f)
        logger.info("Processor state loaded successfully.")

        # Restore encoders
        logger.info("Restoring label encoders...")
        self.encoders = {}
        for cl in encoders:
            enc = LabelEncoder()
            enc.classes_ = encoders[cl]
            self.encoders[cl] = enc
            logger.info(f"Restored encoder for rank '{cl}'.")
        logger.info("All label encoders restored successfully.")

    def tokenize_function(self, examples):
        # Use self.tokenizer_kwargs when calling the tokenizer
        return self.tokenizer(
            examples[self.sequence_column],
            **self.tokenizer_kwargs,  # Unpack the dictionary
        )
