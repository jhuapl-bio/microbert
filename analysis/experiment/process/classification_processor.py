# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC


import os
import pickle
import re

import pandas as pd
from datasets import Dataset

from sklearn.preprocessing import LabelEncoder


class ClassificationDataProcessor:
    def __init__(
        self,
        tokenizer,
        tokenizer_kwargs=None,
        sequence_column="sequence",
        label_column="superkingdom",
        save_file="data_processor.pkl",
    ):
        self.tokenizer = tokenizer
        # Default tokenizer arguments
        default_tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": None,
        }

        # Merge default and user-provided kwargs
        self.tokenizer_kwargs = {**default_tokenizer_kwargs, **(tokenizer_kwargs or {})}

        self.sequence_column = sequence_column
        self.label_column = label_column
        self.save_file = save_file

        self.num_labels = None
        self.encoders = {}
        self.cols_keep = []
        self.cols_keep.append(self.sequence_column)

    def fit_encoder(self, file_name, save_directory):
        """Fits and initializes the label encoder for label column"""
        # Load the dataset from file_name
        print(f"Loading file: {file_name}")
        if re.search(".tsv", file_name):
            df = pd.read_csv(file_name, sep="\t")
        else:
            df = pd.read_csv(file_name)

        df_labels = df.copy()
        print(f"Number of sequences: {df_labels.shape[0]}")
        print(f"Dataset columns: {list(df_labels.columns)}")

        # Initialize label encoders for each column
        print(f"Starting to fit label encoders for label column {self.label_column}...")
        if self.label_column not in df_labels.columns:
            raise ValueError(
                f"Column '{self.label_column}': not found in dataset columns."
            )

        labels = df_labels[self.label_column]
        unique_labels = labels.nunique()
        print(
            f"Processing column '{self.label_column}': {unique_labels} unique labels found."
        )

        # Store number of labels
        self.num_labels = unique_labels

        # Initialize and fit the LabelEncoder
        encoder = LabelEncoder()
        labels_encoded = encoder.fit_transform(labels)
        self.encoders[self.label_column] = encoder

        # Add encoded labels to the dataset
        col = f"label_{self.label_column}"
        df_labels[col] = labels_encoded
        self.cols_keep.append(col)
        print(
            f"Label '{self.label_column}' successfully encoded and added as column '{col}'."
        )
        print(f"Number of labels: {self.num_labels}")

        # Create the labels dataset for further processing
        print("Creating label for Hugging Face dataset...")
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
        print(f"Loading file: {file_name}")
        if re.search(".tsv", file_name):
            df = pd.read_csv(file_name, sep="\t")
        else:
            df = pd.read_csv(file_name)

        df_labels = df.copy()
        print(f"Number of sequences: {df_labels.shape[0]}")
        print(f"Dataset columns: {list(df_labels.columns)}")

        # Check if label_column present in the dataset
        if self.label_column not in df_labels.columns:
            print(
                f"Label col {self.label_column} not found in the dataset. Creating inference columns..."
            )
            hf_dataset_labels = self.create_inference_cols(df_labels)
        else:
            print("Applying pre-trained label encoder to the dataset...")

            labels = df_labels[self.label_column]
            print(f"Encoding column '{self.label_column}' with {len(labels)} entries.")

            # Encode the labels
            if self.label_column not in self.encoders:
                raise ValueError(
                    "Label encoder not initialized. Please ensure that encoders are fitted before applying them."
                )
            labels_encoded = self.encoders[self.label_column].transform(labels)
            col = f"label_{self.label_column}"
            df_labels[col] = labels_encoded
            print(f"Encoded column '{self.label_column}' added as column '{col}'.")
            hf_dataset_labels = self.create_labels_cols(df_labels)

        return hf_dataset_labels

    def create_labels_cols(self, df_labels):
        # Retain only necessary columns
        df_labels = df_labels[self.cols_keep]

        # Convert the DataFrame to a Hugging Face Dataset
        print("Converting DataFrame to Hugging Face Dataset...")
        hf_dataset = Dataset.from_pandas(df_labels)

        # Select required columns
        hf_dataset = hf_dataset.select_columns(self.cols_keep)

        # Apply label creation logic
        hf_dataset_labels = hf_dataset.map(self.create_labels)

        # Tokenize the dataset
        print("Tokenizing the dataset...")
        tokenized_dataset = hf_dataset_labels.map(
            self.tokenize_function,
            batched=True,
            remove_columns=[self.sequence_column],
        )

        return tokenized_dataset

    def create_inference_cols(self, df_labels):
        # Retain only the sequence column
        df_labels = df_labels[[self.sequence_column]]

        # Convert the DataFrame to a Hugging Face Dataset
        print("Converting DataFrame to Hugging Face Dataset...")
        hf_dataset = Dataset.from_pandas(df_labels)

        # Select the sequence column
        hf_dataset = hf_dataset.select_columns([self.sequence_column])

        # Tokenize the dataset
        print("Tokenizing the dataset...")
        hf_dataset_labels = hf_dataset
        tokenized_dataset = hf_dataset_labels.map(
            self.tokenize_function,
            batched=True,
            remove_columns=[self.sequence_column],
        )

        return tokenized_dataset

    def create_labels(self, example):
        example["labels"] = example[f"label_{self.label_column}"]
        return example

    def save_processor(self, save_directory):
        """Saves the processor state to a specified directory."""
        # Ensure the save_directory exists
        print(f"Ensuring directory '{save_directory}' exists...")
        os.makedirs(save_directory, exist_ok=True)
        print(f"Directory '{save_directory}' is ready.")

        # Prepare encoders for saving
        encoders = {cl: self.encoders[cl].classes_ for cl in self.encoders}
        print(f"Encoders prepared: {list(encoders.keys())}")

        # Save the processor state
        save_path = os.path.join(save_directory, self.save_file)
        print(f"Saving processor state to '{save_path}'...")
        with open(save_path, "wb") as f:
            pickle.dump(
                [
                    encoders,
                    self.label_column,
                    self.num_labels,
                    self.cols_keep,
                ],
                f,
            )
        print(f"Processor state saved successfully to '{save_path}'.")

    def load_processor(self, save_directory):
        """Loads the processor state from a specified directory."""
        print(f"Checking if directory '{save_directory}' exists...")
        # Check if the save_directory exists
        if not os.path.exists(save_directory):
            raise FileNotFoundError(f"The directory '{save_directory}' does not exist.")
        print(f"Directory '{save_directory}' found.")

        # Construct the file path
        file_path = os.path.join(save_directory, self.save_file)
        print(f"Checking if file '{file_path}' exists...")
        # Check if the file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                f"The file '{self.save_file}' does not exist in the directory '{save_directory}'."
            )
        print(f"File '{file_path}' found.")

        # Load the processor state
        print(f"Loading processor state from '{file_path}'...")
        with open(file_path, "rb") as f:
            (
                encoders,
                self.label_column,
                self.num_labels,
                self.cols_keep,
            ) = pickle.load(f)
        print("Processor state loaded successfully.")

        # Restore encoders
        print("Restoring label encoders...")
        self.encoders = {}
        for cl in encoders:
            enc = LabelEncoder()
            enc.classes_ = encoders[cl]
            self.encoders[cl] = enc
            print(f"Restored encoder for column '{cl}'.")
        print("All label encoders restored successfully.")

    def tokenize_function(self, examples):
        # Use self.tokenizer_kwargs when calling the tokenizer
        return self.tokenizer(
            examples[self.sequence_column],
            **self.tokenizer_kwargs,  # Unpack the dictionary
        )
