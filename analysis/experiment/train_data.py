# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC

"""
This script is intended to be run on a CPU and is optimized specifically for data tokenization tasks.
It does not require GPU acceleration and should be used in preprocessing workflows where only tokenization is performed.
"""


import os
import time
import pandas as pd
from argparse import ArgumentParser
from transformers import (
    AutoTokenizer,
)


from analysis.experiment.utils.config import Config, CONFIG_DIR
from analysis.experiment.utils.data_processor import DataProcessor, DataTokenizer
from analysis.experiment.utils.train_utils import (
    load_dataset_if_available,
    save_dataset,
    dataset_split,
)
from analysis.experiment.utils.train_logger import add_file_handler, logger


def main(config: Config):

    # Determine run modes based on available datasets and their paths
    training = bool(
        config.training_data and os.path.exists(config.training_data)
    )  # Check if training data exists
    validation = bool(
        config.validation_data and os.path.exists(config.validation_data)
    )  # Check if validation data exists
    testing = bool(
        config.testing_data and os.path.exists(config.testing_data)
    )  # Check if testing data exists

    # If no training is specified, ensure testing is enabled for evaluation, else raise Exception
    if not training and not testing:
        raise ValueError(
            "No training or testing mode is enabled. Provide valid training data for training or testing data for evaluation."
        )

    log_filename = "train" if training else "test"

    if bool(config.new_test_run):
        add_file_handler(logger, config.test_metrics_dir, log_filename)
        logger.info(
            f"Logger {log_filename} initialized. Logs will be saved to {config.new_test_run}"
        )
    else:
        add_file_handler(logger, config.save_dir, log_filename)
        logger.info(
            f"Logger {log_filename} initialized. Logs will be saved to {config.save_dir}"
        )

    logger.info(f"Training: {training}, Validation: {validation}, Testing: {testing}")

    # initializing data processor
    start_time = time.time()
    data_processor = DataProcessor(
        sequence_column=config.sequence_column,
        labels=config.labels,
        save_file=config.data_processor_filename,
    )
    data_processor_load_time = time.time() - start_time
    # Must have a data processor already fit in a valid directory specified in config
    data_processor.load_processor(config.data_processor_path)
    logger.info("Data Processor initialized in %.2f seconds.", data_processor_load_time)

    # Load the tokenizer
    start_time = time.time()
    if "GenomeOcean" in config.base_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name,
            trust_remote_code=True,
            padding_side="left",  # mistral
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name, trust_remote_code=True
        )

    tokenizer_load_time = time.time() - start_time
    logger.info("Tokenizer loaded in %.2f seconds.", tokenizer_load_time)

    # initializing data tokenizer
    start_time = time.time()
    data_tokenizer = DataTokenizer(
        tokenizer=tokenizer,
        tokenizer_kwargs=config.tokenizer_kwargs,
        sequence_column=config.sequence_column,
        label_columns=[f"label_level_{i}" for i in config.labels],
    )
    data_tokenizer_load_time = time.time() - start_time
    logger.info("Data Tokenizer initialized in %.2f seconds.", data_tokenizer_load_time)

    # Load tokenized datasets from disk if they exist
    tokenized_training_data = load_dataset_if_available(
        config.tokenized_training_data, "training"
    )
    tokenized_validation_data = load_dataset_if_available(
        config.tokenized_validation_data, "validation"
    )
    tokenized_testing_data = load_dataset_if_available(
        config.tokenized_testing_data, "testing"
    )

    # Define batch size once
    BATCH_SIZE = 100_000

    # Determine if we need to create tokenized datasets
    create_training_data = training and tokenized_training_data is None
    create_validation_data = validation and tokenized_validation_data is None
    create_testing_data = testing and tokenized_testing_data is None

    # === Iterable Training Dataset ===
    if create_training_data and bool(config.train_iterable):
        logger.info("Using IterableDataset for training...skipping train set tokenization")
        
        if create_validation_data:
            val_df = pd.read_csv(config.validation_data)
            tokenized_validation_data = data_tokenizer.tokenize_dataset_from_df(
                val_df, batch_size=BATCH_SIZE
            )
            save_dataset(
                tokenized_validation_data,
                config.tokenized_validation_data,
                "validation",
            )

        if create_testing_data:
            test_df = pd.read_csv(config.testing_data)
            tokenized_testing_data = data_tokenizer.tokenize_dataset_from_df(
                test_df, batch_size=BATCH_SIZE
            )
            save_dataset(
                tokenized_testing_data, config.tokenized_testing_data, "testing"
            )

    # === Non-Iterable Training Dataset ===
    elif create_training_data:
        logger.info("Creating training dataset from file (non-iterable).")
        train_df = pd.read_csv(config.training_data)

        if create_validation_data and create_testing_data:
            val_df = pd.read_csv(config.validation_data)
            test_df = pd.read_csv(config.testing_data)
        elif not create_validation_data and create_testing_data:
            train_df, val_df = dataset_split(df=train_df, test_size=0.1, config=config)
            test_df = pd.read_csv(config.testing_data)
        elif create_validation_data and not create_testing_data:
            val_df = pd.read_csv(config.validation_data)
            train_df, test_df = dataset_split(df=train_df, test_size=0.1, config=config)
        else:
            train_df, testval_df = dataset_split(
                df=train_df, test_size=0.2, config=config
            )
            val_df, test_df = dataset_split(df=testval_df, test_size=0.5, config=config)

        tokenized_training_data = data_tokenizer.tokenize_dataset_from_df(
            train_df, batch_size=BATCH_SIZE
        )
        save_dataset(
            tokenized_training_data, config.tokenized_training_data, "training"
        )

        if create_validation_data:
            tokenized_validation_data = data_tokenizer.tokenize_dataset_from_df(
                val_df, batch_size=BATCH_SIZE
            )
            save_dataset(
                tokenized_validation_data,
                config.tokenized_validation_data,
                "validation",
            )

        if create_testing_data:
            tokenized_testing_data = data_tokenizer.tokenize_dataset_from_df(
                test_df, batch_size=BATCH_SIZE
            )
            save_dataset(
                tokenized_testing_data, config.tokenized_testing_data, "testing"
            )

    # === Only Testing Dataset Required ===
    elif create_testing_data:
        logger.info("Creating only testing dataset.")
        test_df = pd.read_csv(config.testing_data)
        tokenized_testing_data = data_tokenizer.tokenize_dataset_from_df(
            test_df, batch_size=BATCH_SIZE
        )
        save_dataset(tokenized_testing_data, config.tokenized_testing_data, "testing")

    if create_validation_data:
        logger.info("Creating only validation dataset.")
        val_df = pd.read_csv(config.validation_data)
        tokenized_validation_data = data_tokenizer.tokenize_dataset_from_df(
            val_df, batch_size=BATCH_SIZE
        )
        save_dataset(tokenized_validation_data, config.tokenized_validation_data, "validation",)


    logger.info("All done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    # Add a positional argument for the config path
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML configuration file.",
        default="sample/train.yaml",
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

    # train training
    main(config)
