# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC

import time
import os
import gc
import h5py
import pandas as pd
import numpy as np
import torch
from argparse import ArgumentParser
from torch import nn
from tqdm import tqdm
from datasets import Dataset, IterableDataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from analysis.experiment.utils.config import Config, CONFIG_DIR
from analysis.experiment.utils.train_logger import add_file_handler, logger
from analysis.experiment.utils.data_processor import DataProcessor, DataTokenizer
from analysis.experiment.models.embedding_model import GenomicEmbeddingModel
from analysis.experiment.utils.train_utils import (
    load_dataset_if_available,
    dataset_split,
    save_dataset,
)


def resume_embedding_extraction(
    h5_path: str,
    group_name: str,
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    tokenizer,
):
    """
    Extract embeddings (supports DataParallel for multi-GPU) and write them incrementally
    to an HDF5 group. NO COMPRESSION for speed.

    If partially done, we skip the first 'n_processed' samples in the dataset
    and append to the existing HDF5 datasets. This allows preemptible/resumable jobs.
    """
    if dataset is None:
        logger.info(f"No dataset for group '{group_name}', skipping.")
        return

    collator = DataCollatorWithPadding(tokenizer)

    with h5py.File(h5_path, "a") as f:
        # Create or open the group
        if group_name not in f:
            grp = f.create_group(group_name)
            grp.attrs["n_processed"] = 0
        else:
            grp = f[group_name]
            if "n_processed" not in grp.attrs:
                grp.attrs["n_processed"] = 0

        n_already_processed = grp.attrs["n_processed"]
        total_samples = len(dataset)
        logger.info(
            f"Group '{group_name}': {n_already_processed} samples already processed. "
            f"Total samples = {total_samples}."
        )

        # If we've already processed everything, no work to do
        if n_already_processed >= total_samples:
            logger.info(
                f"All samples in '{group_name}' are already processed. Skipping."
            )
            return

        # Subset the dataset to skip the first n_already_processed samples
        subset_indices = list(range(n_already_processed, total_samples))
        subset_dataset = dataset.select(subset_indices)

        dataloader = torch.utils.data.DataLoader(
            subset_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
        )

        # ---------------------------------------------------------------------
        #  Attempt to OPEN existing datasets if they already exist, or CREATE
        #  them if they do not.
        # ---------------------------------------------------------------------
        dset_emb = grp.get("embeddings", None)
        dset_lbl = grp.get("labels", None)

        # We'll fill these shapes in after we see the *first batch* and know the shapes.
        # If the dataset is brand-new, we'll create them. If it already exists, we just open it.

        current_offset = n_already_processed  # next row to write in the HDF5 dataset

        logger.info(
            f"Starting embedding extraction for '{group_name}' from sample {current_offset} to {total_samples - 1}."
        )

        for batch_data in tqdm(dataloader, desc=f"Extracting {group_name}"):
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            labels = batch_data[
                "labels"
            ]  # shape [batch_size, num_labels] or just [batch_size,]

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # outputs["embedding"] => shape: (batch_size, embedding_dim)
                embeddings = outputs["embedding"].cpu().numpy()

            labels = labels.cpu().numpy()
            bs = embeddings.shape[0]

            # If labels are 1D, you might do labels = labels[:, None] if you want shape (batch_size, 1).
            # But if it's e.g. shape (batch_size, 3), that's also fine. Just be consistent.

            # The global indices for this batch in the entire dataset

            # ------------- If needed, create the datasets ---------------
            if dset_emb is None:
                embed_dim = embeddings.shape[1]
                dset_emb = grp.create_dataset(
                    "embeddings",
                    shape=(0, embed_dim),
                    maxshape=(None, embed_dim),
                    dtype=np.float32,
                    chunks=(min(512, bs), embed_dim),
                    compression=None,  # no compression
                )

            if dset_lbl is None:
                # For example, if your labels are shape=(batch_size, 3)
                # then we'll set that as the second dimension, otherwise you can adapt:
                label_dim = labels.shape[1] if len(labels.shape) > 1 else 1
                dset_lbl = grp.create_dataset(
                    "labels",
                    shape=(0, label_dim),
                    maxshape=(None, label_dim),
                    dtype=labels.dtype,
                    chunks=(min(512, bs), label_dim),
                    compression=None,
                )

            # Now we know embed_dim, label_dim, etc. Let's finalize them if it's the first batch.

            # ------------- Extend each dataset to accommodate this batch -------------
            new_size = current_offset + bs

            # embeddings are 2D => (batch_size, embedding_dim)
            dset_emb.resize((new_size, dset_emb.shape[1]))

            # labels could be 1D or 2D => handle accordingly
            if len(dset_lbl.shape) == 2:
                dset_lbl.resize((new_size, dset_lbl.shape[1]))
            else:
                dset_lbl.resize((new_size,))

            # ------------- Write the data -------------
            dset_emb[current_offset:new_size, :] = embeddings

            if (
                len(labels.shape) == 1
                and len(dset_lbl.shape) == 2
                and dset_lbl.shape[1] == 1
            ):
                # If your label dataset is 2D but you have 1D labels, you can reshape:
                dset_lbl[current_offset:new_size, 0] = labels
            else:
                dset_lbl[current_offset:new_size, ...] = labels

            # Update offset
            current_offset = new_size
            grp.attrs["n_processed"] = current_offset

        logger.info(
            f"Finished writing {group_name} embeddings. Total processed: {current_offset}."
        )


def skip_iterator(iterable, n_skip):
    """
    Skip the first `n_skip` items from an iterable.
    """
    for i, item in enumerate(iterable):
        if i >= n_skip:
            yield item


def resume_embedding_extraction_iterable(
    h5_path: str,
    group_name: str,
    model: nn.Module,
    dataset: IterableDataset,
    device: torch.device,
    batch_size: int,
    tokenizer,
):
    """
    Supports embedding extraction from a HuggingFace IterableDataset.
    Resumes from last written index in HDF5 without assuming dataset length or indexing.
    """
    if dataset is None:
        logger.info(f"No dataset for group '{group_name}', skipping.")
        return

    collator = DataCollatorWithPadding(tokenizer)

    with h5py.File(h5_path, "a") as f:
        # Setup group and tracking
        if group_name not in f:
            grp = f.create_group(group_name)
            grp.attrs["n_processed"] = 0
        else:
            grp = f[group_name]
            if "n_processed" not in grp.attrs:
                grp.attrs["n_processed"] = 0

        n_already_processed = grp.attrs["n_processed"]
        logger.info(
            f"Resuming from sample {n_already_processed} in group '{group_name}'."
        )

        # Prepare HDF5 datasets
        dset_emb = grp.get("embeddings", None)
        dset_lbl = grp.get("labels", None)
        current_offset = n_already_processed

        # Wrap dataset with skip logic
        dataloader = torch.utils.data.DataLoader(
            skip_iterator(dataset, n_already_processed),
            batch_size=batch_size,
            collate_fn=collator,
        )

        for batch_data in tqdm(dataloader, desc=f"Extracting {group_name}"):
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            labels = batch_data["labels"]

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs["embedding"].cpu().numpy()

            labels = labels.cpu().numpy()
            bs = embeddings.shape[0]

            # Create datasets if needed
            if dset_emb is None:
                embed_dim = embeddings.shape[1]
                dset_emb = grp.create_dataset(
                    "embeddings",
                    shape=(0, embed_dim),
                    maxshape=(None, embed_dim),
                    dtype=np.float32,
                    chunks=(min(512, bs), embed_dim),
                    compression=None,
                )

            if dset_lbl is None:
                label_dim = labels.shape[1] if len(labels.shape) > 1 else 1
                dset_lbl = grp.create_dataset(
                    "labels",
                    shape=(0, label_dim),
                    maxshape=(None, label_dim),
                    dtype=labels.dtype,
                    chunks=(min(512, bs), label_dim),
                    compression=None,
                )

            # Resize and write to HDF5
            new_size = current_offset + bs
            dset_emb.resize((new_size, dset_emb.shape[1]))
            dset_lbl.resize(
                (new_size, dset_lbl.shape[1] if len(labels.shape) > 1 else 1)
            )

            if (
                len(labels.shape) == 1
                and len(dset_lbl.shape) == 2
                and dset_lbl.shape[1] == 1
            ):
                dset_lbl[current_offset:new_size, 0] = labels
            else:
                dset_lbl[current_offset:new_size, ...] = labels

            dset_emb[current_offset:new_size, :] = embeddings

            current_offset = new_size
            grp.attrs["n_processed"] = current_offset

        logger.info(
            f"Finished writing {group_name} embeddings. Total processed: {current_offset}."
        )


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
    # Must have a data processor already fit ina valid directory specified in config
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
        logger.info("Using IterableDataset for training.")
        tokenized_training_data = data_tokenizer.tokenize_iterable_dataset_from_file(
            config.training_data
        )

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

    # 5) Build model
    model = GenomicEmbeddingModel(config.base_model_name)

    # 6) GPU detection
    n_gpus = torch.cuda.device_count()

    #   If you want to scale batch size by #GPUs:
    #   We'll multiply the user-supplied eval_batch_size by n_gpus.
    base_batch_size = config.eval_batch_size
    if n_gpus > 1:
        logger.info(
            f"Multiple GPUs detected: {n_gpus}. Will scale batch size from {base_batch_size} to "
            f"{base_batch_size * n_gpus}."
        )
        effective_batch_size = base_batch_size * n_gpus
        model = nn.DataParallel(model)
    else:
        effective_batch_size = base_batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 7) HDF5 file path for saving incremental embeddings (no compression)
    h5_checkpoint_path = os.path.join(config.save_dir, "checkpoints.h5")

    # 8) Extract embeddings with global indexing, scaled batch size
    logger.info(
        f"Starting embedding extraction. Effective batch size: {effective_batch_size}"
    )
    if not bool(config.train_iterable) and tokenized_training_data is not None:
        resume_embedding_extraction(
            h5_path=h5_checkpoint_path,
            group_name="train",
            model=model,
            dataset=tokenized_training_data,
            device=device,
            batch_size=effective_batch_size,
            tokenizer=tokenizer,
        )
        gc.collect()
        torch.cuda.empty_cache()
        print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
        print(torch.cuda.memory_reserved() / 1e9, "GB reserved")

    elif bool(config.train_iterable):
        resume_embedding_extraction_iterable(
            h5_path=h5_checkpoint_path,
            group_name="train",
            model=model,
            dataset=tokenized_training_data,
            device=device,
            batch_size=effective_batch_size,
            tokenizer=tokenizer,
        )
        gc.collect()
        torch.cuda.empty_cache()
        print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
        print(torch.cuda.memory_reserved() / 1e9, "GB reserved")

    if tokenized_validation_data is not None:
        resume_embedding_extraction(
            h5_path=h5_checkpoint_path,
            group_name="val",
            model=model,
            dataset=tokenized_validation_data,
            device=device,
            batch_size=effective_batch_size,
            tokenizer=tokenizer,
        )
        gc.collect()
        torch.cuda.empty_cache()
        print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
        print(torch.cuda.memory_reserved() / 1e9, "GB reserved")

    if tokenized_testing_data is not None:
        resume_embedding_extraction(
            h5_path=h5_checkpoint_path,
            group_name="test",
            model=model,
            dataset=tokenized_testing_data,
            device=device,
            batch_size=effective_batch_size,
            tokenizer=tokenizer,
        )
        gc.collect()
        torch.cuda.empty_cache()
        print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
        print(torch.cuda.memory_reserved() / 1e9, "GB reserved")

    logger.info(
        "All done! Embeddings saved incrementally to 'checkpoints.h5' (no compression)."
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the YAML configuration file.",
        default="sample/train_embeddings.yaml",
        nargs="?",
    )
    args = parser.parse_args()

    config_file = args.config_file
    config_path = os.path.join(CONFIG_DIR, config_file)
    logger.info(f"Using configuration file: {config_path}")
    config = Config(config_path)

    for key, val in vars(config).items():
        logger.info(f"{key}: {val}")

    main(config)
