# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC

import h5py
import numpy as np
from analysis.experiment.utils.train_logger import logger

BATCH_SIZE = int(1e6)


def read_group_in_batches(
    h5_file_path, group_name="train", batch_size=BATCH_SIZE, sample_size=None
):
    """Reads dataset in mini-batches and returns the full dataset.

    If sample_size is provided, only sample_size samples are read.
    """
    with h5py.File(h5_file_path, "r") as f:
        if group_name not in f:
            raise ValueError(f"Group {group_name} not found in {h5_file_path}")

        dataset = f[group_name]["embeddings"]
        labels = f[group_name]["labels"]
        total_samples = dataset.shape[0]

        # Limit total samples if sample_size is provided
        if sample_size is not None:
            total_samples = min(total_samples, sample_size)
            logger.info(f"Limiting to {total_samples} samples for testing purposes.")

        logger.info(
            f"\nProcessing {total_samples} samples from '{group_name}' in batches of {batch_size}...\n"
        )

        # To store all embeddings and labels
        all_batches = []
        all_labels = []

        # make sure batch size is not larger than total samples
        if batch_size > total_samples:
            batch_size = total_samples
            logger.warning(
                f"Batch size adjusted to {batch_size} as it exceeds total samples."
            )

        for i in range(0, total_samples, batch_size):
            batch = dataset[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]
            all_batches.append(batch)
            all_labels.append(batch_labels)

            logger.info(f"Batch {i // batch_size + 1}: Shape {batch.shape}")

        # Concatenating batches to return full dataset
        embeddings = np.concatenate(all_batches, axis=0)
        labels = np.concatenate(all_labels, axis=0)

    logger.info(f"Group: {group_name}")
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    return embeddings, labels


def read_group(h5_file_path, group_name="train", sample_size=None):
    """Reads group from HDF5 file.

    If sample_size is provided, only sample_size samples are read.
    """
    with h5py.File(h5_file_path, "r") as f:
        if group_name not in f:
            raise ValueError(f"Group {group_name} not found in {h5_file_path}")
        embeddings = f[group_name]["embeddings"][:]
        labels = f[group_name]["labels"][:]

        # Limit samples if sample_size is provided
        if sample_size is not None:
            embeddings = embeddings[:sample_size]
            labels = labels[:sample_size]
            logger.info(
                f"Limiting to {embeddings.shape[0]} samples for testing purposes."
            )

        logger.info(f"Group: {group_name}")
        logger.info(f"Embeddings shape: {embeddings.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        return embeddings, labels
