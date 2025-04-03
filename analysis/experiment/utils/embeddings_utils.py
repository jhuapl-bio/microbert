# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC


import h5py
import numpy as np
from analysis.experiment.utils.train_logger import logger

BATCH_SIZE = 512


def read_group_in_batches(h5_file_path, group_name="train", batch_size=BATCH_SIZE):
    """Reads dataset in mini-batches and returns the full dataset in the end"""
    with h5py.File(h5_file_path, "r") as f:
        if group_name not in f:
            raise ValueError(f"Group {group_name} not found in {h5_file_path}")

        dataset = f[group_name]["embeddings"]
        labels = f[group_name]["labels"]
        total_samples = dataset.shape[0]

        logger.info(
            f"\nProcessing {total_samples} samples from '{group_name}' in batches of {batch_size}...\n"
        )

        # To store all embeddings and labels
        all_batches = []
        all_labels = []

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


def read_group(h5_file_path, group_name="train"):
    """Reads group from HDF5 file"""
    with h5py.File(h5_file_path, "r") as f:
        if group_name not in f:
            raise ValueError(f"Group {group_name} not found in {h5_file_path}")
        embeddings = f[group_name]["embeddings"][:]
        labels = f[group_name]["labels"][:]
        logger.info(f"Group: {group_name}")
        logger.info(f"Embeddings shape: {embeddings.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        return embeddings, labels
