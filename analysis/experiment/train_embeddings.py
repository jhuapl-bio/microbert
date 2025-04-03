# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC

import os
import gc
import h5py
import numpy as np
import torch
from argparse import ArgumentParser
from torch import nn
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from analysis.experiment.utils.config import Config, CONFIG_DIR
from analysis.experiment.utils.train_logger import add_file_handler, logger
from analysis.experiment.process.hierarchical_processor import HierarchicalDataProcessor
from analysis.experiment.process.classification_processor import (
    ClassificationDataProcessor,
)
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


def main(config: Config):
    # 1) Setup logger / config
    training = bool(config.training_data and os.path.exists(config.training_data))
    testing = bool(config.testing_data and os.path.exists(config.testing_data))
    validation = bool(config.validation_data and os.path.exists(config.validation_data))

    if not training and not testing:
        raise ValueError("No training or testing mode is enabled.")

    log_filename = "train" if training else "test"
    add_file_handler(logger, config.save_dir, log_filename)
    logger.info(f"Logger '{log_filename}' initialized. Logs -> {config.save_dir}")

    # 2) Load tokenizer
    if "GenomeOcean" in config.base_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name, trust_remote_code=True, padding_side="left"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name, trust_remote_code=True
        )

    # 3) Setup data processor
    if config.classification:
        data_processor = ClassificationDataProcessor(
            tokenizer=tokenizer,
            tokenizer_kwargs=config.tokenizer_kwargs,
            sequence_column=config.sequence_column,
            label_column=config.label_column,
            save_file=config.data_processor_filename,
        )
    else:
        data_processor = HierarchicalDataProcessor(
            tokenizer=tokenizer,
            tokenizer_kwargs=config.tokenizer_kwargs,
            sequence_column=config.sequence_column,
            taxonomic_ranks=config.taxonomic_ranks,
            save_file=config.data_processor_filename,
        )

    # 4) Load or create tokenized datasets
    train_dataset = load_dataset_if_available(
        config.tokenized_training_data, "training"
    )
    val_dataset = load_dataset_if_available(
        config.tokenized_validation_data, "validation"
    )
    test_dataset = load_dataset_if_available(config.tokenized_testing_data, "testing")

    create_training_data = training and train_dataset is None
    create_validation_data = validation and val_dataset is None
    create_testing_data = testing and test_dataset is None
    
    # create training data if required
    if create_training_data:
        # Encode the full training dataset and store labels for splitting later
        hf_dataset_labels = data_processor.fit_encoder(
            config.training_data, config.data_processor_path
        )
        # Handle cases for validation and testing datasets
        if create_validation_data and create_testing_data:
            # Use predefined validation and testing datasets
            val_dataset = data_processor.apply_encoder(config.validation_data)
            test_dataset = data_processor.apply_encoder(config.testing_data)
        elif not create_validation_data and create_testing_data:
            # Split training data into train/val (90% train, 10% val) and use predefined testing dataset
            train_val_split = dataset_split(dataset=hf_dataset_labels, test_size=0.1, config=config)
            train_dataset, val_dataset = (
                train_val_split["train"],
                train_val_split["test"],
            )
            test_dataset = data_processor.apply_encoder(config.testing_data)
        elif create_validation_data and not create_testing_data:
            # Use predefined validation dataset and split training data into train/test (90% train, 10% test)
            val_dataset = data_processor.apply_encoder(config.validation_data)
            train_test_split = dataset_split(dataset=hf_dataset_labels, test_size=0.1, config=config)
            train_dataset, test_dataset = (
                train_test_split["train"],
                train_test_split["test"],
            )
        else:
            # Perform an 80/10/10 split for train/val/test when no predefined validation or testing datasets are provided
            train_testval_split = dataset_split(dataset=hf_dataset_labels, test_size=0.2, config=config) # Split 80% train, 20% test+val
            train_dataset, testval_dataset = (
                train_testval_split["train"],
                train_testval_split["test"],
            )
            testval_split = dataset_split(dataset=testval_dataset, test_size=0.5, config=config)  # Split remaining 20% into 10% val, 10% test
            val_dataset, test_dataset = testval_split["train"], testval_split["test"]


        # Remove label columns from datasets
        label_cols_remove = data_processor.label_columns
        train_dataset = train_dataset.remove_columns(label_cols_remove)
        val_dataset = val_dataset.remove_columns(label_cols_remove)
        test_dataset = test_dataset.remove_columns(label_cols_remove)

        # Save off datasets and log sizes of created datasets for debugging and confirmation
        save_dataset(train_dataset, config.tokenized_training_data, "training")
        save_dataset(val_dataset, config.tokenized_validation_data, "validation")
        save_dataset(test_dataset, config.tokenized_testing_data, "testing")

    elif create_testing_data:
        # Load existing data processor and apply it to the test dataset for evaluation
        data_processor.load_processor(config.data_processor_path)
        test_dataset = data_processor.apply_encoder(config.testing_data)
        test_dataset = test_dataset.remove_columns(label_cols_remove)
        save_dataset(test_dataset, config.tokenized_testing_data, "testing")

    else:
        data_processor.load_processor(config.data_processor_path)

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
    if train_dataset is not None:
        resume_embedding_extraction(
            h5_path=h5_checkpoint_path,
            group_name="train",
            model=model,
            dataset=train_dataset,
            device=device,
            batch_size=effective_batch_size,
            tokenizer=tokenizer,
        )
        gc.collect()
        torch.cuda.empty_cache()
        print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
        print(torch.cuda.memory_reserved() / 1e9, "GB reserved")

    if val_dataset is not None:
        resume_embedding_extraction(
            h5_path=h5_checkpoint_path,
            group_name="val",
            model=model,
            dataset=val_dataset,
            device=device,
            batch_size=effective_batch_size,
            tokenizer=tokenizer,
        )
        gc.collect()
        torch.cuda.empty_cache()
        print(torch.cuda.memory_allocated() / 1e9, "GB allocated")
        print(torch.cuda.memory_reserved() / 1e9, "GB reserved")

    if test_dataset is not None:
        resume_embedding_extraction(
            h5_path=h5_checkpoint_path,
            group_name="test",
            model=model,
            dataset=test_dataset,
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
        default="bertax/sample/train_embeddings.yaml",
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
