# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC

import os
import time
from argparse import ArgumentParser

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

from accelerate import Accelerator
from accelerate.data_loader import IterableDatasetShard

from analysis.experiment.utils.config import Config, CONFIG_DIR
from analysis.experiment.utils.randomize_model import randomize_model
from analysis.experiment.utils.data_processor import DataProcessor, DataTokenizer
from analysis.experiment.models.hierarchical_model import (
    HierarchicalClassificationModel,
)
from analysis.experiment.utils.train_utils import (
    SafetensorsTrainer,
    EpochTimingCallback,
    LogMetricsCallback,
    count_csv_rows_fast,
    calculate_steps_per_epoch,
    calculate_max_steps,
    load_dataset_if_available,
    save_dataset,
    dataset_split,
    load_model_weights,
    extract_and_plot_training_metrics,
    evaluate_model,
    save_model_metrics,
    compute_metrics_hierarchical,
)
from analysis.experiment.utils.train_peft import prepare_peft_model
from analysis.experiment.utils.train_logger import add_file_handler, logger

def main(config: Config):
    # Define the device
    accelerator = Accelerator()
    device = (
        accelerator.device
        if config.multi_gpu_count > 1
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

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

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
        subset_columns = [f'label_level_{i}' for i in config.labels]
        logger.info(f"Generating Iterable Dataset for {subset_columns}...")
        tokenized_training_data = data_tokenizer.tokenize_iterable_dataset_from_file(
            file_name=config.training_data, subset_columns=subset_columns,
        )

        if create_validation_data:
            logger.info("Creating tokenized validation data...")
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
            logger.info("Creating tokenized testing data...")
            ext = os.path.splitext(config.testing_data)[-1].lower()
            delimiter = '\t' if ext == '.tsv' else ','

            test_df = pd.read_csv(config.testing_data, delimiter=delimiter)

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
        ext = os.path.splitext(config.testing_data)[-1].lower()
        delimiter = '\t' if ext == '.tsv' else ','

        test_df = pd.read_csv(config.testing_data, delimiter=delimiter)
        testing_df_encoded = data_processor.apply_encoder_to_df(test_df)
        tokenized_testing_data = data_tokenizer.tokenize_dataset_from_df(
            testing_df_encoded, batch_size=BATCH_SIZE
        )
        save_dataset(tokenized_testing_data, config.tokenized_testing_data, "testing")

    class_weights = data_processor.class_weights if bool(config.use_class_weights) else None
    if class_weights:
        logger.info("Using computed class weights in classification loss function")

    # loop through data processor labels to get subset (or all) for model training labels
    data_processor_labels = dict(zip(data_processor.labels, data_processor.num_labels))
    num_labels = []
    for label in config.labels:
        if label not in data_processor_labels:
            raise ValueError(f"Label {label} not found in encoded data processor labels")
        num_labels.append(data_processor_labels[label])
    
    # Assume: data_processor.labels is total list of label names in order
    indices_to_keep = []
    for label in config.labels:
        if label not in data_processor.labels:
            raise ValueError(f"Label {label} not found in data_processor.labels")
        indices_to_keep.append(data_processor.labels.index(label))

    if len(config.labels) != len(data_processor.labels):
        logger.info(f"Tokenized data has {data_processor.labels} labels but config specifies {config.labels} labels...")
        def filter_labels(example, indices_to_keep):
            example["labels"] = [example["labels"][i] for i in indices_to_keep]
            return example
        
        if training and not bool(config.train_iterable):
            sample_example = tokenized_training_data[0]
            if len(sample_example["labels"]) != len(config.labels):
                logger.info("Filtering training data (non-iterable)...")
                tokenized_training_data = tokenized_training_data.map(
                    filter_labels,
                    fn_kwargs={"indices_to_keep": indices_to_keep}
                )    
        if validation:
            sample_example = tokenized_validation_data[0]
            if len(sample_example["labels"]) != len(config.labels):
                logger.info("Filtered validation data...")
                tokenized_validation_data = tokenized_validation_data.map(
                    filter_labels,
                    fn_kwargs={"indices_to_keep": indices_to_keep}
                )
            
        if testing:
            sample_example = tokenized_testing_data[0]
            if len(sample_example["labels"]) != len(config.labels):
                logger.info("Filtering testing data...")
                tokenized_testing_data = tokenized_testing_data.map(
                    filter_labels,
                    fn_kwargs={"indices_to_keep": indices_to_keep}
                )    
        
    # initialize model
    model = HierarchicalClassificationModel(
        config.base_model_name,
        num_labels,
        class_weights,
    )

    if training and config.randomization:
        model = randomize_model(model)

    total_parameters = sum([f.numel() for f in model.parameters()])
    if training and config.freeze_layers_fraction:
        count = 0
        for param in model.named_parameters():
            logger.info(f"Freezing layer: {param[0]}, {param[1].shape}")
            param[1].requires_grad = False
            count += param[1].numel()
            if config.freeze_layers_fraction <= (count / total_parameters):
                break

    # determine if using PEFT to train
    if training and config.peft_method:
        # update model
        model = prepare_peft_model(
            config.model_type, model, method=config.peft_method, lora_r=config.lora_r
        )
        model.print_trainable_parameters()

    # if only testing, then load best model or most recent checkpoint in experiment run
    model, num_checkpoints_saved = load_model_weights(model, config.model_save_dir)
    
    # all training completed, placeholder for completed training run based on config parameters    
    if num_checkpoints_saved == -1:
        num_epochs_trained = int(config.epochs)

    logger.info(
        f"Loading model checkpoint from previous run based on {num_checkpoints_saved} saved checkpoints"
    )

    
    # move model to GPU
    model.to(device)

    # Parse and cast config values in common
    output_dir = config.model_save_dir
    learning_rate = float(config.learning_rate)
    train_batch_size = int(config.train_batch_size)
    eval_batch_size = int(config.eval_batch_size)
    weight_decay = float(config.weight_decay)
    lr_scheduler_type = config.lr_scheduler_type
    warmup_ratio = float(config.warmup_ratio)
    fp16 = bool(config.fp16)
    bf16 = bool(config.bf16)
    metric_for_best_model = config.metric_for_best_model
    greater_is_better = bool(config.greater_is_better)
    early_stopping_patience = int(config.early_stopping_patience)
    gradient_accumulation_steps = int(config.gradient_accumulation_steps)
    eval_accumulation_steps = int(config.eval_accumulation_steps)
    prediction_loss_only = bool(config.prediction_loss_only)
    num_devices = int(config.multi_gpu_count)
    epochs_config = int(config.epochs)

    # Init training/eval control - default to no training
    epochs = 0
    eval_strategy = "no"

    # Define trainer callbacks
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience
    )
    epoch_timing_callback = EpochTimingCallback(config.epochs_trained_path)
    log_metrics_callback = LogMetricsCallback()
    custom_callbacks = [early_stopping_callback, epoch_timing_callback, log_metrics_callback]
    
    # Determine evaluation strategy based on whether training is enabled
    if training:
        if bool(config.train_iterable):
            # option to load in num_rows from config to save time
            if config.num_rows_iterable:
                train_data_size = int(config.num_rows_iterable)
            else:
                train_data_size = count_csv_rows_fast(config.training_data)

            logger.info(
                f"Creating IterableDataset from train data with {train_data_size} rows..."
            )
            
            # number of epochs to train based on saved file in callback
            num_epochs_trained = epoch_timing_callback.epochs_trained
            epochs = epochs_config - num_epochs_trained
            logger.info(
                f"Training {epochs} number of epochs based on previous runs"
            )

            eval_strategy = "steps"

            eval_steps = calculate_steps_per_epoch(
                dataset_size=train_data_size,
                per_device_batch_size=train_batch_size,
                num_devices=num_devices,
                accumulation_steps=gradient_accumulation_steps,
            )
            max_steps = calculate_max_steps(
                dataset_size=train_data_size,
                per_device_batch_size=train_batch_size,
                num_devices=num_devices,
                accumulation_steps=gradient_accumulation_steps,
                num_epochs=epochs,
            )
            logger.info(f"Creating IterableDataset using max steps {max_steps}...")

            if num_devices > 1:
                # Shard the dataset across multiple processes
                sharded_dataset = IterableDatasetShard(
                    dataset=tokenized_training_data,
                    batch_size=train_batch_size,
                    drop_last=False, # the last batch will be included, even if it's smaller. 
                    num_processes=accelerator.num_processes,
                    process_index=accelerator.process_index,
                    split_batches=False,
                )
                logger.info(
                    f"Creating sharded dataset: "
                    f"batch_size={train_batch_size}, "
                    f"num_processes={accelerator.num_processes}, "
                    f"process_index={accelerator.process_index}, "
                    f"drop_last=False, split_batches=False"
                )
    
                # Create the DataLoader
                train_dataloader = DataLoader(
                    sharded_dataset, batch_size=train_batch_size, collate_fn=data_collator
                )

        else:
            # save checkpoint after every epoch
            eval_strategy = "epoch"  # Evaluate after each epoch during training
            # number of epochs to train based on saved file in callback
            num_epochs_trained = epoch_timing_callback.epochs_trained
            epochs = epochs_config - num_epochs_trained
            logger.info(
                f"Training {epochs} number of epochs based on previous runs"
            )

    # Shared training configuration
    training_args_kwargs = {
        "output_dir": output_dir,
        "eval_strategy": eval_strategy,
        "save_strategy": eval_strategy,
        "logging_strategy": eval_strategy,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "num_train_epochs": epochs,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "eval_accumulation_steps": eval_accumulation_steps,
        "prediction_loss_only": prediction_loss_only,
        "bf16": bf16,
        "fp16": fp16,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
        "load_best_model_at_end": True,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": greater_is_better,
    }

    # additional training args if using IterableDataset
    if training and bool(config.train_iterable):
        training_args_kwargs["logging_steps"] = eval_steps
        training_args_kwargs["eval_steps"] = eval_steps
        training_args_kwargs["save_steps"] = eval_steps
        training_args_kwargs["max_steps"] = max_steps
        training_args_kwargs["save_total_limit"] = epochs
        training_args_kwargs["remove_unused_columns"] = False

    else:
        training_args_kwargs["num_train_epochs"] = epochs

    # Create TrainingArguments instance
    training_args = TrainingArguments(**training_args_kwargs)

    if bool(config.train_iterable) and num_devices > 1:
        # set train_dataset to None in trainer initialization, to then be overrided by .get_train_dataloader method
        if config.model_type == "HYENA":
            trainer = SafetensorsTrainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=None,
                eval_dataset=tokenized_validation_data,
                compute_metrics=compute_metrics_hierarchical,
                callbacks=custom_callbacks,
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=None,
                eval_dataset=tokenized_validation_data,
                compute_metrics=compute_metrics_hierarchical,
                callbacks=custom_callbacks,
            )

        # Override get_train_dataloader to return our custom DataLoader
        def get_train_dataloader(self):
            return train_dataloader

        # Bind the new method to the trainer instance
        trainer.get_train_dataloader = get_train_dataloader.__get__(trainer)
        logger.info("get_train_dataloader modified with custom DataLoader")

    else:
        if config.model_type == "HYENA":
            trainer = SafetensorsTrainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=tokenized_training_data,
                eval_dataset=tokenized_validation_data,
                compute_metrics=compute_metrics_hierarchical,
                callbacks=custom_callbacks,
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=tokenized_training_data,
                eval_dataset=tokenized_validation_data,
                compute_metrics=compute_metrics_hierarchical,
                callbacks=custom_callbacks,
            )

    if training and epochs >= 1:
        if num_devices > 1:
            logger.info(f"[Rank {accelerator.process_index}] Starting training...")
            trainer.train()
            logger.info(f"[Rank {accelerator.process_index}] Finished training.")
        else:
            logger.info("Training model")
            trainer.train()

        # Explicitly save the best model
        if config.peft_method:
            # merge and unload peft model adapter modules with base model
            trainer.model = trainer.model.merge_and_unload()
        trainer.save_model(config.model_save_dir)  # Save the best model to a new dir
        logger.info(f"The best model has been saved to: {config.save_dir}")
        extract_and_plot_training_metrics(trainer, config.train_history_dir)
        logger.info(f"Model training history saved to: {config.train_history_dir}")

    # get testing results
    logger.info("Evaluating model on test set")
    evaluate_model(trainer, tokenized_testing_data, data_processor, config)
    logger.info("Saving results of model evaluation")
    all_metrics = save_model_metrics(data_processor, config)
    
    # print results of metrics to log
    for col, metrics_dict in all_metrics.items():
        logger.info(f"Summary Metrics on test set for column {col}:")
        for metric, value in metrics_dict.items():
            logger.info(f"{metric}: {value}")

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
