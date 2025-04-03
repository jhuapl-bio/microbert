# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC


import os
import time
from argparse import ArgumentParser
import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
)

from accelerate import Accelerator

from analysis.experiment.utils.config import Config, CONFIG_DIR
from analysis.experiment.utils.randomize_model import randomize_model
from analysis.experiment.process.hierarchical_processor import HierarchicalDataProcessor
from analysis.experiment.models.hierarchical_model import (
    HierarchicalClassificationModel,
)
from analysis.experiment.process.classification_processor import (
    ClassificationDataProcessor,
)
from analysis.experiment.models.classification_model import ClassificationModel
from analysis.experiment.utils.train_utils import (
    load_dataset_if_available,
    save_dataset,
    dataset_split,
    load_model_weights,
    extract_and_plot_training_metrics,
    evaluate_and_save_model_metrics,
    compute_metrics_hierarchical,
    compute_metrics_classification,
    SafetensorsTrainer,
)
from analysis.experiment.utils.train_peft import prepare_peft_model
from analysis.experiment.utils.train_logger import add_file_handler, logger


# Custom callback to log timing after each epoch
class EpochTimingCallback(TrainerCallback):
    def __init__(self):
        self.epoch_times = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        end_time = time.time()
        epoch_time = end_time - self.start_time
        self.epoch_times.append(epoch_time)
        logger.info(f"Epoch {state.epoch:.0f} completed in {epoch_time:.2f} seconds.")


class LogMetricsCallback(TrainerCallback):
    """
    A custom callback to log metrics during the training loop.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:  # Check if there are logs to process
            for key, value in logs.items():
                logger.info(f"{key}: {value}")


# Add the callback to your Trainer initialization
timing_callback = EpochTimingCallback()


def main(config: Config):
    # Define the device
    accelerator = Accelerator()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.multi_gpu_count > 1
        else accelerator.device
    )

    # Determine run modes based on available datasets and their paths
    training = bool(
        config.training_data and os.path.exists(config.training_data)
    )  # Check if training data exists
    testing = bool(
        config.testing_data and os.path.exists(config.testing_data)
    )  # Check if testing data exists
    validation = bool(
        config.validation_data and os.path.exists(config.validation_data)
    )  # Check if validation data exists

    # If no training is specified, ensure testing is enabled for evaluation, else raise Exception
    if not training and not testing:
        raise ValueError(
            "No training or testing mode is enabled. Provide valid training data for training or testing data for evaluation."
        )

    if training:
        filename = "train"
    else:
        filename = "test"
        
    if bool(config.new_test_run):
        add_file_handler(logger, config.test_metrics_dir, filename)
        logger.info(
        f"Logger {filename} initialized. Logs will be saved to {config.new_test_run}"
    )
    else:
        add_file_handler(logger, config.save_dir, filename)
        logger.info(
        f"Logger {filename} initialized. Logs will be saved to {config.save_dir}"
    )

    logger.info(f"Training: {training}, Validation: {validation}, Testing: {testing}")

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

    # initializing data processor
    start_time = time.time()
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
    data_processor_load_time = time.time() - start_time
    logger.info("Data Processor initialized in %.2f seconds.", data_processor_load_time)

    # Initialize datasets as None; will populate based on mode
    train_dataset, val_dataset, test_dataset = None, None, None

    # Load datasets if they exist
    train_dataset = load_dataset_if_available(
        config.tokenized_training_data, "training"
    )
    val_dataset = load_dataset_if_available(
        config.tokenized_validation_data, "validation"
    )
    test_dataset = load_dataset_if_available(config.tokenized_testing_data, "testing")

    # Bools to determine whether datasets need to be created
    create_training_data = training and train_dataset is None
    create_validation_data = validation and val_dataset is None
    create_testing_data = testing and test_dataset is None
    
    # Remove label columns from datasets once processed
    label_cols_remove = data_processor.label_columns

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

    # initialize model
    if config.classification:
        model = ClassificationModel(config.base_model_name, data_processor.num_labels)
    else:
        model = HierarchicalClassificationModel(
            config.base_model_name,
            data_processor.taxonomic_ranks,
            data_processor.num_labels,
        )

    if training and config.randomization:
        model = randomize_model(model)

    total_parameters = sum([f.numel() for f in model.parameters()])
    if training and config.freeze_layers_fraction:
        count = 0
        for param in model.named_parameters():
            print(f"Freezing layer: {param[0]}, {param[1].shape}")
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
    if config.peft_method:
        # TODO save after every checkpoint for peft model training
        # num_epochs_trained will always eval to -1 if doing peft
        model, num_epochs_trained = load_model_weights(model, config.model_save_dir)
    else:
        model, num_epochs_trained = load_model_weights(model, config.model_save_dir)
    if num_epochs_trained == -1:
        # placeholder for completed training run based on config parameters
        num_epochs_trained = int(config.epochs)

    logger.info(
        f"Loading model checkpoint from previous run training {num_epochs_trained} epochs"
    )
    # move model to GPU
    model.to(device)

    # Determine evaluation strategy based on whether training is enabled
    if training:
        # we always save checkpoint after every epoch
        eval_strategy = "epoch"  # Evaluate after each epoch during training
        epochs_config = int(config.epochs)
        # number of epochs to train based on current checkpoint if it exists
        epochs = epochs_config - num_epochs_trained
        logger.info(f"Training {epochs} number of epochs based on existing checkpoint")

    else:
        epochs = 0  # No training, so set epochs to 0
        eval_strategy = "no"  # No evaluation during non-training mode

    learning_rate = float(config.learning_rate)
    train_batch_size = int(config.train_batch_size)
    eval_batch_size = int(config.eval_batch_size)
    weight_decay = float(config.weight_decay)
    warmup_ratio = float(config.warmup_ratio)
    fp16 = bool(config.fp16)
    bf16 = bool(config.bf16)
    metric_for_best_model = config.metric_for_best_model
    greater_is_better = bool(config.greater_is_better)
    early_stopping_patience = int(config.early_stopping_patience)
    eval_accumulation_steps = int(config.eval_accumulation_steps)

    if config.classification:
        compute_metrics_function = compute_metrics_classification
    else:
        compute_metrics_function = compute_metrics_hierarchical

    training_args = TrainingArguments(
        output_dir=config.model_save_dir,
        eval_strategy=eval_strategy,
        save_strategy=eval_strategy,
        logging_strategy=eval_strategy,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        eval_accumulation_steps=eval_accumulation_steps,
        bf16=bf16,
        fp16=fp16,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
    )

    # Define early stopping callback with patience
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience
    )

    if config.model_type == "HYENA":
        trainer = SafetensorsTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_function,
            callbacks=[
                early_stopping_callback,
                timing_callback,
                LogMetricsCallback(),
            ],  # Early stopping with patience
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_function,
            callbacks=[
                early_stopping_callback,
                timing_callback,
                LogMetricsCallback(),
            ],  # Early stopping with patience
        )

    if training and epochs >= 1:
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
    all_metrics = evaluate_and_save_model_metrics(
        trainer, test_dataset, data_processor, config
    )

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
        default="bertax/sample/train.yaml",
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
