# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC

import yaml
import json
import random
from typing import Union
import string
import os
from pathlib import Path
from analysis.experiment.utils.constants import MODELS
from analysis.experiment.utils.train_logger import logger

# set config_dir and experiment_dir relative to current file path
CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")
EXPERIMENT_DIR = str(Path(__file__).resolve().parent.parent.parent / "experiment")

# Set unrestrictive umask before any file operations
os.umask(0o000)

class Config:
    """
    A configuration class for Multiclass Classification, reads parameters from a YAML file and dynamically loads them as class attributes.
    """

    def __init__(self, config_path: Union[str, bytes, Path]):
        self.config_path = config_path

        # data paths to raw data csvs before tokenization (can be overridden by config file)
        self.training_data = None
        self.validation_data = None
        self.testing_data = None
        self.stratify = None  # column name, if any, to stratify train/val/test splits on (e.g. genus)
        self.embeddings_checkpoint_h5 = None  # it will look in the save_dir if this isn't set (this was added for MLP training)
        # name of dir to store test set results, default to test_results for back compatability
        self.testing_name = "test_results"
        # default, this gets overriden if original train run, else needs to be specified explicitly as a full dir path to save test metrics dir
        self.test_metrics_dir = None
        # if testing on new dataset different from original train run
        self.new_test_run = False

        # create paths to tokenized dataset save_paths, if these are not defined in config, use default values
        self.tokenized_training_data = None
        self.tokenized_validation_data = None
        self.tokenized_testing_data = None
        
        # data processor
        self.data_processor_path = None  # create path to data_processor
        self.data_processor_filename = "data_processor.pkl"  # default filename for data_processor

        # column names
        self.sequence_column = "sequence"  # dna sequence column name
        self.labels = (
            None  # by default do hierarchical classification on self.taxonomic_ranks
        )
        self.taxonomic_ranks = [
            "superkingdom",
            "phylum",
            "genus",
        ]

        # model parameters (defaults to NT-50m model)
        self.model_type = "NT"
        self.base_model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
        self.tokenizer_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
        self.tokenizer_kwargs = None  # if any tokenizer_kwargs different from default

        # curr learning
        self.model_trained_dir = None # load in previously trained model for curr learning
        
        # training parameters
        self.train_iterable = False  # bool to check if need to use IterableDataset for tokenizing training data as argument for HF Trainer
        self.num_rows_iterable = None
        self.use_class_weights = False # bool whether to use class_weights in data_processor for computing loss, valid for imbalanced classification
        self.train_batch_size = 16  # batch size per GPU for training
        self.eval_batch_size = 16  # batch size per GPU for eval
        self.epochs = 3  # Max number of train epochs (default 3)
        self.learning_rate = 0.00002  # initial learning rate for AdamW optimizer
        self.fp16 = False  # Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training.
        self.bf16 = False # True seems to be preferred for newer GPUS if we can do it
        self.weight_decay = 0 # 0.01 can be default
        self.warmup_ratio = 0 # 0.05 can be default
        # https://huggingface.co/docs/transformers/v4.52.2/en/main_classes/optimizer_schedules#transformers.SchedulerType
        self.lr_scheduler_type = "cosine" # calls get_cosine_schedule_with_warmup
        self.gradient_accumulation_steps = 1  # HF trainer arguments default
        
        # model validation parameters
        self.eval_accumulation_steps = 4  # to offload predictions from GPU to CPU, avoids GPU OOM
        self.prediction_loss_only = False # to only output/save validation loss instead of all individual probabilities during train loop

        # model testing parameters
        self.predictions_batch = None # use to batch test dataset and save off results in chunks to concat for summary metrics, defaults to None
        self.save_probabilities = True # use to save probabilities for each class in test set 
        # careful: this can be very memory intensive, especially if large number of classes, defaults to True

        
        self.multi_gpu_count = 1
        self.metric_for_best_model = (
            "eval_loss"  # default loss for determining best moel
        )
        self.greater_is_better = (
            False  # set to False if your metric is better when lower, e.g. eval loss
        )
        self.early_stopping_patience = (
            3  # Stop training after num epochs of no improvement
        )

        # alternative model training method parameters, e.g. PEFT
        self.peft_method = None  # should be None, "lora" or "ia3"
        self.lora_r = 1 # lora r parameter
        self.randomization = False # randomization of model weights
        self.freeze_layers_fraction = 0.0 # fraction of layers to freeze

        # save parameters
        self.script_args_file = "config_arguments.txt" # file name for storing config arguments saved in save_dir
        self.epochs_trained_file = "epochs_trained.txt" # file name for updating number of epochs trained during training for pre-emptabile training
        
        self.experiment_name = (
            None  # Initialize as None so generates random set of characters
        )
        
        # Load configuration from the YAML file and overwrite default values if necessary
        config_json = self.read_yaml()
        if config_json:
            self._load_class_vars(config_json)

        # initalize labels attribute
        if self.labels:
            if isinstance(self.labels, str):
                self.labels = [self.labels]
        else:
            # doing taxonomic classification, backward compatibility
            self.labels = self.taxonomic_ranks

        # Initialize attributes for directories
        self.experiment_dir = EXPERIMENT_DIR
        self.run_dir = os.path.join(self.experiment_dir, "runs")
        os.makedirs(self.run_dir, exist_ok=True)

        # Generate experiment_name if None
        self.experiment_name = self.experiment_name or "".join(
            random.choices(string.ascii_uppercase + string.digits, k=24)
        )

        # Create model dir
        self.results_dir = os.path.join(self.run_dir, self.experiment_name)
        os.makedirs(self.results_dir, exist_ok=True)

        # clean model save name
        self.model_save_name = self.base_model_name[
            self.base_model_name.index("/") + 1 :
        ]

        # Create save_dir
        self.save_dir = os.path.join(self.results_dir, self.model_save_name)
        os.makedirs(self.save_dir, exist_ok=True)

        # Create epochs trained logger
        self.epochs_trained_path = os.path.join(self.save_dir, self.epochs_trained_file)

        # Create training history dir
        self.train_history_dir = os.path.join(self.save_dir, "train_history")
        os.makedirs(self.train_history_dir, exist_ok=True)

        # model save dir
        self.model_save_dir = os.path.join(self.save_dir, "models")
        os.makedirs(self.model_save_dir, exist_ok=True)

        # Create data processor save path if not already set using save_dir as default
        if self.data_processor_path is None:
            self.data_processor_path = self.save_dir

        if self.embeddings_checkpoint_h5 is None:
            self.embeddings_checkpoint_h5 = os.path.join(
                self.save_dir, "checkpoints.h5"
            )

        # create tokenized dataset save_paths if not already set using save_dir as default
        if self.tokenized_training_data is None:
            self.tokenized_training_data = os.path.join(
                self.save_dir, "tokenized_training_data"
            )
        if self.tokenized_validation_data is None:
            self.tokenized_validation_data = os.path.join(
                self.save_dir, "tokenized_validation_data"
            )

        # new test run besides original train run
        if bool(self.new_test_run):
            # must specify test_metrics_dir in this case
            if self.test_metrics_dir:
                os.makedirs(self.test_metrics_dir, exist_ok=True)
            else:
                raise ValueError(
                    "Must specify test_metrics_dir if running prev trained model on new test set"
                )

            if self.tokenized_testing_data is None:
                # tokenized test dataset gets saved to subdirectory specified by test_metrics_dir to keep track of different test runs with same trained model
                self.tokenized_testing_data = os.path.join(
                    self.test_metrics_dir, "tokenized_testing_data"
                )
        else:
            # Create testing metrics folder as default within self.save_dir
            self.test_metrics_dir = os.path.join(self.save_dir, self.testing_name)
            os.makedirs(self.test_metrics_dir, exist_ok=True)

            if self.tokenized_testing_data is None:
                # tokenized test dataset gets saved to subdirectory specified by test_metrics_dir to keep track of different test runs with same trained model
                self.tokenized_testing_data = os.path.join(
                    self.test_metrics_dir, "tokenized_testing_data"
                )

        # Validate base_model_name
        self._validate_base_model_name()


        # Save config arguments to a file in the new run dir
        self.save_args_to_file()

    def read_yaml(self):
        """
        Reads the YAML configuration file.
        """
        if not os.path.exists(self.config_path):
            error = f"Config file not found at path: {self.config_path}"
            logger.error(error)
            raise FileNotFoundError(error)

        try:
            with open(self.config_path, "r") as f:  # Use standard open with os path
                config = yaml.safe_load(f)
                if not isinstance(config, dict):
                    logger.error("The YAML file must contain a dictionary at the root.")
                    raise ValueError(
                        "The YAML file must contain a dictionary at the root."
                    )
                return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise RuntimeError(f"Error parsing YAML file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error reading YAML file: {e}")
            raise RuntimeError(f"Unexpected error reading YAML file: {e}")

    def _load_class_vars(self, config_json):
        """
        Dynamically loads variables from the YAML config into the class,
        overriding default values where applicable.
        """
        for k, v in config_json.items():
            # Derive internal property names from provided names
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                # Handle new configuration keys not defined in defaults
                setattr(self, k, v)

    def _validate_base_model_name(self):
        """
        Validates if the base_model_name exists in models_dict for the given model_type.
        """
        if self.model_type not in MODELS:
            logger.error(f"Model type '{self.model_type}' not found in models_dict.")
            raise ValueError(
                f"Model type '{self.model_type}' not found in models_dict."
            )

        if self.base_model_name not in MODELS[self.model_type]:
            error = f"Base model name '{self.base_model_name}' is not valid for model type '{self.model_type}'. Available models {list(MODELS[self.model_type])}"
            logger.error(error)
            raise ValueError(error)


    def save_args_to_file(self):
        """
        Saves the config arguments to a file in the save dir.
        """
        # Define the output file path
        if self.new_test_run:
            args_file = os.path.join(self.test_metrics_dir, self.script_args_file)
        else:
            args_file = os.path.join(self.save_dir, self.script_args_file)

        # Extract all instance attributes of `self`
        self_attributes = vars(
            self
        )  # Returns a dictionary of all instance attributes and their values
        # Save the dictionary as JSON
        with open(args_file, "w") as f:
            json.dump(self_attributes, f, indent=2)
