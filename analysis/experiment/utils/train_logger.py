# © 2024-2025 The Johns Hopkins University Applied Physics Laboratory LLC


import datetime
import logging
from transformers import logging as hf_logging

# Define global log_formatter
log_formatter = logging.Formatter("[%(asctime)s] %(levelname)s : %(message)s")

# Custom logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s : %(message)s",
)
logger = logging.getLogger(__name__)


def add_file_handler(logger_object, outdir, filename="train"):
    log_handle = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = outdir + f"/{filename}_{log_handle}.log"

    # Delete the log file if it already exists
    # if os.path.exists(log_path):
    #     os.remove(log_path)

    # Create a file handler for logging
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(log_formatter)

    # Add the file handler to both the custom logger and the root logger
    logging.getLogger().addHandler(file_handler)  # Root logger

    # Configure Hugging Face logging
    hf_logging.set_verbosity_info()  # Set Hugging Face logging level
    hf_logging.enable_propagation()  # Enable propagation to the root logger
    hf_logging.disable_default_handler()  # Disable Hugging Face's default handler

    return log_path


def get_log_path(outdir, filename="train"):
    log_handle = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = outdir + f"/{filename}_{log_handle}.log"
    return log_path
