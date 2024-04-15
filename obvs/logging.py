from __future__ import annotations

import logging

from tqdm import tqdm


# Define TqdmLoggingHandler
class TqdmLoggingHandler(logging.Handler):
    def __init__(self):
        super().__init__()

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg, end="\n")


def set_tqdm_logging(exclude_loggers=None):
    exclude_loggers = set(exclude_loggers or [])
    tqdm_handler = TqdmLoggingHandler()

    # Get all existing loggers (including the root) and replace their handlers.
    loggers = [logging.root] + list(logging.root.manager.loggerDict.values())

    for a_logger in loggers:
        if (
            isinstance(a_logger, logging.Logger) and a_logger.name not in exclude_loggers
        ):  # Exclude specified loggers
            a_logger.handlers = [tqdm_handler]


# Now exclude your file logger by name when calling set_tqdm_logging
set_tqdm_logging(exclude_loggers={"patchscope"})

logger = logging.getLogger("patchscope")
logging.basicConfig(handlers=[TqdmLoggingHandler()], level=logging.INFO)

# Ensure the 'patchscope' logger level is set to allow debug messages through
logger.setLevel(logging.DEBUG)

# File Handler for my_logger only
file_handler = logging.FileHandler("experiments.log")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
logger.propagate = False  # Prevents the logger from passing messages to the root logger
