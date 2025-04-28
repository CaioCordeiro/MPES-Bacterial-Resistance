"""
Centralized logging configuration for the application.
Provides a consistent logging interface with configurable log levels.
"""

import logging
import os
import sys
from typing import Optional

# Define log levels with their corresponding integer values
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Default format for log messages
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Global logger instance
_logger = None


def setup_logging(
    log_level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger with the specified log level and output destinations.

    Args:
        log_level: String representation of the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to a log file. If provided, logs will be written to this file
                  in addition to the console.

    Returns:
        Configured logger instance
    """
    global _logger

    if _logger is not None:
        return _logger

    # Convert string log level to logging module constant
    numeric_level = LOG_LEVELS.get(log_level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger("genetic_analysis")
    logger.setLevel(numeric_level)
    logger.propagate = False

    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)

    # Create formatter
    formatter = logging.Formatter(DEFAULT_FORMAT)
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        # Create directory for log file if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """
    Get the configured logger instance. If the logger hasn't been set up yet,
    it will be initialized with default settings.

    Returns:
        Logger instance
    """
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger
