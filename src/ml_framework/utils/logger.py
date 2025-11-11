"""
Logging utilities for the ML framework.

Provides consistent logging configuration across all modules.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logger with consistent formatting.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to save logs
        format_string: Custom format string for log messages

    Returns:
        Configured logger instance

    Examples:
        >>> logger = setup_logger(__name__, level="DEBUG")
        >>> logger.info("Training started")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger by name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """
    Enhanced logger for tracking experiment progress with metrics.

    Examples:
        >>> exp_logger = ExperimentLogger("experiment_1")
        >>> exp_logger.log_metric("accuracy", 0.85)
        >>> exp_logger.log_parameter("learning_rate", 0.01)
    """

    def __init__(self, experiment_name: str, log_dir: str = "artifacts/logs", level: str = "INFO"):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to save log files
            level: Logging level
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"

        self.logger = setup_logger(
            name=f"experiment.{experiment_name}", level=level, log_file=str(log_file)
        )

        # Storage for metrics and parameters
        self.metrics = {}
        self.parameters = {}

        self.logger.info(f"Experiment '{experiment_name}' initialized")

    def log_parameter(self, name: str, value: any) -> None:
        """
        Log experiment parameter.

        Args:
            name: Parameter name
            value: Parameter value
        """
        self.parameters[name] = value
        self.logger.info(f"Parameter - {name}: {value}")

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """
        Log experiment metric.

        Args:
            name: Metric name
            value: Metric value
            step: Optional step/iteration number
        """
        if name not in self.metrics:
            self.metrics[name] = []

        metric_entry = {"value": value}
        if step is not None:
            metric_entry["step"] = step

        self.metrics[name].append(metric_entry)

        step_str = f" (step {step})" if step is not None else ""
        self.logger.info(f"Metric - {name}: {value:.4f}{step_str}")

    def log_metrics(self, metrics_dict: dict, step: Optional[int] = None) -> None:
        """
        Log multiple metrics at once.

        Args:
            metrics_dict: Dictionary of metric names and values
            step: Optional step/iteration number
        """
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step)

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def get_summary(self) -> dict:
        """
        Get summary of logged parameters and metrics.

        Returns:
            Dictionary containing parameters and final metric values
        """
        summary = {
            "experiment_name": self.experiment_name,
            "parameters": self.parameters,
            "metrics": {},
        }

        # Get final values for each metric
        for metric_name, values in self.metrics.items():
            if values:
                summary["metrics"][metric_name] = values[-1]["value"]

        return summary
