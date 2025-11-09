"""
Configuration management utilities.

This module provides functions to load and validate configuration files,
ensuring reproducibility across experiments.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration container with attribute-style access.
    
    Examples:
        >>> config = Config({"model": {"type": "xgboost"}})
        >>> config.model.type
        'xgboost'
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with optional default.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config back to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        return self._config
    
    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with loaded configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
        
    Examples:
        >>> config = load_config("configs/config.yaml")
        >>> print(config.model.type)
        logistic_regression
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_file, 'r') as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    # Validate required sections
    required_sections = ['data', 'model', 'training', 'artifacts']
    missing_sections = [s for s in required_sections if s not in config_dict]
    
    if missing_sections:
        raise ValueError(
            f"Configuration missing required sections: {missing_sections}"
        )
    
    logger.info("Configuration loaded successfully")
    return Config(config_dict)


def save_config(config: Config, save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object to save
        save_path: Path where to save configuration
    """
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_file, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)
    
    logger.info(f"Configuration saved to {save_path}")


def merge_configs(base_config: Config, override_config: Dict[str, Any]) -> Config:
    """
    Merge override configuration into base configuration.
    
    Useful for experiment variations without creating new config files.
    
    Args:
        base_config: Base configuration
        override_config: Dictionary of overrides
        
    Returns:
        New Config with merged values
        
    Examples:
        >>> base = load_config("configs/config.yaml")
        >>> override = {"model": {"hyperparameters": {"C": 0.5}}}
        >>> new_config = merge_configs(base, override)
    """
    def deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge dictionaries."""
        merged = base.copy()
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    merged_dict = deep_merge(base_config.to_dict(), override_config)
    return Config(merged_dict)
