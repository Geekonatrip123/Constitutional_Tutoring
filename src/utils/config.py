"""
Configuration management system.
Loads and validates YAML configuration files.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager that loads and provides access to all config files.
    
    Usage:
        config = Config()
        api_key = config.api_keys.gemini.api_key
        student_classifier_config = config.student_classifier
    """
    
    def __init__(self, config_dir: str = "./configs"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Path to directory containing YAML config files
        """
        self.config_dir = Path(config_dir)
        self._configs = {}
        
        # Load all configurations
        self._load_all_configs()
        
        # Validate critical configurations
        self._validate_configs()
    
    def _load_all_configs(self):
        """Load all YAML files from config directory."""
        config_files = {
            'api_keys': 'api_keys.yaml',
            'skills_taxonomy': 'skills_taxonomy.yaml',
            'student_classifier': 'student_classifier_config.yaml',
            'alignment_scorer': 'alignment_scorer_config.yaml',
            'deliberation_dpo': 'deliberation_dpo_config.yaml',
            'planner': 'planner_config.yaml',
            'evaluation': 'evaluation_config.yaml',
            'sec_training': 'sec_training_config.yaml',
        }
        
        for key, filename in config_files.items():
            filepath = self.config_dir / filename
            try:
                with open(filepath, 'r') as f:
                    self._configs[key] = yaml.safe_load(f)
                logger.info(f"Loaded config: {filename}")
            except FileNotFoundError:
                if key == 'api_keys':
                    logger.error(
                        f"CRITICAL: {filename} not found! "
                        f"Copy api_keys.yaml.template and add your keys."
                    )
                    raise
                else:
                    logger.warning(f"Config file not found: {filename}")
                    self._configs[key] = {}
            except yaml.YAMLError as e:
                logger.error(f"Error parsing {filename}: {e}")
                raise
    
    def _validate_configs(self):
        """Validate that critical configuration values are set."""
        # Check API keys
        gemini_key = self.get('api_keys.gemini.api_key')
        if not gemini_key or gemini_key == "YOUR_GEMINI_API_KEY_HERE":
            logger.error(
                "Gemini API key not configured! "
                "Please set it in configs/api_keys.yaml"
            )
            raise ValueError("Gemini API key not configured")
        
        # Check critical paths exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            "./data/raw",
            "./data/processed",
            "./data/external",
            "./models/checkpoints",
            "./logs",
            "./results",
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value
                     (e.g., 'api_keys.gemini.api_key')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> config.get('api_keys.gemini.model')
            'gemini-1.5-pro'
        """
        keys = key_path.split('.')
        value = self._configs
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def __getattr__(self, name: str) -> Any:
        """
        Allow attribute-style access to configs.
        
        Example:
            >>> config.api_keys
            {'gemini': {...}, ...}
        """
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        
        if name in self._configs:
            return DotDict(self._configs[name])
        
        raise AttributeError(f"Config '{name}' not found")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return all configurations as a dictionary."""
        return self._configs.copy()
    
    def save_snapshot(self, output_path: str):
        """
        Save current configuration snapshot for reproducibility.
        
        Args:
            output_path: Path to save configuration snapshot
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self._configs, f, default_flow_style=False)
        
        logger.info(f"Configuration snapshot saved to {output_path}")


class DotDict:
    """
    Wrapper to allow dictionary access via dot notation.
    
    Example:
        >>> d = DotDict({'a': {'b': 1}})
        >>> d.a.b
        1
    """
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data
    
    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        
        value = self._data.get(name)
        
        if isinstance(value, dict):
            return DotDict(value)
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        return self._data[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self._data
    
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        return self._data.copy()


# Convenience function for quick access
def load_config(config_dir: str = "./configs") -> Config:
    """
    Load configuration.
    
    Args:
        config_dir: Path to configuration directory
        
    Returns:
        Config object
    """
    return Config(config_dir)


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"Gemini model: {config.get('api_keys.gemini.model')}")
    print(f"Number of skills: {len(config.get('skills_taxonomy.skills', []))}")