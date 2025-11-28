"""
Logging utilities for experiment tracking and debugging.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json


def setup_logger(
    name: str = "deliberative_tutor",
    log_dir: str = "./logs",
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        console: Whether to log to console
        file: Whether to log to file
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


class ExperimentLogger:
    """
    Structured logging for experiments with JSON output.
    """
    
    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "./results",
        log_to_wandb: bool = False
    ):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save experiment logs
            log_to_wandb: Whether to log to Weights & Biases
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / f"{experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file
        self.log_file = self.experiment_dir / "experiment_log.jsonl"
        
        # Setup standard logger
        self.logger = setup_logger(
            name=f"experiment_{experiment_name}",
            log_dir=str(self.experiment_dir),
        )
        
        # Weights & Biases integration
        self.log_to_wandb = log_to_wandb
        if log_to_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.wandb_run = None
            except ImportError:
                self.logger.warning("wandb not installed, disabling W&B logging")
                self.log_to_wandb = False
        
        # Experiment metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'experiment_dir': str(self.experiment_dir),
        }
        
        self.logger.info(f"Experiment initialized: {experiment_name}")
        self.logger.info(f"Output directory: {self.experiment_dir}")
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log a structured event.
        
        Args:
            event_type: Type of event (e.g., 'metric', 'dialogue_turn', 'planner_output')
            data: Event data as dictionary
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': data,
        }
        
        # Write to JSONL file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
        
        # Log to W&B if enabled
        if self.log_to_wandb and self.wandb_run:
            self.wandb.log(data)
    
    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        """
        Log a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            step: Optional step/iteration number
        """
        data = {'metric': metric_name, 'value': value}
        if step is not None:
            data['step'] = step
        
        self.log_event('metric', data)
        self.logger.info(f"Metric - {metric_name}: {value}")
    
    def log_config(self, config: Dict[str, Any]):
        """
        Log experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_file = self.experiment_dir / "config.yaml"
        
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        self.log_event('config', config)
        self.logger.info(f"Configuration saved to {config_file}")
    
    def log_dialogue(
        self,
        dialogue_id: str,
        turn_number: int,
        student_message: str,
        tutor_response: str,
        student_state: Dict[str, float],
        planner_internals: Optional[Dict[str, Any]] = None
    ):
        """
        Log a dialogue turn.
        
        Args:
            dialogue_id: Unique dialogue identifier
            turn_number: Turn number in dialogue
            student_message: Student's message
            tutor_response: Tutor's response
            student_state: Detected student state
            planner_internals: Optional internal planner data (deliberations, scores, etc.)
        """
        data = {
            'dialogue_id': dialogue_id,
            'turn': turn_number,
            'student_message': student_message,
            'tutor_response': tutor_response,
            'student_state': student_state,
        }
        
        if planner_internals:
            data['planner_internals'] = planner_internals
        
        self.log_event('dialogue_turn', data)
    
    def init_wandb(self, project: str, config: Dict[str, Any]):
        """
        Initialize Weights & Biases logging.
        
        Args:
            project: W&B project name
            config: Experiment configuration
        """
        if not self.log_to_wandb:
            return
        
        self.wandb_run = self.wandb.init(
            project=project,
            name=self.experiment_name,
            config=config,
        )
        
        self.logger.info(f"W&B run initialized: {self.wandb_run.url}")
    
    def finish(self):
        """Finish experiment logging."""
        self.metadata['end_time'] = datetime.now().isoformat()
        
        # Save metadata
        metadata_file = self.experiment_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        self.logger.info(f"Experiment finished: {self.experiment_name}")
        
        # Finish W&B run
        if self.log_to_wandb and self.wandb_run:
            self.wandb_run.finish()


def log_experiment(func):
    """
    Decorator to automatically log experiment functions.
    
    Usage:
        @log_experiment
        def run_experiment(config):
            # Your experiment code
            pass
    """
    def wrapper(*args, **kwargs):
        logger = setup_logger()
        logger.info(f"Starting experiment: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            logger.info(f"Experiment completed successfully: {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Experiment failed: {func.__name__}", exc_info=True)
            raise
    
    return wrapper


if __name__ == "__main__":
    # Test logging
    logger = setup_logger()
    logger.info("Test log message")
    
    # Test experiment logger
    exp_logger = ExperimentLogger("test_experiment")
    exp_logger.log_metric("accuracy", 0.85, step=1)
    exp_logger.log_event("test", {"foo": "bar"})
    exp_logger.finish()