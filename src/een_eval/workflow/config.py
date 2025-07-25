"""
Configuration handling for the evaluation framework.

This module provides configuration loading, validation, and management
from various sources (YAML files, dictionaries, environment variables).
"""

import os
import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from ..core.models import ModelConfig, ModelType


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    type: str = "built_in"  # "built_in", "custom", or "file"
    path: Optional[str] = None  # Custom file containing dataset function
    module: Optional[str] = None  # Module name for custom dataset function
    function_name: Optional[str] = None  # Function name for custom dataset loading
    params: Dict[str, Any] = field(default_factory=dict)  # Parameters for dataset function
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        if isinstance(data, str):
            # Simple string format - assume it's a file path
            return cls(type="file", path=data)
        
        return cls(
            type=data.get("type", "built_in"),
            path=data.get("path"),
            module=data.get("module"),
            function_name=data.get("function_name"),
            params=data.get("params", {})
        )


@dataclass
class EvaluationMethodConfig:
    """Configuration for evaluation method."""
    name: str
    type: str = "built_in"  # "built_in" or "custom"
    path: Optional[str] = None
    module: Optional[str] = None
    function_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationMethodConfig":
        return cls(
            name=data["name"],
            type=data.get("type", "built_in"),
            path=data.get("path"),
            module=data.get("module"),
            function_name=data.get("function_name"),
            params=data.get("params", {})
        )


@dataclass
class MetricConfig:
    """Configuration for metric."""
    name: str
    type: str = "built_in"  # "built_in" or "custom"
    path: Optional[str] = None
    module: Optional[str] = None
    function_name: Optional[str] = None
    facets: List[str] = field(default_factory=list)  # Added facets support
    labels: Optional[Dict[str, List[str]]] = field(default_factory=dict)  # support include/exclude labels
    params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricConfig":
        return cls(
            name=data["name"],
            type=data.get("type", "built_in"),
            path=data.get("path"),
            module=data.get("module"),
            function_name=data.get("function_name"),
            facets=data.get("facets", []),
            labels=data.get("labels", {}),
            params=data.get("params", {})
        )


@dataclass
class Config:
    """Main configuration for evaluation workflow."""
    # Core components
    models: List[ModelConfig] = field(default_factory=list)
    dataset: Optional[DatasetConfig] = None  # Changed from str to DatasetConfig
    evaluation_methods: List[EvaluationMethodConfig] = field(default_factory=list)
    metrics: List[MetricConfig] = field(default_factory=list)
    
    # Sampling parameters
    sample_params: Dict[str, Any] = field(default_factory=dict)
    
    # Templates and prompts
    eval_prompt_template: Optional[str] = None
    
    # Execution parameters
    mode: str = "inference"  # "inference" or "evaluation"
    batch_size: int = 1
    max_workers: int = 4
    timeout: int = 300
    max_retries: int = 3
    limit: Optional[int] = None  # Limit number of samples processed
    
    # Output configuration
    output_dir: Optional[str] = None
    save_intermediate: bool = True
    
    # Resume configuration
    resume: bool = False
    resume_from: Optional[str] = None
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Environment configuration
    env_vars: Dict[str, str] = field(default_factory=dict)
    inference_engine: Optional[Union[str, Dict[str, str]]] = None  # Custom inference engine support
    
    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load configuration from file."""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration format: {path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        config = cls()

        # Load models
        if "models" in data:
            config.models = []
            for model_data in data["models"]:
                if isinstance(model_data, str):
                    config.models.append(ModelConfig.from_name(model_data))
                else:
                    config.models.append(ModelConfig.from_dict(model_data))
        
        # Load dataset configuration
        if "dataset" in data:
            config.dataset = DatasetConfig.from_dict(data["dataset"])
        
        # Load evaluation methods
        if "evaluation_methods" in data:
            config.evaluation_methods = []
            for method_data in data["evaluation_methods"]:
                if isinstance(method_data, str):
                    config.evaluation_methods.append(
                        EvaluationMethodConfig(name=method_data)
                    )
                else:
                    config.evaluation_methods.append(
                        EvaluationMethodConfig.from_dict(method_data)
                    )
        
        # Load metrics
        if "metrics" in data:
            config.metrics = []
            for metric_data in data["metrics"]:
                if isinstance(metric_data, str):
                    config.metrics.append(MetricConfig(name=metric_data))
                else:
                    config.metrics.append(MetricConfig.from_dict(metric_data))

        # Load other parameters
        config.sample_params = data.get("sample_params", {})
        
        # Handle prompt template - support both inline and file-based templates
        config.eval_prompt_template = cls._load_prompt_template(data)
        
        workflow_config = data.get("workflow", {})
        config.mode = workflow_config.get("mode", "inference")
        config.batch_size = workflow_config.get("batch_size", 1)
        config.max_workers = workflow_config.get("max_workers", 4)

        config.timeout = data.get("timeout", 300)
        config.max_retries = data.get("max_retries", 3)
        config.output_dir = data.get("output_dir")
        config.save_intermediate = data.get("save_intermediate", True)
        config.resume = data.get("resume", False)
        config.resume_from = data.get("resume_from")
        config.limit = data.get("limit", None)

        if "logging" in data:
            logging_data = data["logging"]
            config.log_level = logging_data.get("level", "INFO")
            config.log_file = logging_data.get("file")

        config.env_vars = data.get("env_vars", {})
        
        # Load custom inference engine if specified
        config.inference_engine = data.get("inference_engine")
        
        return config
    
    @classmethod
    def from_env(cls, prefix: str = "EVAL_") -> "Config":
        """Load configuration from environment variables."""
        config = cls()
        
        # Simple mapping for basic parameters
        env_mappings = {
            f"{prefix}MODE": "mode",
            f"{prefix}BATCH_SIZE": "batch_size",
            f"{prefix}MAX_WORKERS": "max_workers",
            f"{prefix}TIMEOUT": "timeout",
            f"{prefix}MAX_RETRIES": "max_retries",
            f"{prefix}OUTPUT_DIR": "output_dir",
            f"{prefix}LOG_LEVEL": "log_level",
            f"{prefix}LOG_FILE": "log_file",
            f"{prefix}RESUME": "resume"
        }
        
        for env_var, config_attr in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                
                # Type conversion
                if config_attr in ["batch_size", "max_workers", "timeout", "max_retries"]:
                    value = int(value)
                elif config_attr in ["resume"]:
                    value = value.lower() in ("true", "1", "yes", "on")
                
                setattr(config, config_attr, value)
        
        # Load dataset from env vars
        if f"{prefix}DATASET" in os.environ:
            dataset_path = os.environ[f"{prefix}DATASET"]
            config.dataset = DatasetConfig(type="file", path=dataset_path)
        
        # Load sample parameters from env vars
        sample_param_prefix = f"{prefix}SAMPLE_"
        for key, value in os.environ.items():
            if key.startswith(sample_param_prefix):
                param_name = key[len(sample_param_prefix):].lower()
                
                # Try to convert to appropriate type
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    pass  # Keep as string
                
                config.sample_params[param_name] = value
        
        return config
    
    def merge(self, other: "Config") -> "Config":
        """Merge with another configuration (other takes precedence)."""
        merged_data = asdict(self)
        other_data = asdict(other)
        
        # Deep merge dictionaries
        for key, value in other_data.items():
            if isinstance(value, dict) and key in merged_data:
                merged_data[key] = {**merged_data[key], **value}
            elif value is not None and value != [] and value != {}:
                merged_data[key] = value
        
        return Config.from_dict(merged_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, file_path: str, format: str = "yaml") -> None:
        """Save configuration to file."""
        data = self.to_dict()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if format.lower() in ["yaml", "yml"]:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    # REVIEW: validate should be called automatically after loading, otherwise user might forget to call it
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Check required fields
        if not self.models:
            errors.append("At least one model must be specified")
        
        if not self.dataset:
            errors.append("Dataset must be specified")
        
        if not self.evaluation_methods:
            errors.append("At least one evaluation method must be specified")
        
        # Validate models
        for i, model in enumerate(self.models):
            if not model.name:
                errors.append(f"Model {i}: name is required")
            
            if model.type == ModelType.VLLM and not model.model_path:
                errors.append(f"Model {i}: model_path is required for VLLM models")
            
            if model.type == ModelType.OPENAI and not model.endpoint and not model.api_key:
                errors.append(f"Model {i}: endpoint or api_key is required for OpenAI models")
        
        # Validate dataset
        if self.dataset:
            if self.dataset.type == "custom" and not (self.dataset.path or self.dataset.module):
                errors.append("Dataset: file or module is required for custom dataset")
            
            if self.dataset.type == "custom" and not self.dataset.function_name:
                errors.append("Dataset: function_name is required for custom dataset")
            
            if self.dataset.type == "file" and not self.dataset.path:
                errors.append("Dataset: path is required for file dataset")
        
        # Validate evaluation methods
        for i, method in enumerate(self.evaluation_methods):
            if not method.name:
                errors.append(f"Evaluation method {i}: name is required")
            
            if method.type == "custom" and not method.path:
                errors.append(f"Evaluation method {i}: path is required for custom methods")
        
        # Validate metrics
        for i, metric in enumerate(self.metrics):
            if not metric.name:
                errors.append(f"Metric {i}: name is required")
            
            if metric.type == "custom" and not metric.path:
                errors.append(f"Metric {i}: path is required for custom metrics")
        
        # Validate parameters
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if self.max_workers <= 0:
            errors.append("max_workers must be positive")
        
        if self.timeout <= 0:
            errors.append("timeout must be positive")
        
        if self.mode not in ["inference", "evaluation"]:
            errors.append("mode must be 'inference' or 'evaluation'")
        
        return errors
    
    def setup_environment(self) -> None:
        """Setup environment variables from configuration."""
        for key, value in self.env_vars.items():
            os.environ[key] = value
        
        # Setup logging
        import logging
        
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
        
        # File handler
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            handlers=handlers,
            force=True
        )
    
    @staticmethod
    def _load_prompt_template(data: Dict[str, Any]) -> Optional[str]:
        """Load prompt template from configuration data.
        
        Supports both inline templates and file-based templates.
        Handles the following formats:
        - eval_prompt_template: "inline content"
        - prompt_template: "inline content"  
        - prompt_template: "path/to/file.md"
        - prompt_template: {path: "path/to/file.md"}
        """
        # First check for eval_prompt_template (direct inline)
        if "eval_prompt_template" in data:
            return data["eval_prompt_template"]
        
        # Then check for prompt_template
        if "prompt_template" not in data:
            return None
            
        prompt_template = data["prompt_template"]
        
        # Handle different prompt_template formats
        if isinstance(prompt_template, str):
            # Could be inline content or a file path
            if os.path.exists(prompt_template):
                # It's a file path
                try:
                    with open(prompt_template, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    raise ValueError(f"Failed to read prompt template file '{prompt_template}': {e}")
            else:
                # It's inline content
                return prompt_template
                
        elif isinstance(prompt_template, dict):
            # Handle {path: "filename"} format
            if "path" in prompt_template:
                file_path = prompt_template["path"]
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Prompt template file not found: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    raise ValueError(f"Failed to read prompt template file '{file_path}': {e}")
            else:
                raise ValueError("prompt_template dict must contain a 'path' key")
        else:
            raise ValueError(f"prompt_template must be a string or dict, got {type(prompt_template)}")
