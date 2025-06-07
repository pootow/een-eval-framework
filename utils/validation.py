"""
Configuration validation utilities.

This module provides validation for configuration parameters and workflow setup.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class ConfigValidator:
    """Validates configuration parameters and setup."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_workflow(self, workflow) -> None:
        """Validate entire workflow configuration."""
        self.errors.clear()
        self.warnings.clear()
        
        self._validate_models(workflow._models)
        self._validate_dataset(workflow._dataset)
        self._validate_evaluation_methods(workflow._evaluation_methods)
        self._validate_metrics(workflow._metrics)
        self._validate_sample_params(workflow._sample_params)
        self._validate_prompt_template(workflow._eval_prompt_template)
        self._validate_output_dir(workflow._output_dir)
        
        if self.errors:
            error_msg = "Validation failed:\n" + "\n".join(f"- {error}" for error in self.errors)
            raise ValidationError(error_msg)
        
        if self.warnings:
            warning_msg = "Validation warnings:\n" + "\n".join(f"- {warning}" for warning in self.warnings)
            self.logger.warning(warning_msg)
    
    def _validate_models(self, models: List) -> None:
        """Validate model configurations."""
        if not models:
            self.errors.append("At least one model must be specified")
            return
        
        for i, model in enumerate(models):
            if not hasattr(model, 'config'):
                self.errors.append(f"Model {i}: Invalid model object")
                continue
            
            config = model.config
            
            # Validate model name
            if not config.name:
                self.errors.append(f"Model {i}: Name is required")
            
            # Validate model type specific requirements
            if config.type.value == "openai":
                if not config.api_key and not os.getenv("OPENAI_API_KEY"):
                    self.warnings.append(f"Model {i} ({config.name}): No API key provided")
                
                if config.endpoint and not self._is_valid_url(config.endpoint):
                    self.errors.append(f"Model {i} ({config.name}): Invalid endpoint URL")
            
            elif config.type.value == "vllm":
                if not config.model_path:
                    self.errors.append(f"Model {i} ({config.name}): model_path is required for VLLM models")
                elif not Path(config.model_path).exists():
                    self.warnings.append(f"Model {i} ({config.name}): model_path does not exist: {config.model_path}")
                
                if config.endpoint and not self._is_valid_url(config.endpoint):
                    self.errors.append(f"Model {i} ({config.name}): Invalid endpoint URL")
            
            # Validate sampling parameters
            if config.temperature < 0 or config.temperature > 2:
                self.warnings.append(f"Model {i} ({config.name}): temperature should be between 0 and 2")
            
            if config.top_p < 0 or config.top_p > 1:
                self.warnings.append(f"Model {i} ({config.name}): top_p should be between 0 and 1")
            
            if config.max_tokens < -1 or config.max_tokens == 0:
                self.warnings.append(f"Model {i} ({config.name}): max_tokens should be -1 (unlimited) or positive")
    
    def _validate_dataset(self, dataset) -> None:
        """Validate dataset."""
        if dataset is None:
            self.errors.append("Dataset is required")
            return
        
        if not hasattr(dataset, '__len__') or len(dataset) == 0:
            self.errors.append("Dataset is empty")
        
        # Validate dataset items
        if hasattr(dataset, 'items') and dataset.items:
            sample_item = dataset.items[0]
            if not hasattr(sample_item, 'data'):
                self.errors.append("Dataset items must have 'data' attribute")
    
    def _validate_evaluation_methods(self, methods: List) -> None:
        """Validate evaluation methods."""
        if not methods:
            self.errors.append("At least one evaluation method must be specified")
            return
        
        for i, method in enumerate(methods):
            if not hasattr(method, 'name'):
                self.errors.append(f"Evaluation method {i}: Missing name attribute")
                continue
            
            if not method.name:
                self.errors.append(f"Evaluation method {i}: Name cannot be empty")
            
            # Validate custom methods
            if hasattr(method, 'function'):
                try:
                    import inspect
                    sig = inspect.signature(method.function)
                    required_params = ['response', 'ground_truth']
                    for param in required_params:
                        if param not in sig.parameters:
                            self.errors.append(f"Evaluation method {i} ({method.name}): Function must have '{param}' parameter")
                except Exception as e:
                    self.errors.append(f"Evaluation method {i} ({method.name}): Could not validate function signature: {e}")
    
    def _validate_metrics(self, metrics: List) -> None:
        """Validate metrics."""
        if not metrics:
            self.warnings.append("No metrics specified - only basic statistics will be available")
            return
        
        for i, metric in enumerate(metrics):
            if not hasattr(metric, 'name'):
                self.errors.append(f"Metric {i}: Missing name attribute")
                continue
            
            if not metric.name:
                self.errors.append(f"Metric {i}: Name cannot be empty")
            
            # Validate pass@k metrics
            if hasattr(metric, 'k') and metric.k <= 0:
                self.errors.append(f"Metric {i} ({metric.name}): k must be positive")
    
    def _validate_sample_params(self, params: Dict[str, Any]) -> None:
        """Validate sampling parameters."""
        if not params:
            return
        
        # Validate temperature
        if 'temperature' in params:
            temp = params['temperature']
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                self.warnings.append("temperature should be a number between 0 and 2")
        
        # Validate top_p
        if 'top_p' in params:
            top_p = params['top_p']
            if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
                self.warnings.append("top_p should be a number between 0 and 1")
        
        # Validate top_k
        if 'top_k' in params:
            top_k = params['top_k']
            if not isinstance(top_k, int) or (top_k <= 0 and top_k != -1):
                self.warnings.append("top_k should be a positive integer or -1")
        
        # Validate max_tokens
        if 'max_tokens' in params:
            max_tokens = params['max_tokens']
            if not isinstance(max_tokens, int) or (max_tokens <= 0 and max_tokens != -1):
                self.warnings.append("max_tokens should be a positive integer or -1")
    
    def _validate_prompt_template(self, template: Optional[str]) -> None:
        """Validate prompt template."""
        if not template:
            self.warnings.append("No prompt template specified - using raw dataset content")
            return
        
        # Check for Jinja2 syntax
        jinja_pattern = r'\{\{.*?\}\}'
        if not re.search(jinja_pattern, template):
            self.warnings.append("Prompt template doesn't appear to use Jinja2 syntax (no {{ }} found)")
        
        # Check for common template variables
        common_vars = ['problem', 'question', 'input', 'context', 'constraints']
        found_vars = re.findall(r'\{\{\s*(\w+)', template)
        
        if found_vars:
            self.logger.debug(f"Found template variables: {found_vars}")
        else:
            self.warnings.append("No template variables found in prompt template")
    
    def _validate_output_dir(self, output_dir: Optional[str]) -> None:
        """Validate output directory."""
        if not output_dir:
            return
        
        output_path = Path(output_dir)
        
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            self.errors.append(f"Permission denied creating output directory: {output_dir}")
        except Exception as e:
            self.errors.append(f"Could not create output directory {output_dir}: {e}")
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return url_pattern.match(url) is not None
    
    def validate_environment(self) -> None:
        """Validate environment setup."""
        # Check Python dependencies
        required_packages = [
            'openai',
            'requests', 
            'jinja2',
            'pyyaml'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                self.errors.append(f"Required package not installed: {package}")
        
        # Check Docker availability (for VLLM models)
        try:
            import subprocess
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                self.warnings.append("Docker not available - VLLM models will not work")
        except FileNotFoundError:
            self.warnings.append("Docker not found - VLLM models will not work")
        
        # Check NVIDIA Docker runtime (for GPU support)
        try:
            import subprocess
            result = subprocess.run(['docker', 'info'], capture_output=True, text=True)
            if 'nvidia' not in result.stdout.lower():
                self.warnings.append("NVIDIA Docker runtime not detected - GPU acceleration may not work")
        except Exception:
            pass  # Docker info failed, already warned about Docker
        
        if self.errors:
            error_msg = "Environment validation failed:\n" + "\n".join(f"- {error}" for error in self.errors)
            raise ValidationError(error_msg)
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get validation report."""
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "passed": len(self.errors) == 0
        }
