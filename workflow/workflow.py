"""
Main workflow orchestration for the evaluation framework.

This module provides the primary client interface for the evaluation framework,
supporting both programmatic usage and configuration-based setup.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field

from ..core.models import Model, ModelConfig
from ..core.evaluation import EvaluationMethod, BuiltInEvaluationMethod
from ..core.metrics import Metric, BuiltInMetric
from ..core.dataset import Dataset
from ..workflow.config import Config
from ..workflow.inference import InferenceEngine
from ..workflow.evaluation import EvaluationEngine
from ..utils.io import OutputManager
from ..utils.validation import ConfigValidator


@dataclass
class WorkflowStatus:
    """Tracks the current status of the evaluation workflow."""
    mode: str = "inference"  # "inference", "evaluation", "complete"
    total_samples: int = 0
    processed_samples: int = 0
    current_model: Optional[str] = None
    start_time: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    
    @property
    def progress_percent(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return (self.processed_samples / self.total_samples) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "total_samples": self.total_samples,
            "processed_samples": self.processed_samples,
            "current_model": self.current_model,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "errors": self.errors,
            "progress_percent": self.progress_percent
        }


class EvalWorkflow:
    """
    Main workflow orchestrator for model evaluation.
    
    Supports builder pattern for programmatic setup and configuration-based setup.
    Handles both inference and evaluation modes with resume capability.
    """
    
    def __init__(
        self,
        models: Optional[List[Union[str, Model, ModelConfig]]] = None,
        dataset: Optional[Union[str, Dataset]] = None,
        evaluation_methods: Optional[List[Union[str, EvaluationMethod]]] = None,
        metrics: Optional[List[Union[str, Metric]]] = None,
        sample_params: Optional[Dict[str, Any]] = None,
        eval_prompt_template: Optional[str] = None,
        output_dir: Optional[str] = None,
        mode: str = "inference",
        resume: bool = False,
        config: Optional[Union[str, Dict[str, Any], Config]] = None,
        **kwargs
    ):
        """
        Initialize the evaluation workflow.
        
        Args:
            models: List of models to evaluate
            dataset: Dataset to use for evaluation
            evaluation_methods: List of evaluation methods to apply
            metrics: List of metrics to calculate
            sample_params: Parameters for model sampling
            eval_prompt_template: Jinja2 template for prompts
            output_dir: Directory for output files
            mode: Workflow mode ("inference" or "evaluation")
            resume: Whether to resume from previous run
            config: Configuration file path or dict
            **kwargs: Additional configuration parameters
        """
        self._models: List[Model] = []
        self._dataset: Optional[Dataset] = None
        self._evaluation_methods: List[EvaluationMethod] = []
        self._metrics: List[Metric] = []
        self._sample_params: Dict[str, Any] = {}
        self._eval_prompt_template: Optional[str] = None
        self._output_dir: Optional[str] = None
        self._mode: str = mode
        self._resume: bool = resume
        self._config: Optional[Config] = None
        
        # Internal state
        self._status = WorkflowStatus()
        self._output_manager: Optional[OutputManager] = None
        self._inference_engine: Optional[InferenceEngine] = None
        self._evaluation_engine: Optional[EvaluationEngine] = None
        self._logger = logging.getLogger(__name__)
        
        # Load configuration first if provided
        if config is not None:
            self._load_config(config)
        
        # Override with explicit parameters (precedence: explicit > config)
        if models is not None:
            self._set_models(models)
        if dataset is not None:
            self._set_dataset(dataset)
        if evaluation_methods is not None:
            self._set_evaluation_methods(evaluation_methods)
        if metrics is not None:
            self._set_metrics(metrics)
        if sample_params is not None:
            self._sample_params.update(sample_params)
        if eval_prompt_template is not None:
            self._eval_prompt_template = eval_prompt_template
        if output_dir is not None:
            self._output_dir = output_dir
            
        # Apply any additional kwargs
        for key, value in kwargs.items():
            if hasattr(self, f"_set_{key}"):
                getattr(self, f"_set_{key}")(value)
    
    @classmethod
    def from_config(
        cls, 
        config_path: str, 
        overrides: Optional[Dict[str, Any]] = None
    ) -> "EvalWorkflow":
        """
        Create workflow from configuration file.
        
        Args:
            config_path: Path to configuration file
            overrides: Dictionary of parameter overrides
            
        Returns:
            Configured EvalWorkflow instance
        """
        workflow = cls(config=config_path)
        if overrides:
            for key, value in overrides.items():
                if hasattr(workflow, f"_set_{key}"):
                    getattr(workflow, f"_set_{key}")(value)
        return workflow
    
    # Builder pattern methods
    def add_models(self, models: List[Union[str, Model, ModelConfig]]) -> "EvalWorkflow":
        """Add models to evaluate."""
        self._set_models(models)
        return self
    
    def load_dataset(self, dataset: Union[str, Dataset]) -> "EvalWorkflow":
        """Set the dataset for evaluation."""
        self._set_dataset(dataset)
        return self
    
    def add_evaluation_method(
        self, 
        name_or_method: Union[str, EvaluationMethod],
        path: Optional[str] = None,
        function_name: Optional[str] = None,
        **params
    ) -> "EvalWorkflow":
        """Add an evaluation method."""
        if isinstance(name_or_method, str) and path:
            method = EvaluationMethod.from_file(name_or_method, path, function_name, **params)
        elif isinstance(name_or_method, str):
            method = BuiltInEvaluationMethod.create(name_or_method, **params)
        else:
            method = name_or_method
        
        self._evaluation_methods.append(method)
        return self
    
    def add_metric(
        self, 
        name_or_metric: Union[str, Metric],
        path: Optional[str] = None,
        function_name: Optional[str] = None,
        **params
    ) -> "EvalWorkflow":
        """Add a metric."""
        if isinstance(name_or_metric, str) and path:
            metric = Metric.from_file(name_or_metric, path, function_name, **params)
        elif isinstance(name_or_metric, str):
            metric = BuiltInMetric.create(name_or_metric, **params)
        else:
            metric = name_or_metric
        
        self._metrics.append(metric)
        return self
    
    def set_sample_params(self, **params) -> "EvalWorkflow":
        """Set sampling parameters."""
        self._sample_params.update(params)
        return self
    
    def set_prompt_template(self, template: str) -> "EvalWorkflow":
        """Set the prompt template."""
        self._eval_prompt_template = template
        return self
    
    def set_output_dir(self, output_dir: str) -> "EvalWorkflow":
        """Set the output directory."""
        self._output_dir = output_dir
        return self
    
    def set_mode(self, mode: str) -> "EvalWorkflow":
        """Set the workflow mode."""
        if mode not in ["inference", "evaluation"]:
            raise ValueError("Mode must be 'inference' or 'evaluation'")
        self._mode = mode
        return self
    
    # Core workflow methods
    def run(self) -> Dict[str, Any]:
        """
        Run the evaluation workflow.
        
        Returns:
            Dictionary containing results and status information
        """
        try:
            self._setup_workflow()
            
            if self._mode == "inference":
                return self._run_inference()
            elif self._mode == "evaluation":
                return self._run_evaluation()
            else:
                raise ValueError(f"Unknown mode: {self._mode}")
                
        except Exception as e:
            self._logger.error(f"Workflow failed: {e}")
            self._status.errors.append(str(e))
            raise
    
    def is_complete(self) -> bool:
        """Check if the workflow is complete."""
        return self._status.mode == "complete"
    
    @property
    def status(self) -> WorkflowStatus:
        """Get current workflow status."""
        return self._status
    
    def get_results(self) -> Optional[Dict[str, Any]]:
        """Get results from the last run."""
        if not self._output_manager:
            return None
        return self._output_manager.load_results()
    
    # Private methods
    def _load_config(self, config: Union[str, Dict[str, Any], Config]) -> None:
        """Load configuration from file or dict."""
        if isinstance(config, str):
            self._config = Config.from_file(config)
        elif isinstance(config, dict):
            self._config = Config.from_dict(config)
        else:
            self._config = config
        
        # Apply configuration
        if self._config.models:
            self._set_models(self._config.models)
        if self._config.dataset:
            self._set_dataset(self._config.dataset)
        if self._config.evaluation_methods:
            self._set_evaluation_methods(self._config.evaluation_methods)
        if self._config.metrics:
            self._set_metrics(self._config.metrics)
        if self._config.sample_params:
            self._sample_params.update(self._config.sample_params)
        if self._config.eval_prompt_template:
            self._eval_prompt_template = self._config.eval_prompt_template
        if self._config.output_dir:
            self._output_dir = self._config.output_dir
    
    def _set_models(self, models: List[Union[str, Model, ModelConfig]]) -> None:
        """Set models for evaluation."""
        self._models = []
        for model in models:
            if isinstance(model, str):
                self._models.append(Model.from_name(model))
            elif isinstance(model, ModelConfig):
                self._models.append(Model.from_config(model))
            else:
                self._models.append(model)

    def _set_dataset(self, dataset: Union[str, Dataset, Dict[str, Any]]) -> None:
        """Set dataset for evaluation."""
        if isinstance(dataset, str):
            self._dataset = Dataset.from_file(dataset)
        elif isinstance(dataset, dict):
            # Handle dataset configuration from config files
            if "file_path" in dataset:
                self._dataset = Dataset.from_file(dataset["file_path"])
            else:
                # Create dataset from dictionary data
                self._dataset = Dataset.from_dict(dataset)
        else:
            self._dataset = dataset
    
    def _set_evaluation_methods(self, methods: List[Union[str, EvaluationMethod]]) -> None:
        """Set evaluation methods."""
        self._evaluation_methods = []
        for method in methods:
            if isinstance(method, str):
                self._evaluation_methods.append(BuiltInEvaluationMethod.create(method))
            else:
                self._evaluation_methods.append(method)
    
    def _set_metrics(self, metrics: List[Union[str, Metric]]) -> None:
        """Set metrics."""
        self._metrics = []
        for metric in metrics:
            if isinstance(metric, str):
                self._metrics.append(BuiltInMetric.create(metric))
            else:
                self._metrics.append(metric)
    
    def _setup_workflow(self) -> None:
        """Setup workflow components."""
        # Validate configuration
        validator = ConfigValidator()
        validator.validate_workflow(self)
        
        # Setup output manager
        if not self._output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_dir = f"output/{timestamp}"
        
        self._output_manager = OutputManager(self._output_dir)
        
        # Try to resume if requested
        if self._resume:
            self._try_resume()
        
        # Setup engines
        self._inference_engine = InferenceEngine(
            models=self._models,
            dataset=self._dataset,
            sample_params=self._sample_params,
            prompt_template=self._eval_prompt_template,
            output_manager=self._output_manager
        )
        
        self._evaluation_engine = EvaluationEngine(
            evaluation_methods=self._evaluation_methods,
            metrics=self._metrics,
            output_manager=self._output_manager
        )
        
        # Initialize status
        if not self._resume:
            self._status = WorkflowStatus(
                mode=self._mode,
                total_samples=len(self._dataset) * len(self._models) if self._dataset else 0,
                start_time=datetime.now()
            )
    
    def _try_resume(self) -> None:
        """Attempt to resume from previous run."""
        try:
            status_file = Path(self._output_dir) / "status.json"
            if status_file.exists():
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                
                self._status.processed_samples = status_data.get("processed_samples", 0)
                self._status.total_samples = status_data.get("total_samples", 0)
                self._status.current_model = status_data.get("current_model")
                self._status.errors = status_data.get("errors", [])
                
                self._logger.info(f"Resumed from {self._status.processed_samples}/{self._status.total_samples} samples")
        except Exception as e:
            self._logger.warning(f"Could not resume from previous run: {e}")
    
    def _run_inference(self) -> Dict[str, Any]:
        """Run inference mode."""
        self._logger.info("Starting inference mode")
        self._status.mode = "inference"
        
        results = self._inference_engine.run(self._status)
        
        # Save status
        self._output_manager.save_status(self._status)
        
        return results
    
    def _run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation mode."""
        self._logger.info("Starting evaluation mode")
        self._status.mode = "evaluation"
        
        results = self._evaluation_engine.run(self._status)
        
        # Mark as complete
        self._status.mode = "complete"
        self._output_manager.save_status(self._status)
        
        return results
