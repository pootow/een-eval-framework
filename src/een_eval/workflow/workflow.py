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
from ..workflow.config import Config, EvaluationMethodConfig, MetricConfig, DatasetConfig
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
    failed_samples: int = 0
    current_model: Optional[str] = None
    start_time: Optional[datetime] = None
    last_update: Optional[datetime] = None
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
            "progress_percent": self.progress_percent,
            "processed_samples": self.processed_samples,
            "failed_samples": self.failed_samples,
            "current_model": self.current_model,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "errors": self.errors,
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
        inference_engine_cls: Optional[type] = None,  # <-- NEW
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
        self._inference_engine_cls = inference_engine_cls  # <-- NEW
        
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
        overrides: Optional[Dict[str, Any]] = None,
        inference_engine_cls: Optional[type] = None
    ) -> "EvalWorkflow":
        """
        Create workflow from configuration file.
        
        Args:
            config_path: Path to configuration file
            overrides: Dictionary of parameter overrides
            
        Returns:
            Configured EvalWorkflow instance
        """
        workflow = cls(config=config_path, inference_engine_cls=inference_engine_cls)
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
        if hasattr(self._config, 'resume'):
            self._resume = self._config.resume
        if hasattr(self._config, 'mode'):
            self._mode = self._config.mode
        # REVIEW: find if any other parameters forget to apply from config

        self._config.setup_environment()
        # Support custom inference engine from config (file path or module)
        from een_eval.utils.module_loader import CustomLoader
        engine_spec = getattr(self._config, 'inference_engine', None)
        if engine_spec:
            if isinstance(engine_spec, dict):
                path = engine_spec.get('path')
                class_name = engine_spec.get('class')
                module = engine_spec.get('module')
                if path:
                    self._inference_engine_cls = CustomLoader.load_object(path=path, object_name=class_name)
                elif module:
                    self._inference_engine_cls = CustomLoader.load_object(module=module, object_name=class_name)
            elif isinstance(engine_spec, str):
                # fallback: treat as module path
                module_name, class_name = engine_spec.rsplit('.', 1)
                import importlib
                module = importlib.import_module(module_name)
                self._inference_engine_cls = getattr(module, class_name)

    
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

    def _set_dataset(self, dataset: Union[str, Dataset, Dict[str, Any], DatasetConfig]) -> None:
        """Set dataset for evaluation."""
        if isinstance(dataset, str):
            self._dataset = Dataset.from_file(dataset)
        elif isinstance(dataset, DatasetConfig):
            # Handle DatasetConfig objects
            self._dataset = Dataset.from_config(dataset)
        elif isinstance(dataset, dict):
            # Handle dataset configuration from config files
            if "file_path" in dataset:
                self._dataset = Dataset.from_file(dataset["file_path"])
            else:
                # Create dataset from dictionary data
                self._dataset = Dataset.from_dict(dataset)
        else:
            self._dataset = dataset
    
    def _set_evaluation_methods(self, methods: List[Union[str, EvaluationMethod, EvaluationMethodConfig]]) -> None:
        """Set evaluation methods."""
        self._evaluation_methods = []
        for method in methods:
            if isinstance(method, str):
                self._evaluation_methods.append(BuiltInEvaluationMethod.create(method))
            elif isinstance(method, EvaluationMethodConfig):
                self._evaluation_methods.append(EvaluationMethod.from_config(method))
            else:
                self._evaluation_methods.append(method)

    def _set_metrics(self, metrics: List[Union[str, Metric, MetricConfig]]) -> None:
        """Set metrics."""
        self._metrics = []
        for metric in metrics:
            if isinstance(metric, str):
                self._metrics.append(BuiltInMetric.create(metric))
            elif isinstance(metric, MetricConfig):
                self._metrics.append(Metric.from_config(metric))
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
        
        # Try to resume if requested, otherwise clean output files for fresh start
        # Only clean files when starting fresh inference (not evaluation)
        if self._resume:
            # TODO: Test resume logic, ensure it
            # - won't overwrite existing data
            # - can handle partial runs, recovering from errors
            # - won't generate data more than needed
            self._try_resume()
        elif self._mode == "inference":
            # Clean output files to avoid reading data from previous runs
            # Only when starting fresh inference - evaluation mode needs previous inference data
            self._output_manager.clean_output_files()

        # Setup engines
        if not self._dataset:
            raise ValueError("Dataset must be specified")
        engine_cls = self._inference_engine_cls or InferenceEngine
        self._inference_engine = engine_cls(
            models=self._models,
            dataset=self._dataset,
            sample_params=self._sample_params,
            prompt_template=self._eval_prompt_template,
            output_manager=self._output_manager,
            batch_size=self._config.batch_size if self._config else 1,
            max_workers=self._config.max_workers if self._config else 1,
            resume=self._resume,
            limit=self._config.limit if self._config else None
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
                total_samples=len(self._dataset) * len(self._models) * self._sample_params.get("num_samples", 1) if self._dataset else 0,
                start_time=datetime.now()
            )

    def _try_resume(self) -> None:
        """Attempt to resume from previous run."""
        try:
            if not self._output_dir:
                return
            
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
        
        if not self._inference_engine:
            raise ValueError("Inference engine not initialized")
        if not self._output_manager:
            raise ValueError("Output manager not initialized")
        
        results = self._inference_engine.run(self._status)
        
        # Save status
        self._output_manager.save_status(self._status)
        
        return results
    
    def _run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation mode."""
        self._logger.info("Starting evaluation mode")
        self._status.mode = "evaluation"
        
        if not self._evaluation_engine:
            raise ValueError("Evaluation engine not initialized")
        if not self._output_manager:
            raise ValueError("Output manager not initialized")
        
        results = self._evaluation_engine.run(self._status)
        
        # Mark as complete
        self._status.mode = "complete"
        self._output_manager.save_status(self._status)
        
        return results
