"""
Een Eval - A flexible evaluation framework for language models.

This framework provides a comprehensive solution for evaluating language models
with support for custom evaluation methods, metrics, and flexible configuration.
"""

__version__ = "0.1.0"

from .workflow.workflow import EvalWorkflow
from .workflow.evaluation import FinalEvaluationResult
from .core.models import Model, ModelConfig, InferenceResult, SimpleInferenceResult
from .core.evaluation import EvaluationMethod
from .core.metrics import Metric
from .workflow.config import Config

__version__ = "0.1.0"
__all__ = [
    "EvalWorkflow",
    "Model", 
    "ModelConfig",
    "InferenceResult",
    "SimpleInferenceResult",
    "EvaluationMethod",
    "FinalEvaluationResult",
    "Metric",
    "Config"
]
