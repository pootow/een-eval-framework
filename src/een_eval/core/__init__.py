"""
Core evaluation framework components.

This package contains the fundamental building blocks for the evaluation framework:
- Model abstractions and interfaces
- Evaluation methods and scoring functions
- Metrics computation and aggregation
- Dataset loading and processing utilities
"""

from .models import Model, ModelConfig, OpenAIModel, VLLMModel
from .evaluation import EvaluationMethod, BuiltInEvaluationMethod
from .metrics import Metric, BuiltInMetric
from .dataset import Dataset

__all__ = [
    'Model', 'ModelConfig', 'OpenAIModel', 'VLLMModel',
    'EvaluationMethod', 'BuiltInEvaluationMethod',
    'Metric', 'BuiltInMetric',
    'Dataset'
]
