"""
Metrics calculation for the evaluation framework.

This module provides both built-in and custom metrics for aggregating
evaluation results across multiple samples.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Callable, Optional, Union
import statistics
import math
import importlib.util
from pathlib import Path
from collections import defaultdict

from een_eval.workflow.config import MetricConfig


@dataclass
class MetricResult:
    """Result from metric calculation."""
    metric_name: str
    value: Union[float, int, Dict[str, Any]]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "metadata": self.metadata
        }


class Metric(ABC):
    """Abstract base class for metrics."""
    
    def __init__(self, name: str, **params):
        self.name = name
        self.params = params
    
    @abstractmethod
    def calculate(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> MetricResult:
        """
        Calculate metric from evaluation results.
        
        Args:
            evaluation_results: List of evaluation results from all samples
            **kwargs: Additional context or parameters
            
        Returns:
            MetricResult containing calculated metric
        """
        pass
    
    @classmethod
    def from_function(
        cls, 
        name: str, 
        function: Callable,
        **params
    ) -> "Metric":
        """Create metric from function."""
        return CustomMetric(name, function, **params)
    
    @classmethod
    def from_file(
        cls, 
        name: str, 
        file_path: str, 
        function_name: Optional[str] = None,
        **params
    ) -> "Metric":
        """Create metric from external file."""
        function = _load_function_from_file(file_path, function_name or name)
        return cls.from_function(name, function, **params)

    @classmethod
    def from_config(cls, config: MetricConfig) -> "Metric":
        """Create metric from configuration object."""
        if config.type == "built_in":
            return BuiltInMetric.create(config.name, **config.params)
        elif config.type == "custom":
            if config.path:
                # File-based custom metric
                return cls.from_file(config.name, config.path, config.function_name, **config.params)
            elif hasattr(config, 'module') and hasattr(config, 'function_name'):
                # Module-based custom metric
                try:
                    import importlib
                    module = importlib.import_module(config.module)
                    function = getattr(module, config.function_name)
                    return cls.from_function(config.name, function, **config.params)
                except (ImportError, AttributeError) as e:
                    raise ValueError(f"Could not load function {config.function_name} from module {config.module}: {e}")
            else:
                raise ValueError("Custom metric requires either 'path' or 'module'+'function_name'")
        else:
            raise ValueError(f"Unknown metric type: {config.type}")


class CustomMetric(Metric):
    """Custom metric using user-provided function."""
    
    def __init__(self, name: str, function: Callable, **params):
        super().__init__(name, **params)
        self.function = function
    
    def calculate(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> MetricResult:
        """Calculate using custom function."""
        try:
            result = self.function(
                evaluation_results=evaluation_results,
                **self.params,
                **kwargs
            )
            
            # Normalize result format
            if isinstance(result, dict):
                value = result
                metadata = {}
            elif isinstance(result, (int, float)):
                value = result
                metadata = {}
            else:
                value = result
                metadata = {}
            
            return MetricResult(
                metric_name=self.name,
                value=value,
                metadata=metadata
            )
            
        except Exception as e:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                metadata={"error": str(e)}
            )


class BuiltInMetric(Metric):
    """Built-in metrics."""
    
    @classmethod
    def create(cls, metric_name: str, **params) -> Metric:
        """Create built-in metric."""
        metric_map = {
            "pass_at_k": PassAtKMetric,
            "mean": MeanMetric,
            "median": MedianMetric,
            "percentile": PercentileMetric,
            "pass_rate": PassRateMetric,
            "count": CountMetric,
            "std": StandardDeviationMetric,
            "min": MinMetric,
            "max": MaxMetric,
            "sum": SumMetric
        }
        
        if metric_name not in metric_map:
            raise ValueError(f"Unknown built-in metric: {metric_name}")
        
        return metric_map[metric_name](**params)


class PassAtKMetric(Metric):
    """Pass@K metric - probability that at least one of K samples passes."""
    
    def __init__(self, k: int = 1, num_samples: int = 1, **params):
        super().__init__(f"pass_at_{k}", **params)
        self.k = k
        self.num_samples = num_samples
    
    def calculate(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> MetricResult:
        """Calculate Pass@K metric."""
        # Group results by problem
        problems = defaultdict(list)
        for result in evaluation_results:
            problem_id = result.get("problem_id", "default")
            problems[problem_id].append(result)
        
        pass_at_k_scores = []
        
        for problem_id, problem_results in problems.items():
            # Count passed samples for this problem
            passed_count = sum(1 for r in problem_results if r.get("passed", False))
            total_samples = len(problem_results)
            
            if total_samples < self.k:
                # Not enough samples, use what we have
                k_effective = total_samples
            else:
                k_effective = self.k
            
            # Calculate pass@k using combinatorial formula
            if passed_count >= k_effective:
                pass_at_k = 1.0
            else:
                pass_at_k = 0.0
            
            pass_at_k_scores.append(pass_at_k)
        
        mean_pass_at_k = statistics.mean(pass_at_k_scores) if pass_at_k_scores else 0.0
        
        return MetricResult(
            metric_name=self.name,
            value=mean_pass_at_k,
            metadata={
                "k": self.k,
                "num_problems": len(problems),
                "total_samples": len(evaluation_results),
                "individual_scores": pass_at_k_scores
            }
        )


class MeanMetric(Metric):
    """Mean/average metric."""
    
    def __init__(self, field: str = "score", **params):
        super().__init__("mean", **params)
        self.field = field
    
    def calculate(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> MetricResult:
        """Calculate mean of specified field."""
        values = []
        for result in evaluation_results:
            if self.field in result:
                value = result[self.field]
                if isinstance(value, (int, float)):
                    values.append(value)
        
        if not values:
            mean_value = 0.0
        else:
            mean_value = statistics.mean(values)
        
        return MetricResult(
            metric_name=f"mean_{self.field}",
            value=mean_value,
            metadata={
                "field": self.field,
                "count": len(values),
                "total_results": len(evaluation_results)
            }
        )


class MedianMetric(Metric):
    """Median metric."""
    
    def __init__(self, field: str = "score", **params):
        super().__init__("median", **params)
        self.field = field
    
    def calculate(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> MetricResult:
        """Calculate median of specified field."""
        values = []
        for result in evaluation_results:
            if self.field in result:
                value = result[self.field]
                if isinstance(value, (int, float)):
                    values.append(value)
        
        if not values:
            median_value = 0.0
        else:
            median_value = statistics.median(values)
        
        return MetricResult(
            metric_name=f"median_{self.field}",
            value=median_value,
            metadata={
                "field": self.field,
                "count": len(values),
                "total_results": len(evaluation_results)
            }
        )


class PercentileMetric(Metric):
    """Percentile metric."""
    
    def __init__(self, percentile: float = 50.0, field: str = "score", **params):
        super().__init__(f"p{percentile}", **params)
        self.percentile = percentile
        self.field = field
    
    def calculate(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> MetricResult:
        """Calculate percentile of specified field."""
        values = []
        for result in evaluation_results:
            if self.field in result:
                value = result[self.field]
                if isinstance(value, (int, float)):
                    values.append(value)
        
        if not values:
            percentile_value = 0.0
        else:
            values.sort()
            n = len(values)
            index = (self.percentile / 100.0) * (n - 1)
            
            if index.is_integer():
                percentile_value = values[int(index)]
            else:
                lower_index = int(math.floor(index))
                upper_index = int(math.ceil(index))
                weight = index - lower_index
                percentile_value = values[lower_index] * (1 - weight) + values[upper_index] * weight
        
        return MetricResult(
            metric_name=f"p{self.percentile}_{self.field}",
            value=percentile_value,
            metadata={
                "percentile": self.percentile,
                "field": self.field,
                "count": len(values),
                "total_results": len(evaluation_results)
            }
        )


class PassRateMetric(Metric):
    """Pass rate metric."""
    
    def __init__(self, **params):
        super().__init__("pass_rate", **params)
    
    def calculate(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> MetricResult:
        """Calculate pass rate."""
        passed_count = sum(1 for result in evaluation_results if result.get("passed", False))
        total_count = len(evaluation_results)
        
        pass_rate = passed_count / total_count if total_count > 0 else 0.0
        
        return MetricResult(
            metric_name="pass_rate",
            value=pass_rate,
            metadata={
                "passed_count": passed_count,
                "total_count": total_count
            }
        )


class CountMetric(Metric):
    """Count metric."""
    
    def __init__(self, field: Optional[str] = None, value: Optional[Any] = None, **params):
        super().__init__("count", **params)
        self.field = field
        self.value = value
    
    def calculate(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> MetricResult:
        """Count total results or results matching criteria."""
        if self.field is None:
            count = len(evaluation_results)
            name = "total_count"
        else:
            if self.value is None:
                count = sum(1 for result in evaluation_results if self.field in result)
                name = f"count_with_{self.field}"
            else:
                count = sum(1 for result in evaluation_results 
                           if result.get(self.field) == self.value)
                name = f"count_{self.field}_{self.value}"
        
        return MetricResult(
            metric_name=name,
            value=count,
            metadata={
                "field": self.field,
                "value": self.value,
                "total_results": len(evaluation_results)
            }
        )


class StandardDeviationMetric(Metric):
    """Standard deviation metric."""
    
    def __init__(self, field: str = "score", **params):
        super().__init__("std", **params)
        self.field = field
    
    def calculate(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> MetricResult:
        """Calculate standard deviation of specified field."""
        values = []
        for result in evaluation_results:
            if self.field in result:
                value = result[self.field]
                if isinstance(value, (int, float)):
                    values.append(value)
        
        if len(values) <= 1:
            std_value = 0.0
        else:
            std_value = statistics.stdev(values)
        
        return MetricResult(
            metric_name=f"std_{self.field}",
            value=std_value,
            metadata={
                "field": self.field,
                "count": len(values),
                "total_results": len(evaluation_results)
            }
        )


class MinMetric(Metric):
    """Minimum value metric."""
    
    def __init__(self, field: str = "score", **params):
        super().__init__("min", **params)
        self.field = field
    
    def calculate(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> MetricResult:
        """Calculate minimum of specified field."""
        values = []
        for result in evaluation_results:
            if self.field in result:
                value = result[self.field]
                if isinstance(value, (int, float)):
                    values.append(value)
        
        if not values:
            min_value = 0.0
        else:
            min_value = min(values)
        
        return MetricResult(
            metric_name=f"min_{self.field}",
            value=min_value,
            metadata={
                "field": self.field,
                "count": len(values),
                "total_results": len(evaluation_results)
            }
        )


class MaxMetric(Metric):
    """Maximum value metric."""
    
    def __init__(self, field: str = "score", **params):
        super().__init__("max", **params)
        self.field = field
    
    def calculate(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> MetricResult:
        """Calculate maximum of specified field."""
        values = []
        for result in evaluation_results:
            if self.field in result:
                value = result[self.field]
                if isinstance(value, (int, float)):
                    values.append(value)
        
        if not values:
            max_value = 0.0
        else:
            max_value = max(values)
        
        return MetricResult(
            metric_name=f"max_{self.field}",
            value=max_value,
            metadata={
                "field": self.field,
                "count": len(values),
                "total_results": len(evaluation_results)
            }
        )


class SumMetric(Metric):
    """Sum metric."""
    
    def __init__(self, field: str = "score", **params):
        super().__init__("sum", **params)
        self.field = field
    
    def calculate(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> MetricResult:
        """Calculate sum of specified field."""
        values = []
        for result in evaluation_results:
            if self.field in result:
                value = result[self.field]
                if isinstance(value, (int, float)):
                    values.append(value)
        
        sum_value = sum(values) if values else 0.0
        
        return MetricResult(
            metric_name=f"sum_{self.field}",
            value=sum_value,
            metadata={
                "field": self.field,
                "count": len(values),
                "total_results": len(evaluation_results)
            }
        )


def _load_function_from_file(file_path: str, function_name: str) -> Callable:
    """Load a function from a Python file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    spec = importlib.util.spec_from_file_location("custom_metric", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in {file_path}")
    
    return getattr(module, function_name)
