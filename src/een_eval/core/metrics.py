"""
Metrics calculation for the evaluation framework.

This module provides both built-in and custom metrics for aggregating
evaluation results across multiple samples with support for faceting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Callable, Optional, Union
import statistics
import math
import importlib.util
from pathlib import Path
from collections import defaultdict

from ..workflow.config import MetricConfig


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
    
    def __init__(self, name: str, facets: Optional[List[str]] = None, **params):
        self.name = name
        self.facets = facets or []
        self.params = params
    
    def calculate(
        self, 
        evaluation_results: List[Dict[str, Any]],
        facets: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Calculate metric from evaluation results with automatic facets support.
        
        Args:
            evaluation_results: List of evaluation results from all samples
            facets: List of fields to group by for metric calculation
            **kwargs: Additional context or parameters
            
        Returns:
            List of metric records for different facet combinations
        """
        used_facets = facets or self.facets
        
        # Always include label as an implicit facet for grouping
        # Group by facets and calculate for each group
        grouped = self._group_by_facets(evaluation_results, used_facets)

        results = []
        for facet_key, all_results in grouped.items():
            if not all_results:
                continue
                
            # Calculate metric for this facet group
            group_result = self._calculate_single(all_results, **kwargs)
            # Parse facet values and create result dict
            facet_dict: Dict[str, Any] = {"facets": used_facets}
            facet_values = facet_key.split('+')
            for i, facet in enumerate(used_facets):
                if i < len(facet_values) - 1:  # Exclude label facet
                    facet_dict[facet] = facet_values[i]

            # Add label as separate field
            if len(facet_values) > len(used_facets):
                facet_dict["label"] = facet_values[-1]
            
            # Merge group result with facet information
            if isinstance(group_result, dict):
                final_result = {**facet_dict, **group_result, "metric_name": self.name}
            else:
                final_result = {**facet_dict, "value": group_result, "metric_name": self.name}
                
            results.append(final_result)
        
        return results
    
    @abstractmethod
    def _calculate_single(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> Union[Dict[str, Any], float, int]:
        """
        Calculate metric for a single group of evaluation results.
        
        This method should be implemented by subclasses to perform the actual
        metric calculation on a homogeneous group of results (same facet values).
        
        Args:
            evaluation_results: List of evaluation results from the same facet group
            **kwargs: Additional context or parameters
            
        Returns:
            Either a dict with metric fields or a single numeric value
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
            return BuiltInMetric.create(config.name, facets=getattr(config, 'facets', []), **config.params)
        elif config.type == "custom":
            if config.path:
                # File-based custom metric
                return cls.from_file(config.name, config.path, config.function_name, 
                                   facets=getattr(config, 'facets', []), **config.params)
            elif hasattr(config, 'module') and hasattr(config, 'function_name') and config.module and config.function_name:
                # Module-based custom metric
                try:
                    import importlib
                    module = importlib.import_module(config.module)
                    function = getattr(module, config.function_name)
                    return cls.from_function(config.name, function, 
                                           facets=getattr(config, 'facets', []), **config.params)
                except (ImportError, AttributeError) as e:
                    raise ValueError(f"Could not load function {config.function_name} from module {config.module}: {e}")
            else:
                raise ValueError("Custom metric requires either 'path' or 'module'+'function_name'")
        else:
            raise ValueError(f"Unknown metric type: {config.type}")
    def _group_by_facets(
        self, 
        evaluation_results: List[Dict[str, Any]], 
        facets: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group evaluation results by facets and label.
        
        Returns:
            Dict with facet combinations as keys, each containing flat list of results
        """
        grouped = defaultdict(list)
        
        for result in evaluation_results:
            # Build facet key
            facet_values = []
            for facet in facets:
                if '.' in facet:
                    # Handle nested fields like metadata.model_id
                    parts = facet.split('.')
                    value = result
                    for part in parts:
                        value = value.get(part, {}) if isinstance(value, dict) else {}
                    facet_values.append(str(value) if value else "unknown")
                else:
                    facet_values.append(str(result.get(facet, "unknown")))
            
            # Also add label as a facet
            facet_values.append(result.get("label", ""))
            facet_key = "+".join(facet_values)

            # Add result to the flat list for this facet combination
            grouped[facet_key].append(result)
        
        return dict(grouped)


class CustomMetric(Metric):
    """Custom metric using user-provided function."""
    
    def __init__(self, name: str, function: Callable, facets: Optional[List[str]] = None, **params):
        super().__init__(name, facets, **params)
        self.function = function
    
    def _calculate_single(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> Union[Dict[str, Any], float, int]:
        """Calculate using custom function."""
        try:
            # Call the custom function - note: we pass empty facets since grouping is handled by base class
            result = self.function(
                evaluation_results=evaluation_results,
                facets=[],  # Custom functions don't need to handle facets anymore
                **self.params,
                **kwargs
            )
            
            if isinstance(result, list) and len(result) > 0:
                # Return the first result, framework handles facets
                return result[0]
            elif isinstance(result, dict):
                return result
            elif isinstance(result, MetricResult):
                return result.to_dict()
            else:
                return {"value": result}
        except Exception as e:
            return {"value": 0.0, "error": str(e)}


class BuiltInMetric(Metric):
    """Built-in metrics."""
    
    @classmethod
    def create(cls, metric_name: str, facets: Optional[List[str]] = None, **params) -> Metric:
        """Create built-in metric."""
        metric_map = {
            "pass_at_k": PassAtKMetric,
            "mean": MeanMetric,
            "median": MedianMetric,
            "percentile": PercentileMetric,
            "pass_rate": PassRateMetric,
            "count": CountMetric
        }
        
        if metric_name not in metric_map:
            raise ValueError(f"Unknown built-in metric: {metric_name}")
        
        return metric_map[metric_name](facets=facets, **params)


class PassAtKMetric(Metric):
    """Pass@K metric calculation."""
    
    def __init__(self, k: int = 1, num_samples: int = 16, aggregation: str = "mean", facets: Optional[List[str]] = None, **params):
        super().__init__("pass_at_k", facets, **params)
        self.k = k
        self.num_samples = num_samples
        self.aggregation = aggregation
    
    def _calculate_single(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate pass@k metric for a single group."""
        # Group by item_id within this facet group
        items_dict = self._group_by_item_id(evaluation_results)
        
        pass_at_k_values = []
        total_samples = 0
        
        for item_id, item_results in items_dict.items():
            # Check if at least k samples passed for this item
            passed_samples = [r for r in item_results if r.get("passed", False)]
            item_passes_at_k = len(passed_samples) >= self.k
            pass_at_k_values.append(1.0 if item_passes_at_k else 0.0)
            total_samples += len(item_results)
        
        if self.aggregation == "mean":
            pass_at_k = statistics.mean(pass_at_k_values) if pass_at_k_values else 0.0
        else:
            pass_at_k = sum(pass_at_k_values) / len(pass_at_k_values) if pass_at_k_values else 0.0
        
        return {
            f"pass_at_{self.k}": pass_at_k,
            "average_sample_count": total_samples / len(items_dict) if items_dict else 0,
            "total_sample_count": total_samples,
            "total_item_count": len(items_dict)
        }
    
    def _group_by_item_id(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group results by item_id only."""
        grouped = defaultdict(list)
        for result in evaluation_results:
            item_id = result.get("item_id", "unknown")
            grouped[item_id].append(result)
        return grouped


class MeanMetric(Metric):
    """Mean score metric."""
    
    def __init__(self, facets: Optional[List[str]] = None, **params):
        super().__init__("mean", facets, **params)
    
    def _calculate_single(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate mean score for a single group."""
        scores = [r.get("score", 0.0) for r in evaluation_results]
        mean_score = statistics.mean(scores) if scores else 0.0
        
        return {
            "mean": mean_score,
            "sample_count": len(scores)
        }


class MedianMetric(Metric):
    """Median score metric."""

    def __init__(self, facets: Optional[List[str]] = None, **params):
        super().__init__("median", facets, **params)
    
    def _calculate_single(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate median score for a single group."""
        scores = [r.get("score", 0.0) for r in evaluation_results]
        median_score = statistics.median(scores) if scores else 0.0
        
        return {
            "median": median_score,
            "sample_count": len(scores)
        }


class PercentileMetric(Metric):
    """Percentile score metric."""
    
    def __init__(self, percentile: float = 95.0, facets: Optional[List[str]] = None, **params):
        super().__init__("percentile", facets, **params)
        self.percentile = percentile
    
    def _calculate_single(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate percentile score for a single group."""
        scores = [r.get("score", 0.0) for r in evaluation_results]
        
        if not scores:
            percentile_score = 0.0
        else:
            scores.sort()
            index = (self.percentile / 100.0) * (len(scores) - 1)
            if index.is_integer():
                percentile_score = scores[int(index)]
            else:
                lower = scores[int(index)]
                upper = scores[int(index) + 1]
                percentile_score = lower + (upper - lower) * (index - int(index))
        
        return {
            f"percentile_{self.percentile}": percentile_score,
            "sample_count": len(scores)
        }


class PassRateMetric(Metric):
    """Pass rate metric."""
    
    def __init__(self, facets: Optional[List[str]] = None, **params):
        super().__init__("pass_rate", facets, **params)
    
    def _calculate_single(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate pass rate for a single group."""
        passed_count = sum(1 for r in evaluation_results if r.get("passed", False))
        total_count = len(evaluation_results)
        pass_rate = passed_count / total_count if total_count > 0 else 0.0
        
        return {
            "pass_rate": pass_rate,
            "passed_count": passed_count,
            "total_count": total_count
        }


class CountMetric(Metric):
    """Count metric."""
    
    def __init__(self, facets: Optional[List[str]] = None, **params):
        super().__init__("count", facets, **params)

    def _calculate_single(
        self, 
        evaluation_results: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate count for a single group."""
        return {
            "count": len(evaluation_results)
        }


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
