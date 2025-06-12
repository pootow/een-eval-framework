"""
Evaluation methods for the evaluation framework.

This module provides both built-in and custom evaluation methods
for assessing model responses against ground truth.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Callable, Optional
import importlib.util
import inspect
import re
import json
from pathlib import Path

from ..workflow.config import EvaluationMethodConfig

@dataclass
class EvaluationLabelConfig:
    """Configuration for a single label in evaluation."""

    name: str
    description: Optional[str] = None

@dataclass
class EvaluationLabelResult:
    """Result for a single label in evaluation."""

    passed: bool
    score: float
    custom_fields: Dict[str, Any]


@dataclass
class EvaluationLabel:
    """Result for a single label in evaluation."""
    
    label: EvaluationLabelConfig
    result: EvaluationLabelResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": {
                "name": self.label.name,
                "description": self.label.description
            },
            "result": {
                "passed": self.result.passed,
                "score": self.result.score,
                "custom_fields": self.result.custom_fields
            }
        }


@dataclass
class EvaluationResult:
    """Result from evaluation method."""

    labels: list[EvaluationLabel]
    custom_fields: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "labels": [label.to_dict() for label in self.labels],
            "custom_fields": self.custom_fields
        }


class EvaluationMethod(ABC):
    """Abstract base class for evaluation methods."""
    
    def __init__(self, name: str, **params):
        self.name = name
        self.params = params
    
    @abstractmethod
    def evaluate(
        self, 
        response: str, 
        ground_truth: Dict[str, Any], 
        inference_result: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a response against ground truth.
        
        Args:
            response: The model's response
            ground_truth: Ground truth data
            inference_result: Full inference result for context
            **kwargs: Additional context or parameters
            
        Returns:
            Dictionary containing labels and evaluation results in the format:
            {
                "labels": [
                    {
                        "label": {"name": "label_name"},
                        "result": {
                            "passed": bool,
                            "score": float,
                            "custom_fields": { ... },
                            ...
                        }
                    },
                    ...
                ],
                "custom_fields": { ... },
                ...
            }
        """
        pass
    
    @classmethod
    def from_function(
        cls, 
        name: str, 
        function: Callable,
        **params
    ) -> "EvaluationMethod":
        """Create evaluation method from function."""
        return CustomEvaluationMethod(name, function, **params)

    @classmethod
    def from_file(
        cls, 
        name: str, 
        file_path: str, 
        function_name: Optional[str] = None,
        **params
    ) -> "EvaluationMethod":
        """Create evaluation method from external file."""
        function = _load_function_from_file(file_path, function_name or name)
        return cls.from_function(name, function, **params)
    
    @classmethod
    def from_config(cls, config: EvaluationMethodConfig) -> "EvaluationMethod":
        """Create evaluation method from configuration object."""
        if config.type == "built_in":
            return BuiltInEvaluationMethod.create(config.name, **config.params)
        elif config.type == "custom":
            if config.path:
                # File-based custom evaluation method
                return cls.from_file(config.name, config.path, config.function_name, **config.params)
            elif hasattr(config, 'module') and hasattr(config, 'function_name') and config.module and config.function_name:
                # Module-based custom evaluation method  
                try:
                    import importlib
                    module = importlib.import_module(config.module)
                    function = getattr(module, config.function_name)
                    return cls.from_function(config.name, function, **config.params)
                except (ImportError, AttributeError) as e:
                    raise ValueError(f"Could not load function {config.function_name} from module {config.module}: {e}")
            else:
                raise ValueError("Custom evaluation method requires either 'path' or 'module'+'function_name'")
        else:
            raise ValueError(f"Unknown evaluation method type: {config.type}")


class CustomEvaluationMethod(EvaluationMethod):
    """Custom evaluation method using user-provided function."""
    
    def __init__(self, name: str, function: Callable, **params):
        super().__init__(name, **params)
        self.function = function
        
        # Validate function signature
        sig = inspect.signature(function)
        required_params = ['response', 'ground_truth']
        for param in required_params:
            if param not in sig.parameters:
                raise ValueError(f"Function {function.__name__} must have '{param}' parameter")
    
    def evaluate(
        self, 
        response: str, 
        ground_truth: Dict[str, Any], 
        inference_result: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Evaluate using custom function."""
        try:
            # Call the custom function
            result = self.function(
                response=response, 
                ground_truth=ground_truth, 
                inference_result=inference_result or {},
                **self.params,
                **kwargs
            )
            
            # Normalize result format to match new specification
            if isinstance(result, dict):
                # If result already has the new format with labels
                if "labels" in result:
                    return result
                
                # Convert old format to new format
                passed = result.get('passed', False)
                score = result.get('score', 0.0)
                details = {k: v for k, v in result.items() if k not in ['passed', 'score']}
                
                return {
                    "labels": [
                        {
                            "label": {"name": self.name},
                            "result": {
                                "passed": passed,
                                "score": score,
                                **details
                            }
                        }
                    ]
                }
            elif isinstance(result, (int, float)):
                passed = result > 0
                score = float(result)
                return {
                    "labels": [
                        {
                            "label": {"name": self.name},
                            "result": {
                                "passed": passed,
                                "score": score
                            }
                        }
                    ]
                }
            elif isinstance(result, bool):
                passed = result
                score = 1.0 if result else 0.0
                return {
                    "labels": [
                        {
                            "label": {"name": self.name},
                            "result": {
                                "passed": passed,
                                "score": score
                            }
                        }
                    ]
                }
            else:
                raise ValueError(f"Invalid result type from evaluation function: {type(result)}")
            
        except Exception as e:
            return {
                "labels": [
                    {
                        "label": {"name": self.name},
                        "result": {
                            "passed": False,
                            "score": 0.0,
                            "error": str(e)
                        }
                    }
                ]
            }


class BuiltInEvaluationMethod(EvaluationMethod):
    """Built-in evaluation methods."""
    
    @classmethod
    def create(cls, method_name: str, **params) -> EvaluationMethod:
        """Create built-in evaluation method."""
        method_map = {
            "exact_match": ExactMatchEvaluation,
            "contains": ContainsEvaluation,
            "regex_match": RegexMatchEvaluation,
            "json_match": JsonMatchEvaluation,
            "numeric_match": NumericMatchEvaluation,
            "length_check": LengthCheckEvaluation
        }
        
        if method_name not in method_map:
            raise ValueError(f"Unknown built-in evaluation method: {method_name}")
        
        return method_map[method_name](**params)


class ExactMatchEvaluation(EvaluationMethod):
    """Exact string match evaluation."""
    
    def __init__(self, 
                 response_field: str = "response", 
                 ground_truth_field: str = "expected_output",
                 case_sensitive: bool = True,
                 strip_whitespace: bool = True,
                 expected_content: Optional[str] = None,
                 **params):
        super().__init__("exact_match", **params)
        self.response_field = response_field
        self.ground_truth_field = ground_truth_field
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace
        self.expected_content = expected_content
    
    def evaluate(
        self, 
        response: str, 
        ground_truth: Dict[str, Any], 
        inference_result: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform exact match evaluation."""
        # Get expected value
        if self.expected_content is not None:
            expected = self.expected_content
        else:
            expected = ground_truth.get(self.ground_truth_field, "")
        
        # Get actual response
        actual = response
        
        # Normalize strings
        if self.strip_whitespace:
            expected = str(expected).strip()
            actual = actual.strip()
        
        if not self.case_sensitive:
            expected = expected.lower()
            actual = actual.lower()
        
        # Compare
        match = expected == actual
        
        return {
            "labels": [
                {
                    "label": {"name": "exact_match"},
                    "result": {
                        "passed": match,
                        "score": 1.0 if match else 0.0,
                        "expected": expected,
                        "actual": actual
                    }
                }
            ]
        }


class ContainsEvaluation(EvaluationMethod):
    """Check if response contains expected content."""
    
    def __init__(self, 
                 expected_content: Optional[str] = None,
                 ground_truth_field: str = "expected_content",
                 case_sensitive: bool = True,
                 **params):
        super().__init__("contains", **params)
        self.expected_content = expected_content
        self.ground_truth_field = ground_truth_field
        self.case_sensitive = case_sensitive
    
    def evaluate(
        self, 
        response: str, 
        ground_truth: Dict[str, Any], 
        inference_result: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Check if response contains expected content."""
        expected = self.expected_content or ground_truth.get(self.ground_truth_field, "")
        
        actual = response
        if not self.case_sensitive:
            actual = actual.lower()
            expected = str(expected).lower()
        
        passed = expected in actual
        
        return {
            "labels": [
                {
                    "label": {"name": "contains"},
                    "result": {
                        "passed": passed,
                        "score": 1.0 if passed else 0.0,
                        "expected_content": expected,
                        "case_sensitive": self.case_sensitive,
                        "found": passed
                    }
                }
            ]
        }


class RegexMatchEvaluation(EvaluationMethod):
    """Regular expression match evaluation."""
    
    def __init__(self, 
                 pattern: Optional[str] = None,
                 ground_truth_field: str = "pattern",
                 flags: int = 0,
                 **params):
        super().__init__("regex_match", **params)
        self.pattern = pattern
        self.ground_truth_field = ground_truth_field
        self.flags = flags
    
    def evaluate(
        self, 
        response: str, 
        ground_truth: Dict[str, Any], 
        inference_result: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Check if response matches regex pattern."""
        pattern = self.pattern or ground_truth.get(self.ground_truth_field, "")
        
        try:
            match = re.search(pattern, response, self.flags)
            passed = match is not None
            
            result_details = {
                "passed": passed,
                "score": 1.0 if passed else 0.0,
                "pattern": pattern,
                "matched": passed
            }
            
            if match:
                result_details["match_groups"] = match.groups()
                result_details["match_span"] = match.span()
            
            return {
                "labels": [
                    {
                        "label": {"name": "regex_match"},
                        "result": result_details
                    }
                ]
            }
            
        except re.error as e:
            return {
                "labels": [
                    {
                        "label": {"name": "regex_match"},
                        "result": {
                            "passed": False,
                            "score": 0.0,
                            "error": f"Invalid regex pattern: {e}"
                        }
                    }
                ]
            }


class JsonMatchEvaluation(EvaluationMethod):
    """JSON structure match evaluation."""
    
    def __init__(self, 
                 expected_keys: Optional[list] = None,
                 ground_truth_field: str = "expected_json",
                 strict: bool = False,
                 **params):
        super().__init__("json_match", **params)
        self.expected_keys = expected_keys
        self.ground_truth_field = ground_truth_field
        self.strict = strict
    
    def evaluate(
        self, 
        response: str, 
        ground_truth: Dict[str, Any], 
        inference_result: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Check if response is valid JSON with expected structure."""
        try:
            # Parse response as JSON
            parsed_response = json.loads(response)
            
            if self.expected_keys:
                expected_keys = set(self.expected_keys)
            else:
                expected_json = ground_truth.get(self.ground_truth_field, {})
                if isinstance(expected_json, dict):
                    expected_keys = set(expected_json.keys())
                else:
                    expected_keys = set()
            
            if isinstance(parsed_response, dict):
                response_keys = set(parsed_response.keys())
                
                if self.strict:
                    passed = response_keys == expected_keys
                    missing_keys = expected_keys - response_keys
                    extra_keys = response_keys - expected_keys
                else:
                    passed = expected_keys.issubset(response_keys)
                    missing_keys = expected_keys - response_keys
                    extra_keys = set()
                
                return {
                    "labels": [
                        {
                            "label": {"name": "json_match"},
                            "result": {
                                "passed": passed,
                                "score": 1.0 if passed else 0.0,
                                "is_valid_json": True,
                                "expected_keys": list(expected_keys),
                                "actual_keys": list(response_keys),
                                "missing_keys": list(missing_keys),
                                "extra_keys": list(extra_keys),
                                "strict_mode": self.strict
                            }
                        }
                    ]
                }
            else:
                return {
                    "labels": [
                        {
                            "label": {"name": "json_match"},
                            "result": {
                                "passed": False,
                                "score": 0.0,
                                "is_valid_json": True,
                                "error": "Response is not a JSON object"
                            }
                        }
                    ]
                }
                
        except json.JSONDecodeError as e:
            return {
                "labels": [
                    {
                        "label": {"name": "json_match"},
                        "result": {
                            "passed": False,
                            "score": 0.0,
                            "is_valid_json": False,
                            "error": f"Invalid JSON: {e}"
                        }
                    }
                ]
            }


class NumericMatchEvaluation(EvaluationMethod):
    """Numeric value match evaluation."""
    
    def __init__(self, 
                 expected_value: Optional[float] = None,
                 ground_truth_field: str = "expected_value",
                 tolerance: float = 1e-6,
                 extract_pattern: Optional[str] = None,
                 **params):
        super().__init__("numeric_match", **params)
        self.expected_value = expected_value
        self.ground_truth_field = ground_truth_field
        self.tolerance = tolerance
        self.extract_pattern = extract_pattern
    
    def evaluate(
        self, 
        response: str, 
        ground_truth: Dict[str, Any], 
        inference_result: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Check if response contains expected numeric value."""
        expected = self.expected_value
        if expected is None:
            expected = ground_truth.get(self.ground_truth_field)
        
        if expected is None:
            return {
                "labels": [
                    {
                        "label": {"name": "numeric_match"},
                        "result": {
                            "passed": False,
                            "score": 0.0,
                            "error": "No expected value provided"
                        }
                    }
                ]
            }
        
        try:
            # Extract numeric value from response
            if self.extract_pattern:
                match = re.search(self.extract_pattern, response)
                if match:
                    actual = float(match.group(1))
                else:
                    raise ValueError("Pattern not found in response")
            else:
                # Try to extract any numeric value
                numbers = re.findall(r'-?\d+\.?\d*', response)
                if numbers:
                    actual = float(numbers[0])
                else:
                    raise ValueError("No numeric value found in response")
            
            # Compare with tolerance
            passed = abs(actual - expected) <= self.tolerance
            score = 1.0 if passed else max(0.0, 1.0 - abs(actual - expected) / max(abs(expected), 1.0))
            
            return {
                "labels": [
                    {
                        "label": {"name": "numeric_match"},
                        "result": {
                            "passed": passed,
                            "score": score,
                            "expected": expected,
                            "actual": actual,
                            "difference": abs(actual - expected),
                            "tolerance": self.tolerance
                        }
                    }
                ]
            }
            
        except (ValueError, TypeError) as e:
            return {
                "labels": [
                    {
                        "label": {"name": "numeric_match"},
                        "result": {
                            "passed": False,
                            "score": 0.0,
                            "error": f"Could not extract numeric value: {e}"
                        }
                    }
                ]
            }


class LengthCheckEvaluation(EvaluationMethod):
    """Response length check evaluation."""
    
    def __init__(self, 
                 min_length: Optional[int] = None,
                 max_length: Optional[int] = None,
                 expected_length: Optional[int] = None,
                 unit: str = "characters",  # "characters", "words", "lines"
                 **params):
        super().__init__("length_check", **params)
        self.min_length = min_length
        self.max_length = max_length
        self.expected_length = expected_length
        self.unit = unit
    
    def evaluate(
        self, 
        response: str, 
        ground_truth: Dict[str, Any], 
        inference_result: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Check if response meets length requirements."""
        if self.unit == "characters":
            length = len(response)
        elif self.unit == "words":
            length = len(response.split())
        elif self.unit == "lines":
            length = len(response.splitlines())
        else:
            raise ValueError(f"Unknown unit: {self.unit}")
        
        passed = True
        result_details = {
            "length": length,
            "unit": self.unit
        }
        
        if self.min_length is not None:
            if length < self.min_length:
                passed = False
            result_details["min_length"] = self.min_length
            result_details["meets_min"] = length >= self.min_length
        
        if self.max_length is not None:
            if length > self.max_length:
                passed = False
            result_details["max_length"] = self.max_length
            result_details["meets_max"] = length <= self.max_length
        
        if self.expected_length is not None:
            if length != self.expected_length:
                passed = False
            result_details["expected_length"] = self.expected_length
            result_details["matches_expected"] = length == self.expected_length
        
        result_details["passed"] = passed
        result_details["score"] = 1.0 if passed else 0.0
        
        return {
            "labels": [
                {
                    "label": {"name": "length_check"},
                    "result": result_details
                }
            ]
        }


def _load_function_from_file(file_path: str, function_name: str) -> Callable:
    """Load a function from a Python file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    spec = importlib.util.spec_from_file_location("custom_eval", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, function_name):
        raise AttributeError(f"Function '{function_name}' not found in {file_path}")
    
    return getattr(module, function_name)
