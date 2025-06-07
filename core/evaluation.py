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


@dataclass
class EvaluationResult:
    """Result from evaluation method."""
    method_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "passed": self.passed,
            "score": self.score,
            "details": self.details
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
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a response against ground truth.
        
        Args:
            response: The model's response
            ground_truth: Ground truth data
            **kwargs: Additional context or parameters
            
        Returns:
            EvaluationResult containing assessment
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
        **kwargs
    ) -> EvaluationResult:
        """Evaluate using custom function."""
        try:
            # Call the custom function
            result = self.function(
                response=response, 
                ground_truth=ground_truth, 
                **self.params,
                **kwargs
            )
            
            # Normalize result format
            if isinstance(result, dict):
                passed = result.get('passed', False)
                score = result.get('score', 0.0)
                details = {k: v for k, v in result.items() if k not in ['passed', 'score']}
            elif isinstance(result, (int, float)):
                passed = result > 0
                score = float(result)
                details = {}
            elif isinstance(result, bool):
                passed = result
                score = 1.0 if result else 0.0
                details = {}
            else:
                raise ValueError(f"Invalid result type from evaluation function: {type(result)}")
            
            return EvaluationResult(
                method_name=self.name,
                passed=passed,
                score=score,
                details=details
            )
            
        except Exception as e:
            return EvaluationResult(
                method_name=self.name,
                passed=False,
                score=0.0,
                details={"error": str(e)}
            )


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
                 **params):
        super().__init__("exact_match", **params)
        self.response_field = response_field
        self.ground_truth_field = ground_truth_field
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace
    
    def evaluate(
        self, 
        response: str, 
        ground_truth: Dict[str, Any], 
        **kwargs
    ) -> EvaluationResult:
        """Check exact match between response and expected output."""
        expected = ground_truth.get(self.ground_truth_field, "")
        
        # Process strings
        actual = response
        if self.strip_whitespace:
            actual = actual.strip()
            expected = str(expected).strip()
        
        if not self.case_sensitive:
            actual = actual.lower()
            expected = expected.lower()
        
        passed = actual == expected
        
        return EvaluationResult(
            method_name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            details={
                "expected": expected,
                "actual": actual,
                "case_sensitive": self.case_sensitive,
                "strip_whitespace": self.strip_whitespace
            }
        )


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
        **kwargs
    ) -> EvaluationResult:
        """Check if response contains expected content."""
        expected = self.expected_content or ground_truth.get(self.ground_truth_field, "")
        
        actual = response
        if not self.case_sensitive:
            actual = actual.lower()
            expected = str(expected).lower()
        
        passed = expected in actual
        
        return EvaluationResult(
            method_name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            details={
                "expected_content": expected,
                "case_sensitive": self.case_sensitive,
                "found": passed
            }
        )


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
        **kwargs
    ) -> EvaluationResult:
        """Check if response matches regex pattern."""
        pattern = self.pattern or ground_truth.get(self.ground_truth_field, "")
        
        try:
            match = re.search(pattern, response, self.flags)
            passed = match is not None
            
            details = {
                "pattern": pattern,
                "matched": passed
            }
            
            if match:
                details["match_groups"] = match.groups()
                details["match_span"] = match.span()
            
            return EvaluationResult(
                method_name=self.name,
                passed=passed,
                score=1.0 if passed else 0.0,
                details=details
            )
            
        except re.error as e:
            return EvaluationResult(
                method_name=self.name,
                passed=False,
                score=0.0,
                details={"error": f"Invalid regex pattern: {e}"}
            )


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
        **kwargs
    ) -> EvaluationResult:
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
                
                return EvaluationResult(
                    method_name=self.name,
                    passed=passed,
                    score=1.0 if passed else 0.0,
                    details={
                        "is_valid_json": True,
                        "expected_keys": list(expected_keys),
                        "actual_keys": list(response_keys),
                        "missing_keys": list(missing_keys),
                        "extra_keys": list(extra_keys),
                        "strict_mode": self.strict
                    }
                )
            else:
                return EvaluationResult(
                    method_name=self.name,
                    passed=False,
                    score=0.0,
                    details={
                        "is_valid_json": True,
                        "error": "Response is not a JSON object"
                    }
                )
                
        except json.JSONDecodeError as e:
            return EvaluationResult(
                method_name=self.name,
                passed=False,
                score=0.0,
                details={
                    "is_valid_json": False,
                    "error": f"Invalid JSON: {e}"
                }
            )


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
        **kwargs
    ) -> EvaluationResult:
        """Check if response contains expected numeric value."""
        expected = self.expected_value
        if expected is None:
            expected = ground_truth.get(self.ground_truth_field)
        
        if expected is None:
            return EvaluationResult(
                method_name=self.name,
                passed=False,
                score=0.0,
                details={"error": "No expected value provided"}
            )
        
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
            
            return EvaluationResult(
                method_name=self.name,
                passed=passed,
                score=score,
                details={
                    "expected": expected,
                    "actual": actual,
                    "difference": abs(actual - expected),
                    "tolerance": self.tolerance
                }
            )
            
        except (ValueError, TypeError) as e:
            return EvaluationResult(
                method_name=self.name,
                passed=False,
                score=0.0,
                details={"error": f"Could not extract numeric value: {e}"}
            )


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
        **kwargs
    ) -> EvaluationResult:
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
        details = {
            "length": length,
            "unit": self.unit
        }
        
        if self.min_length is not None:
            if length < self.min_length:
                passed = False
            details["min_length"] = self.min_length
            details["meets_min"] = length >= self.min_length
        
        if self.max_length is not None:
            if length > self.max_length:
                passed = False
            details["max_length"] = self.max_length
            details["meets_max"] = length <= self.max_length
        
        if self.expected_length is not None:
            if length != self.expected_length:
                passed = False
            details["expected_length"] = self.expected_length
            details["matches_expected"] = length == self.expected_length
        
        return EvaluationResult(
            method_name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            details=details
        )


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
