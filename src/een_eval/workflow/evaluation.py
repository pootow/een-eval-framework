"""
Evaluation engine for processing inference results and computing metrics.

This module handles the evaluation phase of the workflow, taking inference results
and applying evaluation methods and metrics to compute final scores.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from ..core.evaluation import EvaluationMethod
from ..core.metrics import Metric
from ..core.models import InferenceResult
from ..utils.io import OutputManager


@dataclass 
class FinalEvaluationResult:
    """Final evaluation result following DataFlow.md specification."""
    # Preserved from inference
    item_id: str                         # Original dataset item
    sample_id: str                       # Unique sample identifier
    sample_index: int                    # Sample number within item
    label: str                           # Label name (e.g., "clearness", "correctness")
    model_name: str                      # Model name (just a friendly display name)
    
    # From evaluation method
    passed: bool                         # Whether this sample passed
    score: float                         # Numeric score (optional)
    detailed_results: Dict[str, Any]     # Copy of "custom_fields" from evaluation method
    
    # Metadata
    evaluation_time: float               # Time taken for evaluation
    timestamp: float                     # When evaluation was performed
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional inference + evaluation metadata


class EvaluationEngine:
    """
    Engine for running evaluation on inference results.
    
    Takes inference results and applies evaluation methods and metrics
    to compute final evaluation scores.
    """
    
    def __init__(
        self,
        evaluation_methods: List[EvaluationMethod],
        metrics: List[Metric],
        output_manager: OutputManager
    ):
        """
        Initialize evaluation engine.
        
        Args:
            evaluation_methods: List of evaluation methods to apply
            metrics: List of metrics to compute
            output_manager: Manager for handling output operations
        """
        self.evaluation_methods = evaluation_methods
        self.metrics = metrics
        self.output_manager = output_manager
        self.logger = logging.getLogger(__name__)
    
    def run(self, status) -> Dict[str, Any]:
        """
        Run evaluation on inference results.
        
        Args:
            status: Workflow status object
            
        Returns:
            Dictionary containing evaluation results and metrics
        """
        self.logger.info("Starting evaluation phase")
        
        # Load inference results
        inference_results = self._load_inference_results()
        if not inference_results:
            raise ValueError("No inference results found. Run inference first.")
        
        # Apply evaluation methods
        evaluation_results = self._apply_evaluation_methods(inference_results, status)
        
        # Compute metrics
        metrics_results = self._compute_metrics(evaluation_results, status)
        
        # Prepare final results
        final_results = {
            "evaluation_summary": {
                "total_samples": len(inference_results),
                "evaluation_methods": [method.name for method in self.evaluation_methods],
                "metrics": [metric.name for metric in self.metrics]
            },
            "detailed_results": evaluation_results,
            "metrics": metrics_results
        }
        
        # Save results
        self._save_results(final_results)
        
        self.logger.info("Evaluation completed successfully")
        return final_results

    def _load_inference_results(self) -> List[Dict[str, Any]]:
        """Load inference results from output directory."""
        # Use the OutputManager's load_responses method which reads from responses.jsonl
        results = self.output_manager.load_responses()
        
        if not results:
            self.logger.warning("No inference results found in responses file")
            return []
        
        self.logger.info(f"Loaded {len(results)} inference results")
        return results

    def _apply_evaluation_methods(
        self, 
        inference_results: List[Dict[str, Any]], 
        status
    ) -> List[Dict[str, Any]]:
        """Apply evaluation methods to inference results."""
        evaluation_results = []
        
        for idx, result in enumerate(inference_results):
            try:
                # Update status
                status.processed_samples = idx + 1
                if idx % 100 == 0:
                    self.logger.info(f"Processed {idx}/{len(inference_results)} samples")
                
                # Apply each evaluation method
                for method in self.evaluation_methods:
                    try:
                        # Extract response and ground truth from the result
                        response = result.get("response", "")
                        # TODO: maybe 1. passing dataset on the consturctor, and get ground truth from there
                        #       2. or config a ground truth retriever function, and retrieve by item_id
                        ground_truth = None
                          # Apply evaluation method with timing
                        start_time = time.time()
                        evaluation_result = method.evaluate(
                            response=response, 
                            ground_truth=ground_truth, 
                            inference_result=result
                        )
                        evaluation_time = time.time() - start_time
                        
                        # Process the labels from the evaluation result
                        if "labels" in evaluation_result:
                            for label_info in evaluation_result["labels"]:
                                label_name = label_info["label"]["name"]
                                label_result = label_info["result"]
                                
                                # Create evaluation result record following DataFlow.md spec TODO: make this strong typed
                                eval_record = {
                                    # Preserved from inference
                                    "item_id": result.get("item_id"),
                                    "sample_id": result.get("sample_id"),
                                    "sample_index": result.get("sample_index"),
                                    "label": label_name,
                                    "model_name": result.get("model_name"),
                                    
                                    # From evaluation method
                                    "passed": label_result.get("passed", False),
                                    "score": label_result.get("score", 0.0),
                                    "detailed_results": {k: v for k, v in label_result.items() 
                                                       if k not in ["passed", "score"]},

                                    # Metadata
                                    "evaluation_time": evaluation_time,
                                    "timestamp": time.time(),
                                    "metadata": result.get("metadata", {})
                                }
                                
                                evaluation_results.append(eval_record)
                        
                    except Exception as e:
                        self.logger.warning(f"Error applying method {method.name} to sample {idx}: {e}")
                        # Create error record
                        error_record = {
                            "item_id": result.get("item_id"),
                            "sample_id": result.get("sample_id"),
                            "sample_index": result.get("sample_index"),
                            "label": method.name,
                            "model_name": result.get("model_name"),
                            "passed": False,
                            "score": 0.0,
                            "detailed_results": {"error": str(e)},
                            "evaluation_time": 0.0,
                            "timestamp": result.get("timestamp"),
                            "metadata": result.get("metadata", {})
                        }
                        evaluation_results.append(error_record)
                
            except Exception as e:
                self.logger.error(f"Error processing sample {idx}, item: {result.get('item_id')}, sample: {result.get('sample_id')}: {e}")
                continue

        return evaluation_results

    def _compute_metrics(
        self, 
        evaluation_results: List[Dict[str, Any]], 
        status
    ) -> List[Dict[str, Any]]:
        """Compute metrics on evaluation results."""
        all_metrics_results = []
        
        for metric in self.metrics:
            try:
                # Calculate metric with facet support
                metric_results = metric.calculate(evaluation_results)
                
                # Handle different return types
                if isinstance(metric_results, list):
                    # Multiple results (faceted metrics)
                    for result in metric_results:
                        if isinstance(result, dict):
                            # Add metric name if not present
                            if "metric_name" not in result:
                                result["metric_name"] = metric.name
                            all_metrics_results.append(result)
                else:
                    # Single result
                    if hasattr(metric_results, 'to_dict'):
                        all_metrics_results.append(metric_results.to_dict())
                    else:
                        all_metrics_results.append({
                            "metric_name": metric.name,
                            "value": metric_results
                        })
                        
            except Exception as e:
                self.logger.warning(f"Error computing metric {metric.name}: {e}")
                all_metrics_results.append({
                    "metric_name": metric.name,
                    "value": None,
                    "error": str(e)
                })
        
        return all_metrics_results

    def _save_results(self, final_results: Dict[str, Any]) -> None:
        """Save evaluation results to output files."""
        # Save evaluation results
        self.output_manager.save_evaluation_results(final_results["detailed_results"])
        
        # Save metrics results
        self.output_manager.save_metrics(final_results["metrics"])
        
        # Save evaluation summary
        self.output_manager.save_evaluation_summary(final_results["evaluation_summary"])
