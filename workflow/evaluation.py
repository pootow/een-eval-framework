"""
Evaluation engine for processing inference results and computing metrics.

This module handles the evaluation phase of the workflow, taking inference results
and applying evaluation methods and metrics to compute final scores.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from ..core.evaluation import EvaluationMethod
from ..core.metrics import Metric
from ..utils.io import OutputManager


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
                
                eval_result = result.copy()
                eval_result["evaluations"] = {}

                # Apply each evaluation method
                for method in self.evaluation_methods:
                    try:
                        # Extract response and ground truth from the result
                        response = result.get("response", result.get("generated_output", ""))
                        ground_truth = result.get("ground_truth", {})
                        
                        # Apply evaluation method
                        evaluation_result = method.evaluate(response, ground_truth, sample_data=result)
                        
                        # Handle different return types from evaluation methods
                        if hasattr(evaluation_result, 'score'):
                            # EvaluationResult object
                            eval_result["evaluations"][method.name] = {
                                "score": evaluation_result.score,
                                "passed": evaluation_result.passed,
                                "method": method.name,
                                "details": evaluation_result.details
                            }
                        else:
                            # Direct score value (for backward compatibility)
                            eval_result["evaluations"][method.name] = {
                                "score": evaluation_result,
                                "method": method.name
                            }
                        
                    except Exception as e:
                        self.logger.warning(f"Error applying method {method.name} to sample {idx}: {e}")
                        eval_result["evaluations"][method.name] = {
                            "score": 0.0,
                            "method": method.name,
                            "error": str(e)
                        }
                
                evaluation_results.append(eval_result)
                
            except Exception as e:
                self.logger.error(f"Error processing sample {idx}: {e}")
                status.errors.append(f"Sample {idx}: {str(e)}")
                continue
        
        return evaluation_results
    
    def _compute_metrics(
        self, 
        evaluation_results: List[Dict[str, Any]], 
        status
    ) -> Dict[str, Any]:
        """Compute metrics on evaluation results."""
        metrics_results = {}
        
        # Group results by model and evaluation method
        grouped_results = self._group_results_by_model_and_method(evaluation_results)
        
        for model_name, model_results in grouped_results.items():
            metrics_results[model_name] = {}
            
            for method_name, method_results in model_results.items():
                metrics_results[model_name][method_name] = {}
                
                # Extract scores for this method
                scores = [r["score"] for r in method_results if "error" not in r]
                
                if not scores:
                    self.logger.warning(f"No valid scores for {model_name}.{method_name}")
                    continue

                # Apply each metric
                for metric in self.metrics:
                    try:
                        metric_result = metric.calculate(method_results)
                        if hasattr(metric_result, 'value'):
                            # MetricResult object
                            metrics_results[model_name][method_name][metric.name] = metric_result.value
                        else:
                            # Direct value
                            metrics_results[model_name][method_name][metric.name] = metric_result
                    except Exception as e:
                        self.logger.warning(f"Error computing metric {metric.name}: {e}")
                        metrics_results[model_name][method_name][metric.name] = None
        
        return metrics_results
    
    def _group_samples_by_item(
        self, 
        results: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group evaluation results by original dataset item."""
        from collections import defaultdict
        
        item_groups = defaultdict(list)
        
        for result in results:
            item_id = result.get("item_id", "unknown")
            item_groups[item_id].append(result)
        
        return dict(item_groups)

    def _group_results_by_model_and_method(
        self, 
        evaluation_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Group results by model and evaluation method, preserving item grouping."""
        from collections import defaultdict
        
        grouped = defaultdict(lambda: defaultdict(list))
        
        for result in evaluation_results:
            model_name = result.get("model_name", "unknown")
            method_name = result.get("evaluation_method", "unknown")
            
            # Preserve sample grouping information
            grouped[model_name][method_name].append(result)
        
        return dict(grouped)
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to output directory."""
        # Save detailed results
        self.output_manager.save_evaluation_results(results["detailed_results"])
        
        # Save metrics summary
        metrics_file = Path(self.output_manager.output_dir) / "metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(results["metrics"], f, indent=2, ensure_ascii=False)
        
        # Save evaluation summary
        summary_file = Path(self.output_manager.output_dir) / "evaluation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results["evaluation_summary"], f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Evaluation results saved to {self.output_manager.output_dir}")
