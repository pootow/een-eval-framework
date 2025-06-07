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
        inference_file = Path(self.output_manager.output_dir) / "inference_results.jsonl"
        
        if not inference_file.exists():
            return []
        
        results = []
        try:
            with open(inference_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        results.append(json.loads(line))
        except Exception as e:
            self.logger.error(f"Error loading inference results: {e}")
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
                        # Extract expected output from the result
                        expected = result.get("expected_output", result.get("ground_truth", ""))
                        generated = result.get("generated_output", "")
                        
                        # Apply evaluation method
                        evaluation_score = method.evaluate(generated, expected, result)
                        eval_result["evaluations"][method.name] = {
                            "score": evaluation_score,
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
                        metric_value = metric.compute(scores, evaluation_results)
                        metrics_results[model_name][method_name][metric.name] = metric_value
                    except Exception as e:
                        self.logger.warning(f"Error computing metric {metric.name}: {e}")
                        metrics_results[model_name][method_name][metric.name] = None
        
        return metrics_results
    
    def _group_results_by_model_and_method(
        self, 
        evaluation_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Group evaluation results by model and evaluation method."""
        grouped = {}
        
        for result in evaluation_results:
            model_name = result.get("model_name", "unknown")
            
            if model_name not in grouped:
                grouped[model_name] = {}
            
            evaluations = result.get("evaluations", {})
            for method_name, eval_data in evaluations.items():
                if method_name not in grouped[model_name]:
                    grouped[model_name][method_name] = []
                
                grouped[model_name][method_name].append(eval_data)
        
        return grouped
    
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
