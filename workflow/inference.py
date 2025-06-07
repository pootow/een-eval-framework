"""
Inference engine for the evaluation framework.

This module handles model inference execution, prompt processing,
and response collection with support for batching and resumption.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import jinja2

from ..core.models import Model, InferenceResult
from ..core.dataset import Dataset, DatasetItem
from ..utils.io import OutputManager


class PromptProcessor:
    """Processes prompts using Jinja2 templates."""
    
    def __init__(self, template: Optional[str] = None):
        self.template = template
        self.jinja_env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            autoescape=False
        )
        self.logger = logging.getLogger(__name__)
    
    def process_prompt(self, item: DatasetItem, template: Optional[str] = None) -> str:
        """Process a single prompt from dataset item."""
        template_str = template or self.template
        
        if not template_str:
            # No template, use raw data
            if 'prompt' in item.data:
                return str(item.data['prompt'])
            elif 'question' in item.data:
                return str(item.data['question'])
            elif 'problem' in item.data:
                return str(item.data['problem'])
            else:
                # Use all data as context
                return str(item.data)
        
        try:
            template_obj = self.jinja_env.from_string(template_str)
            return template_obj.render(**item.data)
        except Exception as e:
            self.logger.error(f"Failed to process prompt for item {item.id}: {e}")
            # Fallback to raw data
            return str(item.data)


class InferenceEngine:
    """Handles model inference execution."""
    
    def __init__(
        self,
        models: List[Model],
        dataset: Dataset,
        sample_params: Dict[str, Any],
        prompt_template: Optional[str] = None,
        output_manager: Optional[OutputManager] = None,
        batch_size: int = 1,
        max_workers: int = 4
    ):
        self.models = models
        self.dataset = dataset
        self.sample_params = sample_params
        self.prompt_processor = PromptProcessor(prompt_template)
        self.output_manager = output_manager
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # State tracking
        self.completed_samples = 0
        self.total_samples = len(dataset) * len(models)
        self.start_time: Optional[float] = None
    
    def run(self, status) -> Dict[str, Any]:
        """Run inference on all models and dataset items."""
        self.logger.info("Starting inference execution")
        self.start_time = time.time()
        
        # Save inference metadata
        if self.output_manager:
            metadata = {
                "start_time": time.time(),
                "models": [model.name for model in self.models],
                "dataset_size": len(self.dataset),
                "sample_params": self.sample_params,
                "prompt_template": self.prompt_processor.template,
                "batch_size": self.batch_size,
                "max_workers": self.max_workers
            }
            self.output_manager.save_inference_metadata(metadata)
        
        results = []
        
        # Process each model
        for model in self.models:
            self.logger.info(f"Running inference for model: {model.name}")
            status.current_model = model.name
            
            try:
                with model:  # Use context manager for model lifecycle
                    model_results = self._run_model_inference(model, status)
                    results.extend(model_results)
            except Exception as e:
                self.logger.error(f"Failed inference for model {model.name}: {e}")
                status.errors.append(f"Model {model.name}: {e}")
        
        # Calculate final statistics
        total_time = time.time() - self.start_time
        stats = self._calculate_stats(results, total_time)
        
        self.logger.info(f"Inference completed in {total_time:.2f}s")
        return {
            "results": results,
            "statistics": stats,
            "status": status.to_dict()
        }
    
    def _run_model_inference(self, model: Model, status) -> List[Dict[str, Any]]:
        """Run inference for a single model on all dataset items."""
        model_results = []
        
        # Create batches
        batches = self._create_batches()
        
        # Process batches
        if self.max_workers > 1:
            model_results = self._run_parallel_inference(model, batches, status)
        else:
            model_results = self._run_sequential_inference(model, batches, status)
        
        return model_results
    
    def _create_batches(self) -> List[List[DatasetItem]]:
        """Create batches from dataset items."""
        batches = []
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    def _run_sequential_inference(
        self, 
        model: Model, 
        batches: List[List[DatasetItem]], 
        status
    ) -> List[Dict[str, Any]]:
        """Run inference sequentially."""
        results = []
        
        for batch in batches:
            batch_results = self._process_batch(model, batch)
            results.extend(batch_results)
            
            # Update status and save intermediate results
            self.completed_samples += len(batch)
            status.processed_samples = self.completed_samples
            
            if self.output_manager:
                self.output_manager.save_responses_batch(batch_results)
                self.output_manager.save_status(status)
        
        return results
    
    def _run_parallel_inference(
        self, 
        model: Model, 
        batches: List[List[DatasetItem]], 
        status
    ) -> List[Dict[str, Any]]:
        """Run inference in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._process_batch, model, batch): batch
                for batch in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                    
                    # Update status and save intermediate results
                    self.completed_samples += len(batch)
                    status.processed_samples = self.completed_samples
                    
                    if self.output_manager:
                        self.output_manager.save_responses_batch(batch_results)
                        self.output_manager.save_status(status)
                        
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {e}")
                    status.errors.append(f"Batch processing: {e}")
        
        return results
    
    def _process_batch(self, model: Model, batch: List[DatasetItem]) -> List[Dict[str, Any]]:
        """Process a single batch of items."""
        batch_results = []
        
        for item in batch:
            try:
                # Process prompt
                prompt = self.prompt_processor.process_prompt(item)
                
                # Generate response
                inference_result = model.generate(prompt, **self.sample_params)
                
                # Create result record
                result_data = {
                    "item_id": item.id,
                    "model_name": model.name,
                    "prompt": prompt,
                    "response": inference_result.response,
                    "inference_time": inference_result.inference_time,
                    "tokens_per_second": inference_result.tokens_per_second,
                    "total_tokens": inference_result.total_tokens,
                    "prompt_tokens": inference_result.prompt_tokens,
                    "completion_tokens": inference_result.completion_tokens,
                    "timestamp": time.time(),
                    "sample_params": self.sample_params,
                    "metadata": inference_result.metadata,
                    "ground_truth": item.data
                }
                
                batch_results.append(result_data)
                
            except Exception as e:
                self.logger.error(f"Failed to process item {item.id}: {e}")
                # Create error record
                error_result = {
                    "item_id": item.id,
                    "model_name": model.name,
                    "prompt": "",
                    "response": "",
                    "inference_time": 0,
                    "tokens_per_second": 0,
                    "error": str(e),
                    "timestamp": time.time(),
                    "ground_truth": item.data
                }
                batch_results.append(error_result)
        
        return batch_results
    
    def _calculate_stats(self, results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """Calculate inference statistics."""
        if not results:
            return {}
        
        # Filter successful results
        successful_results = [r for r in results if "error" not in r]
        
        # Basic statistics
        stats = {
            "total_samples": len(results),
            "successful_samples": len(successful_results),
            "failed_samples": len(results) - len(successful_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
            "total_time": total_time,
            "average_time_per_sample": total_time / len(results) if results else 0
        }
        
        if successful_results:
            # Timing statistics
            inference_times = [r["inference_time"] for r in successful_results]
            stats.update({
                "avg_inference_time": sum(inference_times) / len(inference_times),
                "min_inference_time": min(inference_times),
                "max_inference_time": max(inference_times)
            })
            
            # Token statistics
            total_tokens = [r.get("total_tokens", 0) for r in successful_results if r.get("total_tokens")]
            if total_tokens:
                stats.update({
                    "avg_total_tokens": sum(total_tokens) / len(total_tokens),
                    "min_total_tokens": min(total_tokens),
                    "max_total_tokens": max(total_tokens),
                    "total_tokens_generated": sum(total_tokens)
                })
            
            # Speed statistics
            speeds = [r.get("tokens_per_second", 0) for r in successful_results if r.get("tokens_per_second")]
            if speeds:
                stats.update({
                    "avg_tokens_per_second": sum(speeds) / len(speeds),
                    "min_tokens_per_second": min(speeds),
                    "max_tokens_per_second": max(speeds)
                })
        
        # Per-model statistics
        model_stats = {}
        for result in results:
            model_name = result.get("model_name", "unknown")
            if model_name not in model_stats:
                model_stats[model_name] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "total_time": 0
                }
            
            model_stats[model_name]["total"] += 1
            if "error" not in result:
                model_stats[model_name]["successful"] += 1
                model_stats[model_name]["total_time"] += result.get("inference_time", 0)
            else:
                model_stats[model_name]["failed"] += 1
        
        # Calculate per-model averages
        for model_name, model_stat in model_stats.items():
            if model_stat["successful"] > 0:
                model_stat["avg_time"] = model_stat["total_time"] / model_stat["successful"]
                model_stat["success_rate"] = model_stat["successful"] / model_stat["total"]
        
        stats["per_model"] = model_stats
        
        return stats
