"""
Inference engine for the evaluation framework.

This module handles model inference execution, prompt processing,
and response collection with support for batching and resumption.
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        max_workers: int = 4,
        resume: bool = False
    ):
        self.models = models
        self.dataset = dataset
        self.sample_params = sample_params
        self.num_samples = sample_params.get('num_samples', 1)  # Extract num_samples
        self.prompt_processor = PromptProcessor(prompt_template)
        self.output_manager = output_manager
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.resume = resume
        self.logger = logging.getLogger(__name__)

        # State tracking
        self.completed_samples = 0
        self.total_samples = len(dataset) * len(models) * self.num_samples
        self.start_time: Optional[float] = None
        
        # Thread synchronization
        self._lock = threading.Lock()

        # Resume tracking - load existing responses if resuming
        self.existing_responses: Dict[str, InferenceResult] = {}
        if self.resume and self.output_manager:
            self._load_existing_responses()
    
    def _load_existing_responses(self) -> None:
        """Load existing responses for resume functionality."""
        if not self.output_manager:
            return
        
        try:
            responses = self.output_manager.load_responses()
            successful_responses = []
            failed_responses = []
            
            for response_data in responses:
                # Create key for tracking: "{model_name}_{item_id}_{sample_index}"
                key = f"{response_data.get('model_name')}_{response_data.get('item_id')}_{response_data.get('sample_index')}"
                
                # Convert dict back to InferenceResult
                result = InferenceResult(
                    item_id=response_data.get('item_id', ''),
                    sample_id=response_data.get('sample_id', ''),
                    sample_index=response_data.get('sample_index', 0),
                    total_samples=response_data.get('total_samples', 1),
                    model_name=response_data.get('model_name', ''),
                    prompt=response_data.get('prompt', ''),
                    response=response_data.get('response', ''),
                    inference_time=response_data.get('inference_time', 0.0),
                    timestamp=response_data.get('timestamp', 0.0),
                    ground_truth=response_data.get('ground_truth'),
                    metadata=response_data.get('metadata', {}),
                    error=response_data.get('error'),
                    tokens_per_second=response_data.get('tokens_per_second'),
                    total_tokens=response_data.get('total_tokens'),
                    prompt_tokens=response_data.get('prompt_tokens'),
                    completion_tokens=response_data.get('completion_tokens')
                )
                
                # Separate successful and failed responses
                if not result.error:
                    self.existing_responses[key] = result
                    successful_responses.append(response_data)
                else:
                    failed_responses.append(response_data)
              # Save failed responses to separate file if any exist
            if failed_responses:
                self.output_manager.save_failed_responses(failed_responses)
            
            # Rewrite responses.jsonl with only successful responses
            if successful_responses:
                self.output_manager.rewrite_responses_file(successful_responses)
            else:
                # Clear the responses file if no successful responses
                self.output_manager.clear_responses_file()
                    
            self.logger.info(f"Loaded {len(self.existing_responses)} existing successful responses for resume")
            if failed_responses:
                self.logger.info(f"Moved {len(failed_responses)} failed responses to failed_responses.jsonl")
            
        except Exception as e:
            self.logger.warning(f"Failed to load existing responses for resume: {e}")
    
    def _is_sample_completed(self, model_name: str, item_id: str, sample_index: int) -> bool:
        """Check if a specific sample is already completed successfully."""
        key = f"{model_name}_{item_id}_{sample_index}"
        return key in self.existing_responses
    
    def _get_existing_sample(self, model_name: str, item_id: str, sample_index: int) -> Optional[InferenceResult]:
        """Get existing sample result if available."""
        key = f"{model_name}_{item_id}_{sample_index}"
        return self.existing_responses.get(key)
    
    def run(self, status) -> Dict[str, Any]:
        """Run inference on all models and dataset items."""
        self.logger.info("Starting inference execution")
        self.start_time = time.time()
        
        # Save inference metadata
        metadata = {
            "start_time": time.time(),
            "models": [model.name for model in self.models],
            "dataset_size": len(self.dataset),
            "sample_params": self.sample_params,
            "prompt_template": self.prompt_processor.template,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers
        }
        if self.output_manager:
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

        # merge stats with metadata and save
        if self.output_manager:
            self.output_manager.save_inference_metadata({**metadata, **stats})

        self.logger.info(f"Inference completed in {total_time:.2f}s")
        return {
            "results": results,
            "statistics": stats,
            "status": status.to_dict()
        }

    def _run_model_inference(self, model: Model, status) -> List[InferenceResult]:
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
        # REVIEW: this batching is wrong, it batches items, not samples
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
    ) -> List[InferenceResult]:
        """Run inference sequentially."""
        results = []

        for batch in batches:
            batch_results = self._process_batch(model, batch)
            
            # Thread-safe updates (defensive programming)
            with self._lock:
                results.extend(batch_results)
                
                # Update status and save intermediate results
                self.completed_samples += len(batch_results)
                status.processed_samples = self.completed_samples
                
                if self.output_manager:
                    # Only save new results (not existing ones)
                    new_results = [r for r in batch_results if r not in self.existing_responses.values()]
                    if new_results:
                        self.output_manager.save_responses_batch(new_results)
                    self.output_manager.save_status(status)
        
        return results

    def _run_parallel_inference(
        self, 
        model: Model, 
        batches: List[List[DatasetItem]], 
        status
    ) -> List[InferenceResult]:
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
                    
                    # Thread-safe updates using lock
                    with self._lock:
                        results.extend(batch_results)
                        
                        # Update status and save intermediate results
                        self.completed_samples += len(batch_results)
                        status.processed_samples = self.completed_samples
                        
                        if self.output_manager:
                            # Only save new results (not existing ones)
                            new_results = [r for r in batch_results if r not in self.existing_responses.values()]
                            if new_results:
                                self.output_manager.save_responses_batch(new_results)
                            self.output_manager.save_status(status)
                        
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {e}")
                    # Thread-safe error logging
                    with self._lock:
                        status.errors.append(f"Batch processing: {e}")
        
        return results

    def _process_batch(self, model: Model, batch: List[DatasetItem]) -> List[InferenceResult]:
        """Process a single batch of items."""
        batch_results = []
        
        for item in batch:
            # Generate multiple samples for each item
            for sample_idx in range(self.num_samples):
                # Check if sample already exists when resuming
                if self.resume and self._is_sample_completed(model.name, item.id, sample_idx):
                    existing_result = self._get_existing_sample(model.name, item.id, sample_idx)
                    if existing_result:
                        batch_results.append(existing_result)
                        self.logger.debug(f"Skipped existing sample: {model.name}_{item.id}_{sample_idx}")
                        continue
                
                prompt = ""
                try:
                    # Process prompt
                    prompt = self.prompt_processor.process_prompt(item)
                    
                    # Generate response
                    simple_result = model.generate(prompt, **self.sample_params)
                    
                    # Create full InferenceResult
                    if simple_result.error:
                        # Handle error case
                        result = InferenceResult(
                            item_id=item.id,
                            sample_id=f"{item.id}_sample_{sample_idx}",
                            sample_index=sample_idx,
                            total_samples=self.num_samples,
                            model_name=model.name,
                            prompt=prompt,
                            response="",
                            inference_time=simple_result.inference_time,
                            timestamp=time.time(),
                            ground_truth=item.data,
                            error=simple_result.error,
                            metadata={
                                "model_id": model.name,
                                "prompt_template": self.prompt_processor.template,
                                "sampling": self.sample_params,
                                **simple_result.metadata
                            },
                            tokens_per_second=simple_result.tokens_per_second,
                            total_tokens=simple_result.total_tokens,
                            prompt_tokens=simple_result.prompt_tokens,
                            completion_tokens=simple_result.completion_tokens
                        )
                    else:
                        # Success case
                        result = InferenceResult(
                            item_id=item.id,
                            sample_id=f"{item.id}_sample_{sample_idx}",
                            sample_index=sample_idx,
                            total_samples=self.num_samples,
                            model_name=model.name,
                            prompt=prompt,
                            response=simple_result.response,
                            inference_time=simple_result.inference_time,
                            timestamp=time.time(),
                            ground_truth=item.data,
                            metadata={
                                "model_id": model.name,
                                "prompt_template": self.prompt_processor.template,
                                "sampling": self.sample_params,
                                **simple_result.metadata
                            },
                            tokens_per_second=simple_result.tokens_per_second,
                            total_tokens=simple_result.total_tokens,
                            prompt_tokens=simple_result.prompt_tokens,
                            completion_tokens=simple_result.completion_tokens
                        )
                    
                    batch_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process item {item.id}, sample {sample_idx}: {e}")
                    # Create error result
                    error_result = InferenceResult(
                        item_id=item.id,
                        sample_id=f"{item.id}_sample_{sample_idx}",
                        sample_index=sample_idx,
                        total_samples=self.num_samples,
                        model_name=model.name,
                        prompt=prompt,
                        response="",
                        inference_time=0.0,
                        timestamp=time.time(),
                        ground_truth=item.data,
                        error=str(e),
                        metadata={
                            "model_id": model.name,
                            "prompt_template": self.prompt_processor.template,
                            "sampling": self.sample_params
                        }
                    )
                    batch_results.append(error_result)
        
        return batch_results

    def _calculate_stats(self, results: List[InferenceResult], total_time: float) -> Dict[str, Any]:
        """Calculate inference statistics."""
        if not results:
            return {}
        
        # Filter successful results
        successful_results = [r for r in results if r.error is None]
        
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
            inference_times = [r.inference_time for r in successful_results]
            stats.update({
                "avg_inference_time": sum(inference_times) / len(inference_times),
                "min_inference_time": min(inference_times),
                "max_inference_time": max(inference_times)
            })
            
            # Token statistics (if available)
            token_results = [r for r in successful_results if r.total_tokens is not None]
            if token_results:
                total_tokens = [r.total_tokens for r in token_results if r.total_tokens is not None]
                tokens_per_sec = [r.tokens_per_second for r in token_results if r.tokens_per_second is not None]
                
                stats.update({
                    "avg_total_tokens": sum(total_tokens) / len(total_tokens),
                    "min_total_tokens": min(total_tokens),
                    "max_total_tokens": max(total_tokens),
                })
                
                if tokens_per_sec:
                    stats.update({
                        "avg_tokens_per_second": sum(tokens_per_sec) / len(tokens_per_sec),
                        "min_tokens_per_second": min(tokens_per_sec),
                        "max_tokens_per_second": max(tokens_per_sec)
                    })
        
        return stats
