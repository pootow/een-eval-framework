#!/usr/bin/env python3
"""
Example usage script for the een_eval framework.

This script demonstrates how to use the framework programmatically
for both simple and advanced evaluation scenarios.
"""

import os
import json
from pathlib import Path

# Import the een_eval framework
from een_eval import EvalWorkflow
from een_eval.core.models import ModelConfig
from een_eval.core.evaluation import EvaluationMethod
from een_eval.core.metrics import Metric
from een_eval.core.dataset import Dataset
from een_eval.workflow.config import Config


def example_basic_usage():
    """Example of basic programmatic usage."""
    print("=== Basic Usage Example ===")
    
    # Create a simple evaluation workflow
    workflow = (EvalWorkflow.builder()
                .dataset_from_file("data/simple_qa.jsonl", 
                                 input_field="question",
                                 expected_output_field="answer")
                .add_openai_model("gpt-3.5-turbo", temperature=0.0)
                .add_evaluation_method("exact_match")
                .add_metric("pass_rate")
                .output_directory("results/basic_example")
                .build())
    
    # Run the evaluation
    results = workflow.run()
    print(f"Evaluation completed! Results: {results}")


def example_advanced_usage():
    """Example of advanced programmatic usage with custom functions."""
    print("=== Advanced Usage Example ===")
    
    # Custom evaluation function
    def similarity_check(prediction: str, expected: str, threshold: float = 0.8) -> bool:
        """Custom evaluation using text similarity."""
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, prediction.lower(), expected.lower()).ratio()
        return similarity >= threshold
    
    # Custom metric function
    def weighted_score(scores: list, weights: list = None) -> float:
        """Calculate weighted average score."""
        if not weights:
            weights = [1.0] * len(scores)
        if len(scores) != len(weights):
            raise ValueError("Scores and weights must have same length")
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    
    # Create workflow with custom functions
    workflow = (EvalWorkflow.builder()
                .dataset_from_file("data/complex_qa.jsonl",
                                 input_field="prompt",
                                 expected_output_field="expected")
                .add_openai_model("gpt-4", temperature=0.0, max_tokens=512)
                .add_vllm_model("llama2-7b", 
                              model_path="meta-llama/Llama-2-7b-chat-hf",
                              docker_image="vllm/vllm-openai:latest")
                .add_evaluation_method("exact_match")
                .add_custom_evaluation_method("similarity", similarity_check,
                                            config={"threshold": 0.85})
                .add_metric("pass_rate")
                .add_custom_metric("weighted_score", weighted_score,
                                 config={"weights": [0.6, 0.4]})
                .inference_config(batch_size=8, max_parallel=2)
                .output_directory("results/advanced_example")
                .enable_resume()
                .build())
    
    # Set up progress monitoring
    def progress_callback(status):
        print(f"Progress: {status.progress_percent:.1f}% - "
              f"Model: {status.current_model}")
    
    workflow.set_progress_callback(progress_callback)
    
    # Run the evaluation
    results = workflow.run()
    print(f"Advanced evaluation completed! Results: {results}")


def example_config_based_usage():
    """Example of using configuration files."""
    print("=== Configuration-based Usage Example ===")
    
    # Load workflow from configuration file
    config_path = "examples/basic_config.yaml"
    workflow = EvalWorkflow.from_config_file(config_path)
    
    # Override some settings programmatically
    workflow.config.inference.batch_size = 4
    workflow.config.output.directory = "results/config_example"
    
    # Run evaluation
    results = workflow.run()
    print(f"Config-based evaluation completed! Results: {results}")


def example_inference_only():
    """Example of running inference only (no evaluation)."""
    print("=== Inference Only Example ===")
    
    workflow = (EvalWorkflow.builder()
                .dataset_from_file("data/prompts.jsonl", input_field="prompt")
                .add_openai_model("gpt-3.5-turbo", temperature=0.7)
                .output_directory("results/inference_only")
                .build())
    
    # Run inference only
    predictions = workflow.run_inference()
    print(f"Inference completed! Generated {len(predictions)} predictions")


def example_evaluation_only():
    """Example of running evaluation on existing predictions."""
    print("=== Evaluation Only Example ===")
    
    # Assume we have predictions from a previous run
    workflow = (EvalWorkflow.builder()
                .dataset_from_file("data/qa_with_predictions.jsonl",
                                 input_field="question",
                                 expected_output_field="answer",
                                 prediction_field="model_prediction")
                .add_evaluation_method("exact_match")
                .add_evaluation_method("contains", case_sensitive=False)
                .add_metric("pass_rate")
                .add_metric("mean")
                .output_directory("results/evaluation_only")
                .build())
    
    # Run evaluation only
    results = workflow.run_evaluation()
    print(f"Evaluation completed! Results: {results}")


def example_resume_workflow():
    """Example of resuming an interrupted workflow."""
    print("=== Resume Workflow Example ===")
    
    # Resume from a previous evaluation directory
    resume_dir = "results/interrupted_evaluation"
    
    if Path(resume_dir).exists():
        try:
            workflow = EvalWorkflow.resume_from_directory(resume_dir)
            results = workflow.resume()
            print(f"Resumed evaluation completed! Results: {results}")
        except Exception as e:
            print(f"Could not resume evaluation: {e}")
    else:
        print(f"Resume directory not found: {resume_dir}")


def create_sample_dataset():
    """Create a sample dataset for testing."""
    print("Creating sample dataset...")
    
    sample_data = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris"
        },
        {
            "question": "What is 2 + 2?",
            "answer": "4"
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "answer": "William Shakespeare"
        },
        {
            "question": "What is the largest planet in our solar system?",
            "answer": "Jupiter"
        },
        {
            "question": "In what year did World War II end?",
            "answer": "1945"
        }
    ]
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Save as JSONL
    with open("data/simple_qa.jsonl", "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    
    print("Sample dataset created at: data/simple_qa.jsonl")


def main():
    """Main function to run all examples."""
    print("een_eval Framework Usage Examples")
    print("=" * 50)
    
    # Create sample data first
    create_sample_dataset()
    
    # Run examples (commented out to avoid API calls in demo)
    # Uncomment the examples you want to run:
    
    print("\n1. Basic Usage:")
    print("   workflow = EvalWorkflow.builder().dataset_from_file(...).add_openai_model(...).build()")
    
    print("\n2. Advanced Usage:")
    print("   workflow = EvalWorkflow.builder().add_custom_evaluation_method(...).build()")
    
    print("\n3. Configuration-based:")
    print("   workflow = EvalWorkflow.from_config_file('config.yaml')")
    
    print("\n4. Inference Only:")
    print("   predictions = workflow.run_inference()")
    
    print("\n5. Evaluation Only:")
    print("   results = workflow.run_evaluation()")
    
    print("\n6. Resume Workflow:")
    print("   workflow = EvalWorkflow.resume_from_directory('results/dir')")
    
    print("\nTo run actual evaluations, uncomment the example function calls in this script.")
    print("Make sure to set up your API keys and model configurations first!")


if __name__ == "__main__":
    main()
