#!/usr/bin/env python3
"""
Example usage script for the een_eval framework.

This script demonstrates how to use the framework programmatically
for both simple and advanced evaluation scenarios, using the actual
working API.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

# Import the een_eval framework
from een_eval import EvalWorkflow
from een_eval.core.models import ModelConfig, Model, ModelType
from een_eval.core.evaluation import EvaluationMethod
from een_eval.core.metrics import Metric


def example_basic_usage():
    """Example of basic programmatic usage."""
    print("=== Basic Usage Example ===")
    
    # Create a simple evaluation workflow using constructor approach
    workflow = EvalWorkflow(
        dataset="data/simple_qa.jsonl",
        sample_params={"temperature": 0.0, "max_tokens": 100},
        eval_prompt_template="Question: {{ question }}\nAnswer:",
        evaluation_methods=[
            EvaluationMethod.from_function(
                name="exact_match_evaluation",
                function=custom_exact_match
            )
        ],
        metrics=[
            Metric.from_function(
                name="accuracy_metric",
                function=calculate_accuracy
            )
        ],
        output_dir="results/basic_example",
        mode="inference"
    )
    
    # Add models
    workflow.add_models([Model.from_name("gpt-3.5-turbo")])
    
    # Run the evaluation
    results = workflow.run()
    print(f"Evaluation completed! Results: {results}")


def custom_exact_match(response: str, ground_truth: dict, inference_result: Optional[dict] = None, **params) -> dict:
    """Custom evaluation function following the actual framework interface."""
    expected = ground_truth.get("answer", "")
    is_correct = response.strip().lower() == expected.strip().lower()
    
    return {
        "labels": [
            {
                "label": {"name": "exact_match"},
                "result": {
                    "passed": is_correct,
                    "score": 1.0 if is_correct else 0.0,
                }
            }
        ]
    }


def calculate_accuracy(results: list, **params) -> list:
    """Custom metric function following the actual framework interface."""
    if not results:
        return []
    
    total_items = len(results)
    correct_items = sum(1 for result in results if result.get("passed", False))
    accuracy = correct_items / total_items if total_items > 0 else 0.0
    
    return [
        {
            "metric_name": "accuracy",
            "value": accuracy,
            "count": total_items,
            "correct": correct_items
        }
    ]



def example_advanced_usage():
    """Example of advanced programmatic usage with custom functions."""
    print("=== Advanced Usage Example ===")
    
    # Custom evaluation function
    def similarity_check(response: str, ground_truth: dict, inference_result: Optional[dict] = None, **params) -> dict:
        """Custom evaluation using text similarity."""
        from difflib import SequenceMatcher
        expected = ground_truth.get("answer", "")
        threshold = params.get("threshold", 0.8)
        
        similarity = SequenceMatcher(None, response.lower(), expected.lower()).ratio()
        is_similar = similarity >= threshold
        
        return {
            "labels": [
                {
                    "label": {"name": "similarity_match"},
                    "result": {
                        "passed": is_similar,
                        "score": similarity,
                    }
                }
            ]
        }
    
    # Custom metric function
    def weighted_score(results: list, **params) -> list:
        """Calculate weighted average score."""
        if not results:
            return []
        
        weights = params.get("weights", [1.0] * len(results))
        if len(results) != len(weights):
            weights = [1.0] * len(results)
        
        weighted_sum = sum(r.get("score", 0) * w for r, w in zip(results, weights))
        total_weight = sum(weights)
        weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return [
            {
                "metric_name": "weighted_average",
                "value": weighted_avg,
                "total_weight": total_weight
            }
        ]
    
    # Create workflow with custom functions
    workflow = EvalWorkflow(
        dataset="data/complex_qa.jsonl",
        sample_params={"temperature": 0.0, "max_tokens": 512},
        eval_prompt_template="Question: {{ question }}\nContext: {{ context }}\nAnswer:",
        evaluation_methods=[
            EvaluationMethod.from_function(
                name="similarity_evaluation",
                function=similarity_check,
                params={"threshold": 0.85}
            )
        ],
        metrics=[
            Metric.from_function(
                name="weighted_score_metric",
                function=weighted_score,
                params={"weights": [0.6, 0.4]}
            )
        ],
        output_dir="results/advanced_example",
        mode="inference"
    )
    
    # Add models
    models = [
        Model.from_name("gpt-4"),
        ModelConfig(name="llama2-7b", type=ModelType.LLAMA_CPP, model_path="path/to/model")
    ]
    workflow.add_models(models)
    
    # Run the evaluation
    results = workflow.run()
    print(f"Advanced evaluation completed! Results: {results}")


def example_config_based_usage():
    """Example of using configuration files."""
    print("=== Configuration-based Usage Example ===")
    
    # Load workflow from configuration file - this is the actual working API
    config_path = "examples/basic_config.yaml"
    workflow = EvalWorkflow.from_config(config_path)
    
    # Override some settings programmatically if needed
    workflow.set_mode("inference")
    workflow.set_output_dir("results/config_example")
    
    # Run evaluation
    results = workflow.run()
    print(f"Config-based evaluation completed! Results: {results}")


def example_inference_only():
    """Example of running inference only (no evaluation)."""
    print("=== Inference Only Example ===")
    
    workflow = EvalWorkflow(
        dataset="data/prompts.jsonl",
        sample_params={"temperature": 0.7, "max_tokens": 200},
        eval_prompt_template="Prompt: {{ prompt }}\nResponse:",
        output_dir="results/inference_only",
        mode="inference"
    )
    
    # Add model
    workflow.add_models([Model.from_name("gpt-3.5-turbo")])
    
    # Run inference only
    results = workflow.run()
    print(f"Inference completed! Results: {results}")


def example_evaluation_only():
    """Example of running evaluation on existing predictions."""
    print("=== Evaluation Only Example ===")
    
    # Create workflow for evaluation mode
    workflow = EvalWorkflow(
        dataset="data/qa_with_predictions.jsonl",
        evaluation_methods=[
            EvaluationMethod.from_function(
                name="qa_evaluation",
                function=custom_exact_match
            )
        ],
        metrics=[
            Metric.from_function(
                name="accuracy_metric", 
                function=calculate_accuracy
            )
        ],
        output_dir="results/evaluation_only",
        mode="evaluation"
    )
    
    # Run evaluation only
    results = workflow.run()
    print(f"Evaluation completed! Results: {results}")


def example_resume_workflow():
    """Example of resuming an interrupted workflow."""
    print("=== Resume Workflow Example ===")
    
    # Create workflow with resume enabled
    workflow = EvalWorkflow(
        dataset="data/large_dataset.jsonl",
        sample_params={"temperature": 0.5, "max_tokens": 300},
        eval_prompt_template="Question: {{ question }}\nAnswer:",
        evaluation_methods=[
            EvaluationMethod.from_function(
                name="exact_match_evaluation",
                function=custom_exact_match
            )
        ],
        output_dir="results/resumable_evaluation",
        resume=True,
        mode="inference"
    )
    
    # Add models
    workflow.add_models([Model.from_name("gpt-3.5-turbo")])
    
    try:
        results = workflow.run()
        print(f"Resumed evaluation completed! Results: {results}")
    except KeyboardInterrupt:
        print("Evaluation interrupted - can be resumed later")
    except Exception as e:
        print(f"Could not resume evaluation: {e}")


def example_custom_dataset_and_evaluation():
    print("=== Custom Dataset and Evaluation Example ===")
    
    def load_custom_dataset() -> list:
        """Custom dataset loader function."""
        return [
            {"id": "q1", "question": "What is 2+2?", "answer": "4"},
            {"id": "q2", "question": "What is the capital of France?", "answer": "Paris"},
            {"id": "q3", "question": "Who wrote Hamlet?", "answer": "Shakespeare"}
        ]
    
    def evaluate_custom_qa(response: str, ground_truth: dict, inference_result: Optional[dict] = None, **params) -> dict:
        expected = ground_truth.get("answer", "")
        response_clean = response.strip().lower()
        expected_clean = expected.strip().lower()
        
        # Multiple evaluation criteria
        exact_match = response_clean == expected_clean
        contains_match = expected_clean in response_clean
        length_check = len(response) > 0
        
        return {
            "labels": [
                {
                    "label": {"name": "exact_match"},
                    "result": {
                        "passed": exact_match,
                        "score": 1.0 if exact_match else 0.0,
                    }
                },
                {
                    "label": {"name": "contains_match"},
                    "result": {
                        "passed": contains_match,
                        "score": 1.0 if contains_match else 0.0,
                    }
                },
                {
                    "label": {"name": "length_check"},
                    "result": {
                        "passed": length_check,
                        "score": 1.0 if length_check else 0.0,
                    }
                }
            ]
        }
    
    workflow = EvalWorkflow(
        dataset="data/custom_dataset.jsonl",  # Use file path instead of function
        sample_params={"temperature": 0.0, "max_tokens": 100, "num_samples": 4},
        eval_prompt_template="Question: {{ question }}\nAnswer:",
        evaluation_methods=[
            EvaluationMethod.from_function(
                name="qa_evaluation",
                function=evaluate_custom_qa
            )
        ],
        metrics=[
            Metric.from_function(
                name="pass_at_k",
                function=calculate_pass_at_k,
                params={"k": 1}
            )
        ],
        output_dir="results/custom_example",
        mode="inference"
    )
    
    # Add models
    workflow.add_models([Model.from_name("gpt-3.5-turbo")])
    
    results = workflow.run()
    print(f"Custom evaluation completed! Results: {results}")


def calculate_pass_at_k(results: list, **params) -> list:
    k = params.get("k", 1)
    
    if not results:
        return []
    
    # Group by item_id for pass@k calculation
    grouped_results = {}
    for result in results:
        item_id = result.get("item_id", "default")
        if item_id not in grouped_results:
            grouped_results[item_id] = []
        grouped_results[item_id].append(result)
    
    # Calculate pass@k for each item
    pass_count = 0
    total_items = len(grouped_results)
    
    for item_id, item_results in grouped_results.items():
        # Sort by score/success
        sorted_results = sorted(item_results, key=lambda x: x.get("score", 0), reverse=True)
        # Check if any of the top k results passed
        top_k_results = sorted_results[:k]
        if any(r.get("passed", False) for r in top_k_results):
            pass_count += 1
    
    pass_rate = pass_count / total_items if total_items > 0 else 0.0
    
    return [
        {
            "metric_name": f"pass@{k}",
            "value": pass_rate,
            "passed_items": pass_count,
            "total_items": total_items
        }
    ]



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
    print("   workflow = EvalWorkflow(dataset=..., evaluation_methods=..., ...)")
    print("   workflow.add_models([Model.from_name('gpt-3.5-turbo')])")
    print("   results = workflow.run()")
    
    print("\n2. Advanced Usage:")
    print("   Custom evaluation and metric functions with actual working interface")
    
    print("\n3. Configuration-based:")
    print("   workflow = EvalWorkflow.from_config('config.yaml')")
    
    print("\n4. Inference Only:")
    print("   workflow.set_mode('inference')")
    print("   results = workflow.run()")
    
    print("\n5. Evaluation Only:")
    print("   workflow.set_mode('evaluation')")
    print("   results = workflow.run()")
    
    print("\n6. Resume Workflow:")
    print("   workflow = EvalWorkflow(..., resume=True)")
    print("   results = workflow.run()")
    
    print("\n7. Custom Dataset and Evaluation:")
    print("   with custom functions")
    
    print("\nTo run actual evaluations, uncomment the example function calls in this script.")
    print("Make sure to set up your API keys and model configurations first!")


if __name__ == "__main__":
    main()
