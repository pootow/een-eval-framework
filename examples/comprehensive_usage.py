#!/usr/bin/env python3
"""
Comprehensive usage examples for the een_eval framework.
This file demonstrates all the patterns described in README.md.
"""

import sys
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import een_eval
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from een_eval import EvalWorkflow
from een_eval.core.evaluation import EvaluationMethod
from een_eval.core.metrics import Metric


def example_1_builder_pattern():
    """Example 1: Builder pattern from README.md"""
    print("=== Example 1: Builder Pattern ===")

    import os
    os.chdir(Path(__file__).parent.parent.parent / "eval-workspace" / "prod2")
    
    workflow = (EvalWorkflow()
        .add_models(["deepseek-v3", "claude-3.5-sonnet"])
        .load_dataset("output/qc_dataset.jsonl")
        .add_evaluation_method("qc_correctness", "gauges/qc_eval.py")
        .add_metric("domain_accuracy", "gauges/qc_metrics.py") 
        .set_sample_params(temperature=0.1, max_tokens=1024)
        .set_prompt_template("Problem: {{ problem }}\nConstraints: {{ constraints }}\nSolution:"))

    print(f"Configured workflow with {len(workflow._models)} models")
    print(f"Evaluation methods: {[m.name for m in workflow._evaluation_methods]}")
    print(f"Metrics: {[m.name for m in workflow._metrics]}")
    
    # results = workflow.run()  # Uncomment to actually run
    print("Builder pattern workflow created successfully!\n")


def example_2_config_file():
    """Example 2: Load from config file"""
    print("=== Example 2: Config File ===")
    
    config_path = "examples/updated_config.yaml"
    if Path(config_path).exists():
        workflow = EvalWorkflow.from_config(config_path)
        print(f"Loaded workflow from {config_path}")
        print(f"Models: {[m.name for m in workflow._models]}")
        print(f"Dataset: {workflow._dataset}")
        # results = workflow.run()  # Uncomment to actually run
    else:
        print(f"Config file {config_path} not found, creating sample workflow")
        workflow = EvalWorkflow()
    
    print("Config-based workflow created successfully!\n")


def example_3_custom_evaluation():
    """Example 3: Custom evaluation function"""
    print("=== Example 3: Custom Evaluation Function ===")
    
    def evaluate_code_correctness(response: str, ground_truth: dict, **params) -> dict:
        """Custom evaluation function matching DataFlow.md spec"""
        # Simple example: check if response contains expected keywords
        expected = ground_truth.get("expected_keywords", [])
        passed = all(keyword.lower() in response.lower() for keyword in expected)
        
        return {
            "labels": [
                {
                    "label": {"name": "code_correctness"},
                    "result": {
                        "passed": passed,
                        "score": 1.0 if passed else 0.0,
                        "custom_fields": {
                            "expected_keywords": expected,
                            "found_keywords": [kw for kw in expected if kw.lower() in response.lower()]
                        }
                    }
                }
            ],
            "custom_fields": {
                "response_length": len(response),
                "evaluation_method": "keyword_matching"
            }
        }

    # Create evaluation method from function
    eval_method = EvaluationMethod.from_function("code_correctness", evaluate_code_correctness)
    
    workflow = EvalWorkflow(
        models=["deepseek-v3", "claude-3.5-sonnet"],
        dataset="datasets/qc_problems.jsonl",
        sample_params={"temperature": 0.1, "max_tokens": 1024},
        eval_prompt_template="Problem: {{ problem }}\nConstraints: {{ constraints }}\nSolution:",
        evaluation_methods=[eval_method]
    )
    
    print("Custom evaluation workflow created successfully!")
    print(f"Evaluation method: {eval_method.name}\n")


def example_4_resume_workflow():
    """Example 4: Resume from status"""
    print("=== Example 4: Resume Workflow ===")
    
    workflow = EvalWorkflow(
        models=["deepseek-v3"],
        resume=True,
    )
    
    # Simulate checking completion status
    print(f"Workflow complete: {workflow.is_complete()}")
    print(f"Current status: {workflow.status.mode}")
    
    # In a real scenario, you'd run this in a loop:
    # while not workflow.is_complete():
    #     results = workflow.run()
    #     print(f"Processed {len(results)} samples, status: {workflow.status}")
    
    print("Resume workflow example completed!\n")


def example_5_parameter_override():
    """Example 5: Override parameters"""
    print("=== Example 5: Parameter Override ===")
    
    # Config precedence: script/inline-code > command line > env vars > config file > defaults
    workflow = EvalWorkflow(
        models=["new-model"],  # This will override the one in config
        config="examples/updated_config.yaml" if Path("examples/updated_config.yaml").exists() else None,
    )
    
    print("Parameter override workflow created!")
    if workflow._models:
        print(f"Override model: {workflow._models[0].name}")
    print()


def example_6_cli_integration():
    """Example 6: CLI usage integration"""
    print("=== Example 6: CLI Integration ===")
    
    # This would be used when the script is called with command line arguments
    # python script.py --models deepseek-v3 claude-3.5-sonnet
    
    parser = argparse.ArgumentParser(description="Run evaluation workflow")
    parser.add_argument("--models", nargs="+", help="Models to evaluate")
    parser.add_argument("--dataset", help="Dataset file path")
    parser.add_argument("--output-dir", help="Output directory")
    
    # Parse known args to avoid conflicts
    args, _ = parser.parse_known_args()
    
    workflow_kwargs = {}
    if args.models:
        workflow_kwargs["models"] = args.models
    if args.dataset:
        workflow_kwargs["dataset"] = args.dataset
    if args.output_dir:
        workflow_kwargs["output_dir"] = args.output_dir
    
    workflow = EvalWorkflow(
        config="examples/updated_config.yaml" if Path("examples/updated_config.yaml").exists() else None,
        **workflow_kwargs
    )
    
    print("CLI integration workflow created!")
    if args.models:
        print(f"CLI models: {args.models}")
    print()


def example_7_advanced_metrics():
    """Example 7: Advanced metrics with facets"""
    print("=== Example 7: Advanced Metrics with Facets ===")
    
    def complexity_weighted_average(evaluation_results, facets=None, weight_factor=0.8, **kwargs):
        """Custom metric function matching the README.md example"""
        # Group by facets (this would be more complex in real implementation)
        if not evaluation_results:
            return []
        
        # Simple example: weighted average of scores
        total_score = sum(r.get("score", 0) * weight_factor for r in evaluation_results)
        count = len(evaluation_results)
        
        return [{
            "metric_name": "complexity_score", 
            "weighted_average": total_score / count if count > 0 else 0,
            "sample_count": count,
            "weight_factor": weight_factor
        }]
    
    # Create custom metric
    custom_metric = Metric.from_function("complexity_score", complexity_weighted_average, weight_factor=0.8)
    
    workflow = EvalWorkflow(
        models=["deepseek-v3"],
        dataset="datasets/qc_problems.jsonl",
        metrics=[custom_metric],
        sample_params={"temperature": 0.1, "num_samples": 16}
    )
    
    print("Advanced metrics workflow created!")
    print(f"Custom metric: {custom_metric.name}")
    print()


def main():
    """Run all examples"""
    print("Een_eval Framework Usage Examples")
    print("=" * 50)
    
    try:
        example_1_builder_pattern()
        example_2_config_file()
        example_3_custom_evaluation()
        example_4_resume_workflow()
        example_5_parameter_override()
        example_6_cli_integration()
        example_7_advanced_metrics()
        
        print("All examples completed successfully!")
        print("\nTo run actual evaluations:")
        print("1. Ensure you have datasets in the correct format")
        print("2. Configure your models (local or API)")
        print("3. Uncomment the workflow.run() calls")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        # print detailed traceback if needed
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
