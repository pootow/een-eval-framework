#!/usr/bin/env python3
"""
Command-line interface for the een_eval framework.

This module provides comprehensive CLI support for running evaluations,
managing configurations, and monitoring progress.
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

from .workflow.workflow import EvalWorkflow
from .workflow.config import Config
from .core.models import ModelConfig
from .core.evaluation import BuiltInEvaluationMethod
from .core.metrics import BuiltInMetric
from .utils.validation import ConfigValidator


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Set up logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def create_sample_config(output_path: str, config_type: str = "basic") -> None:
    """Create a sample configuration file."""
    if config_type == "basic":
        config = {
            "dataset": {
                "file_path": "data/eval_dataset.jsonl",
                "input_field": "prompt",
                "expected_output_field": "expected_response"
            },
            "models": [
                {
                    "name": "gpt-4",
                    "type": "openai",
                    "config": {
                        "model": "gpt-4",
                        "temperature": 0.0,
                        "max_tokens": 512
                    }
                }
            ],
            "evaluation_methods": [
                {
                    "name": "exact_match",
                    "type": "exact_match"
                }
            ],
            "metrics": [
                {
                    "name": "pass_rate",
                    "type": "pass_rate"
                }
            ],
            "output": {
                "directory": "results",
                "save_predictions": True,
                "save_evaluations": True
            }
        }
    elif config_type == "vllm":
        config = {
            "dataset": {
                "file_path": "data/eval_dataset.jsonl",
                "input_field": "prompt",
                "expected_output_field": "expected_response",
                "sample_size": 100
            },
            "models": [
                {
                    "name": "local-llama",
                    "type": "vllm",
                    "config": {
                        "model": "meta-llama/Llama-2-7b-chat-hf",
                        "temperature": 0.0,
                        "max_tokens": 512,
                        "docker_image": "vllm/vllm-openai:latest",
                        "gpu_count": 1,
                        "port": 8000
                    }
                }
            ],
            "evaluation_methods": [
                {
                    "name": "contains_check",
                    "type": "contains",
                    "config": {"case_sensitive": False}
                },
                {
                    "name": "json_validation",
                    "type": "json_match",
                    "config": {"strict": True}
                }
            ],
            "metrics": [
                {
                    "name": "pass_rate",
                    "type": "pass_rate"
                },
                {
                    "name": "mean_score",
                    "type": "mean"
                }
            ],
            "inference": {
                "batch_size": 8,
                "max_workers": 4,
                "timeout": 30
            },
            "output": {
                "directory": "results",
                "save_predictions": True,
                "save_evaluations": True,
                "save_intermediate": True
            }
        }
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if output_file.suffix.lower() in ['.yaml', '.yml']:
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    
    print(f"Sample configuration saved to: {output_file}")


def validate_config_command(args: argparse.Namespace) -> int:
    """Validate a configuration file."""
    try:
        config_data = load_config_file(args.config)
        config = Config.from_dict(config_data)
        
        validator = ConfigValidator()
        is_valid, errors = validator.validate_config(config)
        
        if is_valid:
            print("✅ Configuration is valid!")
            return 0
        else:
            print("❌ Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            return 1
            
    except Exception as e:
        print(f"❌ Error validating configuration: {e}")
        return 1


def list_available_components() -> None:
    """List all available built-in evaluation methods and metrics."""
    print("Available Built-in Evaluation Methods:")
    evaluation_methods = [
        "exact_match", "contains", "regex_match", 
        "json_match", "numeric_match", "length_check"
    ]
    for method in evaluation_methods:
        print(f"  - {method}")
    
    print("\nAvailable Built-in Metrics:")
    metrics = [
        "pass_at_k", "mean", "median", "percentile", 
        "pass_rate", "count", "std", "min", "max", "sum"
    ]
    for metric in metrics:
        print(f"  - {metric}")


def run_evaluation(args: argparse.Namespace) -> int:
    """Run the evaluation workflow."""
    try:
        # Load configuration
        if args.config:
            config_data = load_config_file(args.config)
            config = Config.from_dict(config_data)
        else:
            # Create minimal config from command line args
            config_data = create_minimal_config_from_args(args)
            config = Config.from_dict(config_data)
        
        # Override config with command line arguments
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.max_workers:
            config.max_workers = args.max_workers
          # Create workflow from config
        workflow = EvalWorkflow(config=config)
        
        # Set mode
        if args.mode == "inference":
            workflow.set_mode("inference")
        elif args.mode == "evaluation":
            workflow.set_mode("evaluation")
        # "full" mode runs both inference and evaluation
        
        # Run workflow
        results = workflow.run()
        
        print(f"✅ Evaluation completed successfully!")
        print(f"Results saved to: {config.output_dir or 'output'}")
        
        # Print summary if available
        if results and isinstance(results, dict):
            print("\nSummary:")
            for key, value in results.items():
                if key != 'raw_results':  # Skip large raw data
                    print(f"  {key}: {value}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Evaluation interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def create_minimal_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Create a minimal configuration from command line arguments."""
    if not args.dataset:
        raise ValueError("Dataset path is required when not using a config file")
    
    config = {
        "dataset": {
            "file_path": args.dataset,
            "input_field": args.input_field or "input",
            "expected_output_field": args.expected_field or "expected"
        },
        "models": [
            {
                "name": args.model_name or "default",
                "type": args.model_type or "openai",
                "config": {
                    "model": args.model_name or "gpt-3.5-turbo",
                    "temperature": args.temperature or 0.0,
                    "max_tokens": args.max_tokens or 512
                }
            }
        ],
        "evaluation_methods": [
            {
                "name": "exact_match",
                "type": "exact_match"
            }
        ],
        "metrics": [
            {
                "name": "pass_rate",
                "type": "pass_rate"
            }
        ],
        "output": {
            "directory": args.output_dir or "results",
            "save_predictions": True,
            "save_evaluations": True
        }
    }
    
    return config


def resume_evaluation(args: argparse.Namespace) -> int:
    """Resume a previously interrupted evaluation."""
    try:
        if not args.output_dir:
            print("❌ Output directory is required")
            return 1
        
        # Load config if provided
        config = None
        if args.config:
            config_data = load_config_file(args.config)
            config = Config.from_dict(config_data)
        else:
            # Create default config
            config = Config()
        
        # Set resume settings
        config.resume = True
        config.resume_from = args.output_dir
        config.output_dir = args.output_dir
        
        workflow = EvalWorkflow(config=config)
        results = workflow.run()
        
        print(f"✅ Evaluation resumed and completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error resuming evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="een_eval: Comprehensive Language Model Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference only (default mode)
  een_eval run --config eval_config.yaml
  
  # Run full evaluation (inference + evaluation)
  een_eval run --config eval_config.yaml --mode full
  
  # Run evaluation only on existing predictions
  een_eval eval --config eval_config.yaml --output-dir results/eval_20231201_120000
  
  # Create sample configuration
  een_eval create-config --output sample_config.yaml --type basic
  
  # Validate configuration
  een_eval validate --config eval_config.yaml
  
  # Resume interrupted evaluation
  een_eval resume --output-dir results/eval_20231201_120000 --config eval_config.yaml
  
  # List available components
  een_eval list-components
        """
    )
    
    # Global arguments
    parser.add_argument("--log-level", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO",
                       help="Set logging level")
    parser.add_argument("--log-file", 
                       help="Log to file in addition to console")
    parser.add_argument("--verbose", "-v", 
                       action="store_true",
                       help="Enable verbose output")
      # Create shared parent parsers for common arguments
    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument("--config", "-c", 
                              help="Configuration file path")
    
    execution_parent = argparse.ArgumentParser(add_help=False)
    execution_parent.add_argument("--output-dir", "-o", 
                                 help="Output directory (overrides config)")
    execution_parent.add_argument("--batch-size", 
                                 type=int,
                                 help="Batch size (overrides config)")
    execution_parent.add_argument("--max-workers", 
                                 type=int,
                                 help="Maximum parallel processes (overrides config)")
    
    # Create subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", 
                                      parents=[config_parent, execution_parent],
                                      help="Run evaluation")
    run_parser.add_argument("--mode", 
                           choices=["inference", "evaluation", "full"], 
                           default="inference",
                           help="Evaluation mode")
    
    # Arguments for minimal config creation (only for run command)
    run_parser.add_argument("--dataset", 
                           help="Dataset file path (required if no config)")
    run_parser.add_argument("--input-field", 
                           help="Input field name in dataset")
    run_parser.add_argument("--expected-field", 
                           help="Expected output field name in dataset")
    run_parser.add_argument("--model-type", 
                           choices=["openai", "vllm"],
                           help="Model type")
    run_parser.add_argument("--model-name", 
                           help="Model name/identifier")
    run_parser.add_argument("--temperature", 
                           type=float,
                           help="Model temperature")
    run_parser.add_argument("--max-tokens", 
                           type=int,
                           help="Maximum tokens for model generation")
      # Eval command (shorthand for run --mode evaluation)
    eval_parser = subparsers.add_parser("eval", 
                                       parents=[config_parent],
                                       help="Run evaluation only (equivalent to run --mode evaluation)")
    eval_parser.add_argument("--output-dir", "-o", 
                            required=True,
                            help="Output directory containing predictions (required)")
    eval_parser.add_argument("--batch-size", 
                            type=int,
                            help="Batch size (overrides config)")
    eval_parser.add_argument("--max-workers", 
                            type=int,
                            help="Maximum parallel processes (overrides config)")
    
    # Resume command
    resume_parser = subparsers.add_parser("resume", 
                                         parents=[config_parent],
                                         help="Resume interrupted evaluation")
    resume_parser.add_argument("--output-dir", "-o", 
                              required=True,
                              help="Directory containing evaluation state")
    
    # Create config command
    config_parser = subparsers.add_parser("create-config", 
                                         help="Create sample configuration file")
    config_parser.add_argument("--output", "-o", 
                              required=True,
                              help="Output path for configuration file")
    config_parser.add_argument("--type", "-t", 
                              choices=["basic", "vllm"], 
                              default="basic",
                              help="Type of configuration to create")
    # Validate command
    validate_parser = subparsers.add_parser("validate", 
                                           help="Validate configuration file")
    validate_parser.add_argument("--config", "-c", 
                                required=True,
                                help="Configuration file to validate")
    
    # List components command
    subparsers.add_parser("list-components", 
                         help="List available evaluation methods and metrics")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)

    # Execute command
    if args.command == "run":
        return run_evaluation(args)
    elif args.command == "eval":
        # Set mode to evaluation for eval command
        args.mode = "evaluation"
        return run_evaluation(args)
    elif args.command == "create-config":
        try:
            create_sample_config(args.output, args.type)
            return 0
        except Exception as e:
            print(f"❌ Error creating configuration: {e}")
            return 1
    elif args.command == "validate":
        return validate_config_command(args)
    elif args.command == "resume":
        return resume_evaluation(args)
    elif args.command == "list-components":
        list_available_components()
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())